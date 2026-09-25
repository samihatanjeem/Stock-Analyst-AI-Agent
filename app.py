"""Stock Analyst AI Agent - single-file Streamlit app.

Everything lives here: the regression coefficients, the five tools, the
LangGraph agent with its cross-model fallback chain, the CSS/HTML for the
page, and the chat UI itself. The only things NOT in this file are secrets
(.streamlit/secrets.toml - Streamlit's own mechanism, and credentials belong
out of source control) and the research paper itself (never uploaded anywhere).

Run:      streamlit run app.py
Re-index: python index_paper.py   (only needed if the paper's content changes)
"""

import json
import os
import uuid
from datetime import date, timedelta

import streamlit as st
import yfinance as yf
from langchain.agents import create_agent
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

try:
    import finnhub
except ImportError:
    finnhub = None


# =============================================================================
# 1. Configuration and research data
# =============================================================================

# gemini-flash-lite-latest has its own free-tier quota bucket, separate from
# gemini-flash-latest (20 requests/DAY). One question costs one request per agent
# step, so the lite alias is the sane default for anything user-facing.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMS = 768

DB_NAME = "portfolio_research_agent"
COLLECTION_NAME = "capstone_paper_chunks"
VECTOR_INDEX_NAME = "paper_vector_index"
PAPER_PATH = "US Stocks Prediction Analysis_Research Paper.docx"

# The researcher's own pre-fitted OLS regression, reproduced exactly from the
# original capstone analysis (Tables 2 and 3). Inlined here rather than read
# from a JSON file so this file has no data dependency of its own at runtime.
STUDY = {
    "study": "Predicting Next-Year ROA from XBRL financial ratios (BUSA521 capstone)",
    "period": "2011-2013",
    "observations": 279,
    "companies": 182,
    "estimator": "OLS, ratios winsorized at the 1st/99th percentile",
    "size_split": {
        "basis": "TOTAL ASSETS terciles within the study sample (NOT market capitalisation)",
        "p33_usd": 43684020.0,
        "p67_usd": 392521213.82,
    },
    "scale": "All ratios are raw fractions (e.g. ROA of 5.6% = 0.056), never percentages.",
    "caveats": [
        "The sample skews heavily toward small, distressed firms: mean current ROA is "
        "-0.445 and mean profit margin is -2.006. Large, profitable companies sit at "
        "the extreme healthy edge of the fitted range.",
        "The 'large_firms' tercile means assets >= $392.5M within this sample - "
        "small-cap by today's standards. A mega-cap like AMD is orders of magnitude "
        "outside the fitted range.",
        "Residual standard error on the pooled model is 1.741 in ROA units, which is "
        "very large relative to a typical ROA. Treat any single prediction as "
        "illustrative only.",
    ],
}

MODELS = {
    "full": {
        "intercept": 0.285735,
        "coefficients": {
            "current_roa": 1.284091, "profit_margin": -0.03139, "debt_ratio": -0.622283,
            "asset_turnover": -0.02691, "ar_growth": -0.441245,
            "inventory_growth": -0.434843, "depreciation_to_assets": 2.313077,
        },
        "r_squared": 0.675, "adj_r_squared": 0.666, "residual_std_error": 1.741,
        "n_observations": 279,
        "description": "Table 2: pooled OLS across all 279 firm-year observations.",
    },
    "small_firms": {
        "intercept": 0.662111,
        "coefficients": {
            "current_roa": 1.222093, "profit_margin": -0.02891, "debt_ratio": -0.71979,
            "asset_turnover": -0.17084, "ar_growth": -0.697551,
            "inventory_growth": -1.178629, "depreciation_to_assets": 0.581127,
        },
        "r_squared": 0.649, "adj_r_squared": 0.619, "residual_std_error": 3.074,
        "n_observations": 92,
        "description": "Table 3 (Small): firms in the small tercile by TOTAL ASSETS "
                       "within the study sample (< $43.7M).",
    },
    "medium_firms": {
        "intercept": 0.054741,
        "coefficients": {
            "current_roa": 0.388507, "profit_margin": 0.135459, "debt_ratio": -0.387039,
            "asset_turnover": -0.008693, "ar_growth": 0.320374,
            "inventory_growth": 0.234543, "depreciation_to_assets": 2.183878,
        },
        "r_squared": 0.528, "adj_r_squared": 0.49, "residual_std_error": 0.261,
        "n_observations": 95,
        "description": "Table 3 (Medium): firms in the medium tercile by TOTAL ASSETS "
                       "within the study sample ($43.7M - $392.5M).",
    },
    "large_firms": {
        "intercept": -0.024198,
        "coefficients": {
            "current_roa": 0.265583, "profit_margin": 0.251397, "debt_ratio": 0.027131,
            "asset_turnover": 0.011744, "ar_growth": -0.048971,
            "inventory_growth": 0.705742, "depreciation_to_assets": 0.067276,
        },
        "r_squared": 0.539, "adj_r_squared": 0.501, "residual_std_error": 0.061,
        "n_observations": 92,
        "description": "Table 3 (Large): firms in the large tercile by TOTAL ASSETS "
                       "within the study sample (>= $392.5M).",
    },
}

# The study bucketed firms by TOTAL ASSETS terciles within its own sample -
# NOT by market cap. get_company_financials uses these to pick the right model.
SIZE_P33 = STUDY["size_split"]["p33_usd"]
SIZE_P67 = STUDY["size_split"]["p67_usd"]


# =============================================================================
# 2. Tools
# =============================================================================

def _predict_with_model(ratios: dict, model: dict) -> float:
    """Apply one fitted model's coefficients to a ratio dict."""
    total = model["intercept"]
    for var, coef in model["coefficients"].items():
        total += coef * ratios.get(var, 0.0)
    return total


@tool
def predict_roa(current_roa: float, profit_margin: float, debt_ratio: float,
                asset_turnover: float, ar_growth: float, inventory_growth: float,
                depreciation_to_assets: float, firm_size: str = "full") -> str:
    """Predict next-year ROA using the researcher's own pre-fitted OLS regression
    coefficients (from a 2011-2013 capstone study of 182 U.S. public companies).

    This is NOT a live-trained model and does NOT use any new regression fitting.
    firm_size must be one of: 'full', 'small_firms', 'medium_firms', 'large_firms'.
    Always treat the output as an ILLUSTRATIVE, historically-grounded estimate,
    never as a real forecast, and never as investment advice.
    """
    key = firm_size if firm_size in MODELS else "full"
    model = MODELS[key]
    ratios = {
        "current_roa": current_roa,
        "profit_margin": profit_margin,
        "debt_ratio": debt_ratio,
        "asset_turnover": asset_turnover,
        "ar_growth": ar_growth,
        "inventory_growth": inventory_growth,
        "depreciation_to_assets": depreciation_to_assets,
    }
    prediction = _predict_with_model(ratios, model)
    return (
        f"Model used: {key} (R²={model['r_squared']}, adj R²={model.get('adj_r_squared')}, "
        f"n={model['n_observations']}).\n"
        f"{model.get('description', '')}\n"
        f"Predicted next-year ROA (illustrative, based on a 2011-2013 pattern): {prediction:.4f}\n"
        f"Residual std error of this model: {model.get('residual_std_error')} in ROA units - "
        f"the uncertainty band around this point estimate is very wide.\n"
        f"This is NOT a real forecast and NOT investment advice - it applies an old "
        f"statistical pattern to a new company's current ratios."
    )


@tool
def get_company_financials(ticker: str) -> str:
    """Fetch a company's REAL, current financial data from Yahoo Finance and compute
    a full set of financial indicators: the 7 regression-input ratios from the
    researcher's capstone study (current ROA, profit margin, debt ratio, asset
    turnover, AR growth, inventory growth, depreciation-to-assets, using the exact
    formulas from the original R script) PLUS additional financial health indicators
    for a broader picture (ROE, current ratio, quick ratio, gross margin, revenue
    growth). Always use this tool to get real numbers, never estimate or recall
    financials from memory.
    """
    try:
        t = yf.Ticker(ticker)
        bs = t.balance_sheet          # balance sheet, most recent columns first
        fin = t.financials            # income statement
        cf = t.cashflow               # cash flow statement

        def latest(df, row, col=0):
            return float(df.loc[row].iloc[col]) if row in df.index else None

        def prior(df, row, col=1):
            return float(df.loc[row].iloc[col]) if row in df.index and df.shape[1] > col else None

        total_assets = latest(bs, "Total Assets")
        total_assets_prior = prior(bs, "Total Assets")
        total_liab = latest(bs, "Total Liabilities Net Minority Interest")
        equity = latest(bs, "Stockholders Equity")
        net_income = latest(fin, "Net Income")
        revenue = latest(fin, "Total Revenue")
        revenue_prior = prior(fin, "Total Revenue")
        gross_profit = latest(fin, "Gross Profit")
        ar = latest(bs, "Accounts Receivable")
        ar_prior = prior(bs, "Accounts Receivable")
        inventory = latest(bs, "Inventory")
        inventory_prior = prior(bs, "Inventory")
        depreciation = latest(cf, "Depreciation And Amortization")
        current_assets = latest(bs, "Current Assets")
        current_liab = latest(bs, "Current Liabilities")

        if not total_assets or not net_income:
            return f"Could not retrieve sufficient financial data for '{ticker}'. Try a different ticker."

        # The 7 ratios used in the original regression, matching the exact formulas
        # from the original R script (AR_Growth and Inv_Growth are scaled by PRIOR
        # TOTAL ASSETS, not by the prior AR/inventory value, per the source code).
        regression_ratios = {
            "current_roa": net_income / total_assets,
            "profit_margin": (net_income / revenue) if revenue else 0.0,
            "debt_ratio": (total_liab / total_assets) if total_liab else 0.0,
            "asset_turnover": (revenue / total_assets) if revenue else 0.0,
            "ar_growth": ((ar - ar_prior) / total_assets_prior) if ar and ar_prior and total_assets_prior else 0.0,
            "inventory_growth": ((inventory - inventory_prior) / total_assets_prior) if inventory and inventory_prior and total_assets_prior else 0.0,
            "depreciation_to_assets": (depreciation / total_assets) if depreciation else 0.0,
        }

        # Broader context - not part of the regression, but useful for a fuller analysis
        additional_indicators = {
            "return_on_equity": (net_income / equity) if equity else None,
            "current_ratio": (current_assets / current_liab) if current_assets and current_liab else None,
            "quick_ratio": ((current_assets - (inventory or 0)) / current_liab) if current_assets and current_liab else None,
            "gross_margin": (gross_profit / revenue) if gross_profit and revenue else None,
            "revenue_growth": ((revenue - revenue_prior) / revenue_prior) if revenue and revenue_prior else None,
        }

        # The study split firms into terciles by TOTAL ASSETS within its own sample
        # (p33 ~ $43.7M, p67 ~ $392.5M), NOT by market cap. Bucket the same way, so
        # predict_roa gets the model actually fitted on comparably sized firms.
        if total_assets < SIZE_P33:
            firm_size = "small_firms"
        elif total_assets < SIZE_P67:
            firm_size = "medium_firms"
        else:
            firm_size = "large_firms"

        scale_note = ""
        if total_assets > SIZE_P67 * 10:
            scale_note = (
                f"\n\nSCALE WARNING: total assets are ${total_assets:,.0f}, far beyond the "
                f"firms this study was fitted on - its 'large' tercile begins at only "
                f"${SIZE_P67:,.0f}. A predict_roa result for this company is a deep "
                f"out-of-sample extrapolation and MUST be labelled as such."
            )

        return (
            f"Real current data for {ticker.upper()} (source: Yahoo Finance):\n\n"
            f"Regression-input ratios (for predict_roa):\n"
            f"{json.dumps(regression_ratios, indent=2)}\n\n"
            f"Additional financial health indicators (broader context):\n"
            f"{json.dumps(additional_indicators, indent=2)}\n\n"
            f"Total assets: ${total_assets:,.0f}\n"
            f"Suggested firm_size bucket for predict_roa: {firm_size}\n"
            f"(bucket chosen by total assets, matching the study's own tercile split)"
            f"{scale_note}"
        )
    except Exception as e:
        return f"Error fetching data for '{ticker}': {e}"


_ddg = DuckDuckGoSearchRun()


@tool
def search_recent_news(query: str) -> str:
    """Search the web for recent news, risks, or context about a company.
    Use this for anything time-sensitive that a 2011-2013 study or a static
    financial-statement snapshot wouldn't capture.
    """
    return _ddg.run(query)


def _build_vector_store():
    """Return a vector store over the indexed paper, or None if Atlas isn't set up."""
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return None
    collection = MongoClient(uri)[DB_NAME][COLLECTION_NAME]
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBEDDING_DIMS,  # 768 instead of the 3072 default
    )
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
    )


_vector_store = None


@tool
def search_my_research(query: str) -> str:
    """Search the researcher's own capstone paper for relevant methodology, findings,
    or caveats - e.g. why debt ratio matters, what the R² values mean, or limitations
    of the original 2011-2013 study. Use this to explain the reasoning behind a prediction.
    """
    if _vector_store is None:
        return ("The research paper index is not configured (no MONGODB_URI), so the "
                "paper cannot be quoted. Say so rather than paraphrasing the study "
                "from memory.")
    results = _vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant passages found in the research paper."
    return "\n\n---\n\n".join(r.page_content for r in results)


_finnhub_client = None


@tool
def get_market_snapshot(ticker: str) -> str:
    """Fetch REAL-TIME market data for a ticker from Finnhub: current price and
    intraday move, company profile (industry, exchange, market cap), analyst
    recommendation trends, and company-specific news headlines from the last 7 days.

    Use this for anything about today's price, the market's recent reaction, or
    fresh company news. This does NOT replace get_company_financials - that tool
    stays the only source for financial-statement ratios. Never state a price or a
    headline from memory; call this tool.
    """
    if _finnhub_client is None:
        return (
            "Finnhub is not configured (no FINNHUB_API_KEY set), so real-time market "
            "data is unavailable. Use get_company_financials for statement-based "
            "ratios and search_recent_news for web context instead."
        )

    symbol = ticker.upper()
    parts = [f"Real-time market data for {symbol} (source: Finnhub):"]

    try:
        q = _finnhub_client.quote(symbol)
        if q and q.get("c"):
            parts.append(
                "\nQuote:\n"
                f"  current_price: {q.get('c')}\n"
                f"  change: {q.get('d')} ({q.get('dp')}%)\n"
                f"  day_open: {q.get('o')}  day_high: {q.get('h')}  day_low: {q.get('l')}\n"
                f"  previous_close: {q.get('pc')}"
            )
        else:
            parts.append(f"\nQuote: no live quote returned for '{symbol}' (check the ticker).")
    except Exception as e:
        parts.append(f"\nQuote: unavailable ({e})")

    try:
        p = _finnhub_client.company_profile2(symbol=symbol)
        if p:
            parts.append(
                "\nProfile:\n"
                f"  name: {p.get('name')}\n"
                f"  industry: {p.get('finnhubIndustry')}\n"
                f"  exchange: {p.get('exchange')}\n"
                f"  market_cap_usd_millions: {p.get('marketCapitalization')}\n"
                f"  shares_outstanding_millions: {p.get('shareOutstanding')}"
            )
    except Exception as e:
        parts.append(f"\nProfile: unavailable ({e})")

    try:
        recs = _finnhub_client.recommendation_trends(symbol)
        if recs:
            r = recs[0]
            parts.append(
                f"\nAnalyst recommendation trends ({r.get('period')}):\n"
                f"  strongBuy: {r.get('strongBuy')}  buy: {r.get('buy')}  hold: {r.get('hold')}  "
                f"sell: {r.get('sell')}  strongSell: {r.get('strongSell')}\n"
                "  (These are other analysts' opinions, reported here as data. Do NOT adopt "
                "them as your own recommendation.)"
            )
    except Exception as e:
        parts.append(f"\nAnalyst recommendation trends: unavailable ({e})")

    try:
        today = date.today()
        news = _finnhub_client.company_news(
            symbol,
            _from=(today - timedelta(days=7)).isoformat(),
            to=today.isoformat(),
        )
        if news:
            lines = ["\nCompany news (last 7 days, most recent first):"]
            for item in news[:5]:
                headline = (item.get("headline") or "").strip()
                summary = (item.get("summary") or "").strip().replace("\n", " ")
                if len(summary) > 240:
                    summary = summary[:240] + "..."
                lines.append(f"  - [{item.get('source')}] {headline}\n    {summary}")
            parts.append("\n".join(lines))
        else:
            parts.append("\nCompany news (last 7 days): none returned.")
    except Exception as e:
        parts.append(f"\nCompany news: unavailable ({e})")

    return "\n".join(parts)


# =============================================================================
# 3. Agent assembly (with cross-model fallback)
# =============================================================================

SYSTEM_PROMPT = """You are a Stock Analyst AI Agent. You give a broad financial \
analysis of a company using multiple indicators (profitability, liquidity, \
leverage, efficiency, and growth), not just a single predicted number. You are \
also built on top of the user's own capstone research (an OLS regression study \
of 279 firm-year observations, 182 U.S. public companies, 2011-2013, predicting \
next-year ROA), which is one input among several you draw on.

Rules you must always follow:
1. NEVER state a company's financial numbers from memory. Always call \
   get_company_financials first for any real company data, and use BOTH the \
   regression-input ratios and the additional financial health indicators it \
   returns, not just the ROA-related ones.
2. NEVER refit or invent a new regression. Only use predict_roa, which applies the \
   user's already-fitted coefficients. Treat predict_roa as one specific, \
   research-grounded estimate, not the whole analysis.
3. When explaining WHY a factor matters, use search_my_research to ground your \
   explanation in the actual paper rather than general knowledge. If that tool is \
   not in your toolset, the paper index simply isn't configured - say that you can't \
   quote the paper instead of recalling its contents from memory.
4. Use search_recent_news when the question involves anything current (recent \
   performance, news, risk).
5. If get_market_snapshot is available, use it for anything about today's price, \
   the market's recent reaction, or fresh company-specific headlines. It does not \
   replace get_company_financials - statement-level ratios still come only from \
   that tool. Never state a live price from memory. If the tool is not in your \
   toolset, Finnhub simply isn't configured; fall back on search_recent_news and \
   say you don't have a live quote rather than guessing one.
6. If get_market_snapshot returns analyst recommendation trends, report them as \
   other analysts' opinions - data about the market, not your own recommendation.
7. ALWAYS clearly label any predict_roa output as an illustrative estimate based on \
   a 2011-2013 pattern, not a real forecast. The study's sample skewed small and \
   distressed (mean current ROA -0.445), and its 'large firms' tercile begins at only \
   ~$392M in total assets. If get_company_financials returns a SCALE WARNING, say \
   plainly that the company is far outside the range the model was fitted on, and \
   lean on the other indicators instead of the predicted number.
8. NEVER tell the user whether to buy or sell. Present the full financial picture \
   (regression estimate, other indicators, research context, live market data, and \
   recent news), and explicitly leave the decision to them.
"""

# Friendly labels for the "working on it" status line in the UI.
TOOL_LABELS = {
    "get_company_financials": "Pulling financial statements",
    "predict_roa": "Running your regression model",
    "search_recent_news": "Searching recent news",
    "search_my_research": "Consulting your capstone paper",
    "get_market_snapshot": "Fetching live market data",
}

# Tried in order for each question. Google's free tier regularly puts one alias
# under heavy load (slow or outright 503) while a sibling responds normally, and
# a per-DAY quota is a separate bucket per model - so falling through to the next
# name costs a few seconds, not minutes, and often succeeds outright.
FALLBACK_MODELS = list(dict.fromkeys(
    [GEMINI_MODEL, "gemini-flash-latest", "gemini-3.5-flash-lite"]
))


def _make_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        # Defaults are max_retries=6 with no timeout - a single congested call can
        # silently retry for minutes before surfacing anything to the user. A short
        # ceiling here means a stuck model is abandoned in seconds, not minutes, and
        # stream_answer() moves on to the next model in FALLBACK_MODELS.
        max_retries=0,
        timeout=15,
    )


def build_agent():
    """Build one agent per model in FALLBACK_MODELS, sharing the same toolset.

    Returns (agents, tool_names) where agents is an ordered dict of
    {model_name: agent}. Tools that need an absent credential are simply left
    out. Building N agents is cheap - it wires up local objects only, no
    network call happens until a question is actually asked.
    """
    global _vector_store, _finnhub_client

    rag_enabled = bool(os.environ.get("MONGODB_URI"))
    finnhub_enabled = bool(os.environ.get("FINNHUB_API_KEY")) and finnhub is not None

    _vector_store = _build_vector_store() if rag_enabled else None
    _finnhub_client = (
        finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"]) if finnhub_enabled else None
    )

    tools = [get_company_financials, predict_roa, search_recent_news]
    if rag_enabled:
        tools.append(search_my_research)
    if finnhub_enabled:
        tools.append(get_market_snapshot)

    agents = {
        model: create_agent(_make_llm(model), tools, system_prompt=SYSTEM_PROMPT)
        for model in FALLBACK_MODELS
    }
    return agents, [t.name for t in tools]


def _as_text(message) -> str:
    """Newer Gemini models return content as a list of typed blocks rather than a
    plain string. Flatten either shape down to readable text."""
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


class QuotaExhausted(RuntimeError):
    """Raised when every model in FALLBACK_MODELS failed - out of quota, timed
    out, or otherwise unreachable. The underlying cause is reported in the
    message rather than assumed."""


def stream_answer(agents: dict, question: str, history=None):
    """Run the agent, yielding ('model', name), ('tool', (name, args)), and ('final', text).

    `agents` is the {model_name: agent} dict from build_agent(). Google's free
    tier regularly has one model alias slow or 503 while a sibling answers
    normally, so each model gets exactly one attempt (bounded by that agent's
    own client timeout) before moving to the next - a stuck model costs
    seconds, not minutes. A per-DAY quota is a separate bucket per model, so
    it is treated the same way: skip to the next model, don't wait it out.

    Any exception counts as "this model didn't answer" and moves to the next
    one - Google's free tier fails in too many shapes (GoogleAPIError with
    503/504 text, a bare requests.ReadTimeout with no such text, a connection
    error) to whitelist safely, and every case here means the same thing.
    """
    messages = list(history or []) + [HumanMessage(content=question)]
    last_error = None

    for model, agent in agents.items():
        yield "model", model
        try:
            final_text = ""
            for update in agent.stream({"messages": messages}, stream_mode="updates"):
                for node_output in update.values():
                    for msg in node_output.get("messages", []) or []:
                        for call in getattr(msg, "tool_calls", []) or []:
                            yield "tool", (call["name"], call.get("args") or {})
                        text = _as_text(msg)
                        if text and getattr(msg, "type", "") == "ai":
                            final_text = text
            yield "final", final_text
            return
        except Exception as e:
            last_error = e
            continue

    raise QuotaExhausted(
        f"None of the configured models ({', '.join(agents)}) answered - each is "
        "either over its free-tier quota, timed out, or currently unavailable. "
        f"Last error from {list(agents)[-1]}: {type(last_error).__name__}: "
        f"{str(last_error)[:200]}"
    ) from last_error


def index_paper(path: str = PAPER_PATH) -> str:
    """Chunk the paper, embed it, store it in Atlas, and create the vector index.

    Run once (or whenever the paper changes) via `python index_paper.py`. The
    deployed app never calls this - it reads chunks that are already in Atlas.
    Safe to import without triggering the Streamlit UI below, since that UI
    only runs under `if __name__ == "__main__"`.
    """
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return "MONGODB_URI is not set, so there is nowhere to index into. Skipped."

    collection = MongoClient(uri)[DB_NAME][COLLECTION_NAME]
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIMS
    )

    docs = Docx2txtLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    collection.delete_many({})  # clear old runs so re-indexing doesn't duplicate
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_INDEX_NAME,
    )

    # The collection has to exist before a search index can be attached to it,
    # which is why this runs after the documents are written, not before.
    if VECTOR_INDEX_NAME not in [i["name"] for i in collection.list_search_indexes()]:
        collection.create_search_index(SearchIndexModel(
            definition={"fields": [{
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIMS,
                "similarity": "cosine",
            }]},
            name=VECTOR_INDEX_NAME,
            type="vectorSearch",
        ))

    return f"Indexed {len(chunks)} chunks into MongoDB Atlas."


# =============================================================================
# 4. Presentation - CSS and small HTML helpers
#
# Every selector below is a `data-testid` or a public class, the stable ones
# across Streamlit releases - if one ever stops matching, the page degrades to
# default styling rather than breaking.
# =============================================================================

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root {
    --ink:        #1C2B2D;
    --ink-soft:   #5B6B6D;
    --ink-faint:  #8A9A9B;
    --accent:     #0F766E;
    --accent-dim: #E6F2F0;
    --line:       #E2E8E7;
    --surface:    #FFFFFF;
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 7rem;
    max-width: 48rem;
}

header[data-testid="stHeader"] { background: transparent; height: 0; }

.masthead {
    border-bottom: 1px solid var(--line);
    padding-bottom: 1.1rem;
    margin-bottom: 1.6rem;
}
.masthead h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 500;
    letter-spacing: -0.015em;
    color: var(--ink);
    margin: 0 0 0.3rem 0;
    padding: 0;
}
.masthead p {
    color: var(--ink-soft);
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.5;
}
.masthead .rule {
    display: inline-block;
    width: 2.2rem;
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    margin-bottom: 0.9rem;
}

.strip {
    display: flex;
    gap: 1.4rem;
    flex-wrap: wrap;
    font-size: 0.78rem;
    color: var(--ink-faint);
    margin-top: 0.9rem;
}
.strip b { color: var(--ink-soft); font-weight: 600; }
.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    margin-right: 0.4rem;
    vertical-align: middle;
}

[data-testid="stChatMessage"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 2px rgba(16,32,32,0.04);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--accent-dim);
    border-color: transparent;
    box-shadow: none;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }
[data-testid="stChatMessage"] h3 {
    font-size: 1.05rem;
    margin-top: 1.3rem;
    color: var(--ink);
}
[data-testid="stChatMessage"] hr { margin: 1.1rem 0; border-color: var(--line); }

.pills { margin-top: 0.85rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }
.pill {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--accent);
    background: var(--accent-dim);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    white-space: nowrap;
}
.pill-label {
    font-size: 0.72rem;
    color: var(--ink-faint);
    align-self: center;
    margin-right: 0.15rem;
}

.prompt-head {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-faint);
    margin: 0.5rem 0 0.7rem 0;
}
div[data-testid="stButton"] > button {
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    height: 100%;
    min-height: 4.2rem;
    line-height: 1.4;
    font-size: 0.86rem;
    color: var(--ink);
    background: var(--surface);
    border: 1px solid var(--line);
    padding: 0.8rem 0.95rem;
    transition: border-color .15s ease, transform .15s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: var(--accent);
    color: var(--accent);
    transform: translateY(-1px);
}

[data-testid="stChatInput"] {
    border-radius: 14px;
    border-color: var(--line);
}
[data-testid="stBottomBlockContainer"] { padding-bottom: 1.4rem; }

[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] h2 {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ink-faint);
    font-weight: 600;
}
[data-testid="stSidebar"] .sb-brand {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.15rem;
    color: var(--ink);
    margin-bottom: 0.15rem;
}
.source-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.34rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
}
.source-row span:first-child { color: var(--ink); }
.source-row span:last-child  { color: var(--ink-faint); font-size: 0.75rem; }

.disclaimer {
    font-size: 0.73rem;
    line-height: 1.5;
    color: var(--ink-faint);
    border-left: 2px solid var(--line);
    padding-left: 0.7rem;
}
</style>
"""


def _masthead_html(model: str, tool_count: int) -> str:
    return f"""
    <div class="masthead">
        <div class="rule"></div>
        <h1>Stock Analyst</h1>
        <p>Company analysis grounded in live market data and an OLS regression
           model fitted on 182 U.S. public companies.</p>
        <div class="strip">
            <span><span class="dot"></span><b>{tool_count}</b> data tools active</span>
            <span>Model <b>{model}</b></span>
            <span>Yahoo Finance · Finnhub · Atlas</span>
        </div>
    </div>
    """


def _pills_html(names: list[str], labels: dict) -> str:
    if not names:
        return ""
    chips = "".join(f'<span class="pill">{labels.get(n, n)}</span>' for n in names)
    return f'<div class="pills"><span class="pill-label">Sources consulted</span>{chips}</div>'


def _source_row_html(name: str, detail: str) -> str:
    return f'<div class="source-row"><span>{name}</span><span>{detail}</span></div>'


# Yahoo Finance's exchange codes (from yf.Ticker(...).fast_info["exchange"]),
# mapped to the prefixes TradingView expects. Covers the exchanges essentially
# all US-listed companies trade on; anything unrecognized falls back to a bare
# ticker rather than guessing wrong.
_YAHOO_EXCHANGE_TO_TRADINGVIEW = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",  # Nasdaq tiers
    "NYQ": "NYSE",
    "ASE": "AMEX",
    "PCX": "AMEX",  # NYSE Arca
}


def _resolve_chart_symbol(ticker: str) -> str:
    """Qualify a bare ticker with its exchange for TradingView, e.g. "ARM" ->
    "NASDAQ:ARM". A bare symbol can resolve ambiguously on TradingView - ARM
    Holdings in particular also has UK listing history, and a plain "ARM"
    can show the wrong instrument entirely. Falls back to the bare ticker
    (today's prior behavior) if the lookup fails or the exchange isn't one we
    recognize - never worse than not qualifying it at all.
    """
    try:
        exchange = yf.Ticker(ticker).fast_info.get("exchange")
        prefix = _YAHOO_EXCHANGE_TO_TRADINGVIEW.get(exchange)
        if prefix:
            return f"{prefix}:{ticker}"
    except Exception:
        pass
    return ticker


def _tradingview_widget_html(symbol: str, dark: bool = False) -> str:
    """An embedded TradingView 'Advanced Chart' widget - free, no API key, the
    same interactive candlestick chart (zoom, timeframes, live updates during
    market hours) you'd see on a Google search result for a ticker.

    Genuine real-time direct-from-exchange data is a paid, licensed product -
    not something free for a hobby app, and not something Nasdaq's own public
    site shows either (their public quotes are delayed same as everyone
    else's free tier). This widget is the practical equivalent: same look,
    same behavior, backed by TradingView's own licensed data.

    A bare ticker (no "NASDAQ:" prefix) is intentional - TradingView resolves
    it to the symbol's primary listing, which covers the vast majority of
    well-known US tickers without needing to know the exchange in advance.
    """
    # This string is interpolated straight into a <script> block. The ticker
    # ultimately comes from the LLM's tool-call arguments, which in turn can
    # be steered by whatever a viewer typed into chat - strip it to the
    # characters a real symbol can contain (letters, digits, '.', ':') before
    # it ever reaches the template, closing off script injection.
    symbol = "".join(c for c in symbol.upper() if c.isalnum() or c in ".:") or "NVDA"
    theme = "dark" if dark else "light"
    toolbar_bg = "#1C2B2D" if dark else "#F1F5F4"
    # A unique element/container id per chart - multiple charts render on the
    # same page (one per past answer), and TradingView's widget script can
    # misbehave if instances share an id, even across separate iframes.
    container_id = f"tv_{symbol.replace('.', '_').replace(':', '_')}_{uuid.uuid4().hex[:8]}"
    return f"""
    <div class="tradingview-widget-container">
      <div id="{container_id}"></div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
      new TradingView.widget({{
        "width": "100%",
        "height": 440,
        "symbol": "{symbol}",
        "interval": "D",
        "timezone": "Etc/UTC",
        "theme": "{theme}",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "{toolbar_bg}",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "hide_side_toolbar": true,
        "container_id": "{container_id}"
      }});
      </script>
    </div>
    """


# =============================================================================
# 5. Streamlit UI
#
# Guarded by __main__ so this section only runs under `streamlit run app.py`,
# not when index_paper.py does `from app import index_paper`.
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Stock Analyst",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Material icons, not emoji: st.chat_message only accepts a real emoji, an
    # image, or a ":material/..." name.
    avatar_ai = ":material/query_stats:"
    avatar_user = ":material/person:"

    # -- Credentials ---------------------------------------------------------
    # Streamlit keeps secrets in st.secrets; copy them into os.environ so the
    # tool functions above (which read os.environ, and also work from a plain
    # script) see them the same way regardless of how this app is launched.
    missing = []
    for key in ("GOOGLE_API_KEY", "MONGODB_URI", "FINNHUB_API_KEY"):
        value = st.secrets.get(key) or os.environ.get(key)
        if value:
            os.environ[key] = value
        elif key == "GOOGLE_API_KEY":
            missing.append(key)  # only Gemini is truly required
    if missing:
        st.error(
            f"Missing required secret: {', '.join(missing)}.\n\n"
            "Add it to `.streamlit/secrets.toml` locally, or to **Settings → Secrets** "
            "on Streamlit Cloud."
        )
        st.stop()

    # No password gate - this app is open to anyone with the link. See README
    # for how to add one back if the link ever needs to be restricted.

    # -- Agent (built once per server process, not once per message) --------
    @st.cache_resource(show_spinner="Starting the agent…")
    def _get_agent():
        return build_agent()

    agents, tool_names = _get_agent()

    # -- Main column ----------------------------------------------------------
    st.markdown(_masthead_html(GEMINI_MODEL, len(tool_names)), unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not st.session_state["messages"]:
        examples = [
            ("Should I buy Tesla this month?",
             "Full financial picture — watch it decline to say buy or sell"),
            ("Compare NVIDIA and AMD on profitability and growth",
             "Two real-time statement pulls, reasoned side by side"),
            ("What's Amazon's stock doing today, and this week's news?",
             "Live price, analyst sentiment, and fresh headlines"),
            ("How is ARM Holdings performing as a public company?",
             "Full breakdown: fundamentals, live price, and recent news"),
        ]
        st.markdown('<div class="prompt-head">Start with</div>', unsafe_allow_html=True)
        for row in (examples[:2], examples[2:]):
            for column, (text, hint) in zip(st.columns(2), row):
                if column.button(f"{text}\n\n{hint}", use_container_width=True, key=text):
                    st.session_state["pending"] = text
                    st.rerun()

    for message in st.session_state["messages"]:
        # Derived from the role, never read back from the stored message - a
        # session from before an avatar change would otherwise replay a stale value.
        avatar = avatar_user if message["role"] == "user" else avatar_ai
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("tools"):
                st.markdown(_pills_html(message["tools"], TOOL_LABELS), unsafe_allow_html=True)
            if message.get("ticker"):
                st.caption(f"Live chart — {message['ticker'].split(':')[-1]}")
                st.iframe(_tradingview_widget_html(message["ticker"]), height=420)

    question = st.chat_input("Ask about any US-listed company…") or st.session_state.pop("pending", None)

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user", avatar=avatar_user):
            st.markdown(question)

        with st.chat_message("assistant", avatar=avatar_ai):
            status = st.status("Reading the question…", expanded=True)
            used: list[str] = []
            tried_models: list[str] = []
            answer = ""
            # First ticker looked up via a company-specific tool this turn - used
            # to sync the sidebar chart to whatever was actually discussed.
            queried_ticker = None

            try:
                for kind, payload in stream_answer(agents, question):
                    if kind == "model":
                        tried_models.append(payload)
                        if len(tried_models) > 1:
                            status.write(
                                f"⚠ {tried_models[-2]} was slow or unavailable — trying {payload}"
                            )
                        status.update(label=f"Thinking ({payload})…")
                    elif kind == "tool":
                        name, args = payload
                        if name not in used:
                            used.append(name)
                        if queried_ticker is None and name in (
                            "get_company_financials", "get_market_snapshot"
                        ):
                            ticker = (args.get("ticker") or "").strip().upper()
                            if ticker:
                                # Resolved once here (exchange-qualified, e.g.
                                # "NASDAQ:ARM") and reused as-is on every later
                                # replay of this message - never re-resolved.
                                queried_ticker = _resolve_chart_symbol(ticker)
                        label = TOOL_LABELS.get(name, name)
                        status.write(f"→ {label}")
                        status.update(label=label)
                    elif kind == "final":
                        answer = payload

                status.update(
                    label=f"Analysis complete · {len(used)} source(s)",
                    state="complete",
                    expanded=False,
                )
                st.markdown(answer)
                if used:
                    st.markdown(_pills_html(used, TOOL_LABELS), unsafe_allow_html=True)
                if queried_ticker:
                    st.caption(f"Live chart — {queried_ticker.split(':')[-1]}")
                    st.iframe(_tradingview_widget_html(queried_ticker), height=420)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "tools": used, "ticker": queried_ticker}
                )

            except QuotaExhausted as e:
                status.update(label="No model responded", state="error", expanded=False)
                st.warning(str(e))
            except Exception as e:
                status.update(label="Something went wrong", state="error", expanded=False)
                st.error(f"{type(e).__name__}: {e}")

    # -- Sidebar ----------------------------------------------------------
    sources = [
        ("Yahoo Finance", "Statements"),
        ("Finnhub", "Live quotes"),
        ("DuckDuckGo", "Web context"),
        ("Capstone paper", "Methodology"),
    ]

    with st.sidebar:
        st.markdown('<div class="sb-brand">Stock Analyst</div>', unsafe_allow_html=True)
        st.caption("Research-grounded equity analysis")

        st.divider()
        st.subheader("Data sources")
        st.markdown(
            "".join(_source_row_html(name, detail) for name, detail in sources),
            unsafe_allow_html=True,
        )

        st.subheader("Model")
        st.markdown(
            _source_row_html("Reasoning", GEMINI_MODEL)
            + _source_row_html("Regression", "OLS · n=279 · 2011–13")
            + _source_row_html("Tools active", str(len(tool_names))),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Falls back to {', '.join(list(agents)[1:])} if {GEMINI_MODEL} is slow "
            "or over its free-tier quota."
        )

        with st.expander("Tool detail"):
            for name in tool_names:
                st.markdown(f"**{TOOL_LABELS.get(name, name)}**  \n`{name}`")

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown(
            '<div class="disclaimer"><b>Not investment advice.</b> Figures are '
            "retrieved live at query time. Regression output is an illustrative "
            "estimate from a 2011–2013 pattern, not a forecast.</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
