# 📊 Stock Analyst AI Agent

> Teaching machines to read and analyze market stocks so I don't have to!

**🔗 Live app: [stock-analyst-ai-agent.streamlit.app](https://stock-analyst-ai-agent.streamlit.app/)** — no login, no signup, just ask it about any company.

---

## What this actually does

You ask a question about a public US stock company. Behind the scenes, a [LangGraph](https://github.com/langchain-ai/langgraph) agent decides — on its own, per question — which of five tools it needs: pull real financial statements, check a live quote, search recent news, consult my own capstone research, or run my regression model. It stitches the results into one answer, in the time it'd take you to open four browser tabs and still not read any of them.

It will not tell you to buy or sell anything. It has strong opinions about debt ratios; it has no opinions about your portfolio.

## Why it exists

I ran an OLS regression for my BUSA521 Masters Capstone Project, using live data from the U.S. Securities and Exchange Commission (SEC) — 279 firm-year observations, 182 U.S. public companies, predicting next-year ROA from seven financial ratios. A fitted model sitting in a `.csv` is not a very interesting thing to show anyone. An agent that applies that same model to whatever company you're actually curious about *today*, while being upfront about exactly how far outside its training data that company sits, is a much better demo of the same statistics.

## How it's built

```
your question
     │
     ▼
 LangGraph agent (Gemini) ── decides which tools it needs, if any
     │
     ├── get_company_financials  → Yahoo Finance statements, live
     ├── predict_roa              → my capstone's fitted OLS coefficients
     ├── search_recent_news       → DuckDuckGo
     ├── search_my_research       → capstone paper, embedded in MongoDB Atlas
     └── get_market_snapshot      → Finnhub: quote, analyst trends, headlines
     │
     ▼
 one grounded answer + a live TradingView chart for whatever you asked about
```

| Tool | Source | Needs |
|---|---|---|
| `get_company_financials` | Yahoo Finance | – |
| `predict_roa` | My fitted regression (inlined, no external file) | – |
| `search_recent_news` | DuckDuckGo | – |
| `search_my_research` | Capstone paper, vector-indexed in Atlas | `MONGODB_URI` |
| `get_market_snapshot` | Finnhub live quotes + news | `FINNHUB_API_KEY` |

Whichever tool is missing its credential just quietly drops out of the toolset — the agent is told to say so rather than improvise a number.

## Engineering details worth mentioning

- **It doesn't panic when Google does.** Free-tier Gemini occasionally puts one model alias under heavy load while its siblings answer fine. Every question tries `gemini-flash-lite-latest` → `gemini-flash-latest` → `gemini-3.5-flash-lite` in order, abandoning a stuck model after ~15 seconds instead of hanging for minutes. If all three are down at once, it says so plainly instead of pretending everything's fine.
- **It knows when its own model doesn't apply.** The regression was fitted on companies where "large" tops out around $392M in assets. Ask it about a mega-cap and it flags the estimate as a serious extrapolation rather than quietly handing you a confident-sounding number.
- **The chart actually matches the stock.** A live TradingView chart renders under each answer — exchange-qualified (`NASDAQ:ARM`, not just `ARM`), because a bare ticker can resolve to the wrong instrument entirely, and I found that out the hard way.
- **It's one file.** Every tool, the regression coefficients, the agent, the styling, and the chat UI live in [`app.py`](app.py) — nothing to go hunting through five modules to understand.

## Stack

`LangGraph` · `LangChain` · `Google Gemini` (flash-lite, with fallback) · `MongoDB Atlas` (vector search) · `yfinance` · `Finnhub` · `DuckDuckGo` · `Streamlit` · `TradingView` embed

## Running it yourself

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your own keys
streamlit run app.py
```

Opens at `http://localhost:8501`. Only `GOOGLE_API_KEY` is required — the Atlas and Finnhub tools are optional and the app runs fine without them, just with a smaller toolset.

There's also a [notebook](stock_analyst_agent.ipynb) with the same agent broken into cells, for poking at individual tools without the UI in the way.

## The fine print

Every number in an answer comes from a live tool call, never from the model's own memory. Every regression output is labeled as an illustrative estimate from a 2011–2013 pattern, not a forecast. It will describe a company's financial health from five different angles and then, deliberately, decline to have an opinion about whether you should buy it.

---

Built by **Samiha Tanjeem**
