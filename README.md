# Stock Analyst AI Agent

A single-file Streamlit app over a LangGraph agent that analyses US-listed
companies using live financial data plus an OLS regression model from a
2011-2013 capstone study.

Everything - the five tools, the regression coefficients, the agent, the CSS,
and the chat UI - lives in [app.py](app.py). The only things outside it are
credentials (`.streamlit/secrets.toml`, gitignored) and `index_paper.py`, a
one-off script for re-indexing the research paper.

## Running locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill it in
streamlit run app.py
```

Opens at http://localhost:8501. `streamlit run` only serves the app while
that terminal stays open - closing it, or pressing Ctrl+C, stops the server
and the URL stops responding.

**No password by default** - anyone with the link can use the app and spend
your Gemini/Finnhub free-tier quota. To gate it, uncomment `APP_PASSWORD` in
`.streamlit/secrets.toml` and add back a password check at the top of
`main()` in `app.py` (removed intentionally; ask if you want it restored).

## Deploying (free, shareable link)

1. Push this folder to a **private** GitHub repo.
   `.streamlit/secrets.toml` and `.env` are gitignored - confirm they are not in
   the commit before pushing.
2. Go to https://share.streamlit.io → **New app** → pick the repo → main file `app.py`.
3. Open **Advanced settings → Secrets** and paste the contents of your local
   `secrets.toml`.
4. Deploy. You get a `*.streamlit.app` URL that stays up independent of your
   laptop.

## The tools

| Tool | Source | Needs |
|---|---|---|
| `get_company_financials` | Yahoo Finance statements | - |
| `predict_roa` | Regression coefficients (inlined in `app.py`) | - |
| `search_recent_news` | DuckDuckGo | - |
| `search_my_research` | Capstone paper in Atlas | `MONGODB_URI` |
| `get_market_snapshot` | Finnhub live quotes/news | `FINNHUB_API_KEY` |

Tools whose credential is absent are dropped from the agent at startup; the
agent is told to say so rather than answer from memory.

## Model fallback

Google's free tier regularly puts one Gemini alias under heavy load (slow or
outright `503`) while a sibling responds normally. Each question tries, in
order: `gemini-flash-lite-latest` → `gemini-flash-latest` →
`gemini-3.5-flash-lite`. A stuck model is abandoned after ~15s, not minutes,
and the status panel shows a note when a fallback kicks in. If all three fail
- a real, if uncommon, possibility during a widespread outage - the app
reports which models it tried and the last error, rather than hanging.

## Re-indexing the paper

Only needed if the .docx changes. Requires the paper and a `.env` in the folder:

```bash
python index_paper.py
```

This re-chunks, re-embeds, and rebuilds the Atlas vector index. The deployed app
never runs it - it reads chunks already in Atlas, so the paper itself is never
uploaded to the host.

## Notes

- **Model quota.** `gemini-flash-lite-latest` is the default because its free-tier
  bucket is separate from `gemini-flash-latest` (20 requests/day). One question
  costs one request per agent step, so a 3-tool answer costs ~4 - across
  whichever model in the fallback chain ends up answering.
- **Latency.** 15-45s per question is normal; the status panel shows which tool
  is running, and which model if a fallback happened.
- **Not investment advice.** The agent is instructed never to recommend buying or
  selling, and to label regression output as an illustrative estimate.
