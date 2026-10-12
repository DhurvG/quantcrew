# QuantCrew

An adaptive multi-agent system in which five AI agents research a Nifty 50 stock together and
an Orchestrator turns their debate into a BUY / HOLD / SELL recommendation with a full reasoning trail.

> **Advisory only.** QuantCrew is an academic decision-support prototype. It is not investment
> advice and it never places trades.

## The crew

| Agent | Role | Tools it can choose from |
|---|---|---|
| Orchestrator | Plans which specialists to activate, retries failures, runs the debate, makes the trust-weighted final call | LLM planner, trust store |
| Fundamental Analyst | Valuation, profitability, leverage, growth | `get_fundamentals` |
| Technical Analyst | Trend, momentum, 52-week position | `get_technical_indicators`, `get_price_window` |
| News Sentiment | Tone and materiality of recent headlines | `get_recent_headlines` |
| Risk Analyst | Volatility, drawdown, VaR/CVaR, beta vs Nifty | `get_risk_metrics`, `get_benchmark_risk` |
| Critic | Fact-checks claims against the data and challenges weak reasoning | technical, risk and fundamental tools |

Full architecture, orchestration pattern and trust-weight method: [`docs/DESIGN.md`](docs/DESIGN.md).
Input/output schemas: [`docs/IO.md`](docs/IO.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then put your API keys in .env
```

`QUANTCREW_PROFILE=groq` uses Groq's free tier (development); `claude` uses Claude Haiku 5.5
(backtest and demo). In PowerShell you can switch for one terminal session with
`$env:QUANTCREW_PROFILE="claude"`.

## Run

```bash
python -m pytest                                              # tests (no API key or network needed)
streamlit run app.py                                          # web UI
python -m scripts.run_once --ticker INFY --question "Buy for 6 months?"   # CLI, writes a trace
python -m scripts.update_trust                                # score past live calls, update trust
```

Every run writes its execution trace (every agent message and tool call) to `traces/`.

## Evaluation (backtest + ablation)

```bash
python -m scripts.backtest --tickers INFY,HDFCBANK,RELIANCE --start 2024-06-01 --end 2025-12-31
python -m scripts.backtest --tickers INFY,HDFCBANK,RELIANCE --start 2024-06-01 --end 2025-12-31 --no-debate
python -m scripts.backtest --tickers INFY,HDFCBANK,RELIANCE --start 2024-06-01 --end 2025-12-31 --agents technical
python -m scripts.backtest --tickers INFY,HDFCBANK,RELIANCE --start 2024-06-01 --end 2025-12-31 --agents risk
python -m scripts.compare_backtests      # -> results/comparison.md
python -m scripts.seed_trust             # live app starts from the crew backtest's track record
```

Add `--resume` to continue an interrupted run. Results land in `results/`.

## Repository layout

```
quantcrew/
  agents/      base ReAct loop, four specialists, critic
  tools/       market data (yfinance, Google News), indicators, risk metrics, LangChain tool wrappers
  graph.py     LangGraph orchestration: plan -> parallel agents -> review -> debate -> decide
  decision.py  conflict detection and trust-weighted aggregation (pure, tested)
  trust.py     outcome-based trust weights (pure, tested)
  state.py     shared blackboard state
  trace.py     execution-trace capture
scripts/       CLI run, trust update/seeding, backtest, ablation comparison
tests/         unit tests: indicators, risk, decisions, trust, routing, plans, data cleaning, backtest
docs/          design document, I/O definitions, real sample input/output/trace
results/       backtest outputs and the comparison table
```

## Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub (public, or add `examstudio-bot` as a collaborator).
2. share.streamlit.io → New app → pick the repo, branch `main`, main file `app.py`.
   Under Advanced settings choose Python 3.12.
3. In Secrets, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   QUANTCREW_PROFILE = "claude"
   ```
4. `data/trust.json` (seeded from the backtest) is committed so the deployed app starts with it.
   Live calls logged on the server are lost on redeploy; that is acceptable for a demo.

## Known limitations

- yfinance offers only current fundamentals and recent news, so backtests exclude the Fundamental
  and Sentiment agents to avoid look-ahead bias.
- Backtests ignore transaction costs and slippage.
- LLM outputs vary between runs; temperature is 0, and the debate and weighted vote reduce the variation.
- LLMs may carry knowledge of events after a backtest date.
