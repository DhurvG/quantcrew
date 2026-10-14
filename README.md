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

The submitted system runs on Groq's free tier (`QUANTCREW_PROFILE=groq`): gpt-oss-20b for the
Orchestrator and specialists, gpt-oss-120b for the Critic. `QUANTCREW_PROFILE=claude` switches every
role to Claude Haiku 5.5 if a paid key is available.

## Run

```bash
python -m pytest                                              # tests (no API key or network needed)
streamlit run app.py                                          # web UI
python -m scripts.run_once --ticker INFY --question "Buy for 6 months?"   # CLI, writes a trace
python -m scripts.update_trust                                # score past live calls, update trust
```

Every run writes its execution trace (every agent message and tool call) to `traces/`.

## Evaluation (pilot backtest)

```bash
python -m scripts.backtest --tickers INFY --start 2025-07-01 --end 2025-12-31                     # full crew
python -m scripts.backtest --tickers INFY --start 2025-07-01 --end 2025-12-31 --agents technical  # single agent
python -m scripts.compare_backtests      # -> results/comparison.md
```

A 6-decision pilot on INFY fits the free API tier; results are in `results/`. The same scripts run
the full study (more stocks, `--no-debate`, `--agents risk`) with a paid key. Add `--resume` to
continue an interrupted run.

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
scripts/       CLI run, trust update, backtest, ablation comparison
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
   GROQ_API_KEY = "gsk_..."
   QUANTCREW_PROFILE = "groq"
   ```
4. Live app: https://quantcrew-kn4cvvrt6apppkycthnbs7y.streamlit.app
   Trust weights start neutral (1.0) and change only after 15 scored calls per agent. Calls logged
   on the server are lost on redeploy; that is acceptable for a demo.

## Known limitations

- The pilot backtest (6 decisions on one stock) is far too small for statistical conclusions; it was
  limited by the free API tier (~200k tokens per model per day).
- yfinance offers only current fundamentals and recent news, so backtests exclude the Fundamental
  and Sentiment agents to avoid look-ahead bias.
- Backtests ignore transaction costs and slippage.
- LLM outputs vary between runs; temperature is 0, and the debate and weighted vote reduce the variation.
- LLMs may carry knowledge of events after a backtest date.
