"""Walk-forward backtest + ablation.

    python -m scripts.backtest --tickers INFY,HDFCBANK,RELIANCE --start 2024-06-01 --end 2025-12-31
    python -m scripts.backtest ... --agents technical     # single-agent ablation
    python -m scripts.backtest ... --agents risk          # single-agent ablation
    python -m scripts.backtest ... --no-debate            # debate ablation
    python -m scripts.backtest ... --resume               # continue after an interruption

Every `horizon` trading days, the crew analyses each stock using only data up to that date.
Strategy: hold the stock for the next horizon if the call is BUY, else stay in cash.
Baseline: equal-weight buy-and-hold of the same stocks.

No look-ahead: the loop is date-major, and a call's outcome is fed into the trust weights only on
the date it becomes known (when its horizon has elapsed), never earlier. Each finished call is
appended to the CSV immediately, so a crash or rate limit loses nothing; --resume skips done calls.

Limitations (state these in the report): yfinance has no point-in-time fundamentals or news, so
the Fundamental and Sentiment agents are excluded; no transaction costs; LLMs may have memorised
some post-period information."""
import argparse
import json

import numpy as np
import pandas as pd

from quantcrew.config import BENCHMARK, HORIZON_DAYS, RESULTS_DIR, normalise_ticker
from quantcrew.tools.market_data import price_history
from quantcrew.trust import TrustStore

BACKTEST_AGENTS = ["technical", "risk"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--horizon", default="medium", choices=list(HORIZON_DAYS))
    ap.add_argument("--agents", help="comma list to force (ablation), e.g. technical")
    ap.add_argument("--no-debate", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    from quantcrew.graph import run_analysis  # imported late so --help works without an API key

    hold = HORIZON_DAYS[a.horizon]
    forced = a.agents.split(",") if a.agents else None
    label = a.label or ("crew" if not forced else "+".join(forced)) + ("_nodebate" if a.no_debate else "")
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"backtest_{label}.csv"
    trust_file = RESULTS_DIR / f"trust_{label}.json"
    trust_file.unlink(missing_ok=True)  # always rebuilt from the calls themselves

    done, pending = set(), []
    if a.resume and out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev["ticker"], prev["date"].astype(str)))
        pending = prev.to_dict("records")
        print(f"resuming: {len(done)} calls already done")
    else:
        out.unlink(missing_ok=True)

    tickers = [normalise_ticker(t) for t in a.tickers.split(",")]
    prices = {t: price_history(t, a.end, years=4)["Close"] for t in tickers}
    calendar = price_history(BENCHMARK, a.end, years=4).loc[a.start:a.end].index
    store = TrustStore(trust_file)

    for d in calendar[::hold]:
        pending = score_known_outcomes(store, pending, d)
        store.save()
        for ticker in tickers:
            px = prices[ticker]
            if (ticker, str(d.date())) in done or d not in px.index:
                continue
            i = px.index.get_loc(d)
            if i + hold >= len(px):
                continue
            try:
                state = run_analysis(ticker, f"Outlook for the next {hold} trading days?", horizon=a.horizon,
                                     as_of=d.date(), mode="backtest", forced_agents=forced,
                                     max_rounds=0 if a.no_debate else None, trust_file=trust_file)
            except Exception as exc:  # rate limit, network: skip this call, --resume can retry it
                print(f"SKIP {ticker} {d.date()}: {type(exc).__name__}: {str(exc)[:150]}")
                continue
            f = state["final"]
            if f.get("status") == "rejected":
                print(f"REJECTED {ticker} {d.date()}: {f.get('error')}")
                continue
            fwd = float(px.iloc[i + hold] / px.iloc[i] - 1)
            row = {"ticker": ticker, "date": str(d.date()), "outcome_date": str(px.index[i + hold].date()),
                   "call": f["recommendation"], "confidence": f["confidence"], "lean": f.get("lean"),
                   "fwd_return": round(fwd, 5),
                   "strategy_return": round(fwd, 5) if f["recommendation"] == "BUY" else 0.0,
                   "debate_rounds": f["debate"]["rounds"],
                   "signals_changed_in_debate": sum(r["changed"] for r in f["debate"]["revisions"]),
                   "status": f["status"]}
            for agent in BACKTEST_AGENTS:
                op = f["agent_opinions"].get(agent)
                row[f"sig_{agent}"] = op["signal"] if op else ""
                row[f"conf_{agent}"] = op["confidence"] if op else ""
                row[f"weight_{agent}"] = f["trust"].get(agent, {}).get("weight", "")
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
            pending.append(row)
            print(row["ticker"], row["date"], row["call"], f"fwd {fwd:+.2%}")

    pending = score_known_outcomes(store, pending, pd.Timestamp.max)  # final weights, for seeding
    store.save()
    df = pd.read_csv(out)
    summary = summarise(df, hold) | {"label": label, "tickers": tickers, "start": a.start, "end": a.end,
                                     "horizon": a.horizon,
                                     "final_trust": {ag: store.stats(ag) for ag in BACKTEST_AGENTS}}
    (RESULTS_DIR / f"summary_{label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), f"\nsaved -> {out}")


def score_known_outcomes(store: TrustStore, pending: list[dict], today) -> list[dict]:
    """Feed every call whose horizon has elapsed by `today` into the trust store; keep the rest."""
    still = []
    for r in sorted(pending, key=lambda r: r["date"]):
        if pd.Timestamp(r["outcome_date"]) <= today:
            for agent in BACKTEST_AGENTS:
                sig = r.get(f"sig_{agent}")
                if isinstance(sig, str) and sig:
                    store.add_outcome(agent, str(r["date"]), r["ticker"], sig, float(r["fwd_return"]))
        else:
            still.append(r)
    return still


def summarise(df: pd.DataFrame, hold: int) -> dict:
    if df.empty:
        return {}
    per_year = 252 / hold
    by_date = df.groupby("date")[["strategy_return", "fwd_return"]].mean()  # equal-weight portfolio
    strat, bh = by_date["strategy_return"], by_date["fwd_return"]

    def sharpe(r):
        return round(float(r.mean() / r.std() * np.sqrt(per_year)), 2) if len(r) > 1 and r.std() > 0 else None

    def max_dd(r):
        curve = (1 + r).cumprod()
        return round(float((curve / curve.cummax() - 1).min()) * 100, 2)

    directional = df[df["call"] != "HOLD"]
    hits = ((directional["call"] == "BUY") & (directional["fwd_return"] > 0)) | \
           ((directional["call"] == "SELL") & (directional["fwd_return"] < 0))
    return {"periods": int(len(by_date)), "calls": int(len(df)),
            "strategy_cum_return_pct": round(float((1 + strat).prod() - 1) * 100, 2),
            "buy_hold_cum_return_pct": round(float((1 + bh).prod() - 1) * 100, 2),
            "strategy_sharpe": sharpe(strat), "buy_hold_sharpe": sharpe(bh),
            "strategy_max_drawdown_pct": max_dd(strat), "buy_hold_max_drawdown_pct": max_dd(bh),
            "time_in_market_pct": round(float((df["call"] == "BUY").mean()) * 100, 1),
            "directional_hit_rate": round(float(hits.mean()), 3) if len(directional) else None,
            "call_mix": {k: int(v) for k, v in df["call"].value_counts().items()},
            "calls_changed_by_debate": int((df["signals_changed_in_debate"] > 0).sum())}


if __name__ == "__main__":
    main()
