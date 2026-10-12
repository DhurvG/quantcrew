"""Combine backtest summaries into one comparison table (the ablation study).

    python -m scripts.compare_backtests

Reads results/summary_*.json and writes results/comparison.md."""
import json

from quantcrew.config import RESULTS_DIR

COLUMNS = [("label", "Configuration"), ("calls", "Calls"),
           ("strategy_cum_return_pct", "Strategy return %"), ("buy_hold_cum_return_pct", "Buy & hold %"),
           ("strategy_sharpe", "Strategy Sharpe"), ("buy_hold_sharpe", "B&H Sharpe"),
           ("strategy_max_drawdown_pct", "Strategy max DD %"), ("buy_hold_max_drawdown_pct", "B&H max DD %"),
           ("time_in_market_pct", "Time in market %"), ("directional_hit_rate", "Hit rate"),
           ("calls_changed_by_debate", "Calls changed by debate")]


def main():
    summaries = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("summary_*.json"))]
    if not summaries:
        print("No results/summary_*.json found. Run scripts.backtest first.")
        return
    lines = ["| " + " | ".join(h for _, h in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for s in summaries:
        lines.append("| " + " | ".join("" if s.get(k) is None else str(s.get(k)) for k, _ in COLUMNS) + " |")
    first = summaries[0]
    text = (f"# Backtest comparison\n\nStocks: {', '.join(first.get('tickers', []))}. "
            f"Period: {first.get('start')} to {first.get('end')}. Horizon: {first.get('horizon')}.\n\n"
            + "\n".join(lines) + "\n\n"
            "Strategy: hold the stock for the horizon on BUY, otherwise cash; equal-weight across stocks. "
            "No transaction costs. Fundamental and Sentiment agents are excluded from backtests because "
            "no point-in-time data exists for them.\n")
    (RESULTS_DIR / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
