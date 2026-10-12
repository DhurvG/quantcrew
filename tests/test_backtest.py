"""The backtest must never let a call's outcome influence trust before that outcome is known."""
import pandas as pd

from scripts.backtest import score_known_outcomes, summarise
from quantcrew.trust import TrustStore


def _row(date, outcome_date, sig, fwd):
    return {"ticker": "INFY.NS", "date": date, "outcome_date": outcome_date, "sig_technical": sig,
            "sig_risk": "", "fwd_return": fwd}


def test_outcomes_are_scored_only_once_known(tmp_path):
    store = TrustStore(tmp_path / "t.json")
    pending = [_row("2024-01-01", "2024-01-29", "BUY", 0.05), _row("2024-01-29", "2024-02-26", "BUY", 0.05)]
    left = score_known_outcomes(store, pending, pd.Timestamp("2024-01-29"))
    assert store.stats("technical")["n"] == 1       # first call's outcome is known on its outcome date
    assert len(left) == 1 and left[0]["date"] == "2024-01-29"
    assert store.stats("risk")["n"] == 0           # empty signal = agent did not run


def test_summary_compares_strategy_with_buy_and_hold():
    df = pd.DataFrame([
        {"date": "d1", "call": "BUY", "fwd_return": 0.10, "strategy_return": 0.10, "signals_changed_in_debate": 1},
        {"date": "d2", "call": "HOLD", "fwd_return": -0.10, "strategy_return": 0.0, "signals_changed_in_debate": 0},
    ])
    s = summarise(df, hold=20)
    assert s["strategy_cum_return_pct"] == 10.0
    assert s["buy_hold_cum_return_pct"] == -1.0    # 1.1 * 0.9 - 1
    assert s["directional_hit_rate"] == 1.0 and s["calls_changed_by_debate"] == 1
