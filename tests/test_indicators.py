import numpy as np
import pandas as pd

from quantcrew.tools.indicators import macd, rsi, technical_snapshot


def _prices(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    s = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 1_000_000})


def test_rsi_stays_in_bounds():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 1, 300).cumsum())
    r = rsi(close).dropna()
    assert r.between(0, 100).all()


def test_rsi_high_in_steady_uptrend_and_low_in_downtrend():
    assert rsi(pd.Series(np.linspace(100, 200, 100))).iloc[-1] > 90
    assert rsi(pd.Series(np.linspace(200, 100, 100))).iloc[-1] < 10


def test_macd_positive_when_trend_accelerates_up():
    close = pd.Series(np.exp(np.linspace(0, 1, 200)) * 100)
    assert macd(close)["macd"].iloc[-1] > 0


def test_snapshot_detects_golden_cross_in_uptrend():
    snap = technical_snapshot(_prices(np.linspace(100, 300, 260)))
    assert snap["golden_cross"] is True
    assert snap["pct_vs_sma_200"] > 0
    assert snap["pct_from_52w_high"] == 0.0
