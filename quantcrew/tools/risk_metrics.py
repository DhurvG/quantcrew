"""Pure risk-metric functions (no network, unit-tested)."""
import numpy as np
import pandas as pd


def max_drawdown(close: pd.Series) -> float:
    """Most negative peak-to-trough fall, as a fraction (e.g. -0.23)."""
    return float((close / close.cummax() - 1).min())


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """1-day historical VaR as a positive loss fraction."""
    return float(-np.percentile(returns.dropna(), (1 - level) * 100))


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    cutoff = np.percentile(r, (1 - level) * 100)
    return float(-r[r <= cutoff].mean())


def beta(stock_returns: pd.Series, market_returns: pd.Series) -> float | None:
    joined = pd.concat([stock_returns, market_returns], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return None
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


def risk_snapshot(df: pd.DataFrame, market: pd.DataFrame | None = None) -> dict:
    close = df["Close"].iloc[-252:]
    rets = close.pct_change().dropna()
    out = {
        "window_days": int(len(rets)),
        "annualised_volatility_pct": round(float(rets.std() * np.sqrt(252) * 100), 2),
        "max_drawdown_1y_pct": round(max_drawdown(close) * 100, 2),
        "var_95_1d_pct": round(historical_var(rets) * 100, 2),
        "cvar_95_1d_pct": round(cvar(rets) * 100, 2),
        "worst_day_pct": round(float(rets.min() * 100), 2),
        "recent_vol_20d_pct": round(float(rets.iloc[-20:].std() * np.sqrt(252) * 100), 2),
    }
    if market is not None and not market.empty:
        b = beta(rets, market["Close"].iloc[-252:].pct_change().dropna())
        out["beta_vs_nifty"] = None if b is None else round(b, 2)
    return out
