"""Pure technical-indicator functions (no network, unit-tested)."""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def technical_snapshot(df: pd.DataFrame) -> dict:
    close = df["Close"]
    last = float(close.iloc[-1])
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    m = macd(close).iloc[-1]
    window = close.iloc[-252:]

    def pct(a, b):
        return None if pd.isna(a) or pd.isna(b) or b == 0 else round((a / b - 1) * 100, 2)

    return {
        "as_of": str(close.index[-1].date()),
        "last_close": round(last, 2),
        "rsi_14": round(float(rsi(close).iloc[-1]), 2),
        "macd": round(float(m["macd"]), 3),
        "macd_signal": round(float(m["signal"]), 3),
        "macd_histogram": round(float(m["hist"]), 3),
        "sma_50": None if pd.isna(sma50) else round(float(sma50), 2),
        "sma_200": None if pd.isna(sma200) else round(float(sma200), 2),
        "pct_vs_sma_50": pct(last, sma50),
        "pct_vs_sma_200": pct(last, sma200),
        "golden_cross": None if pd.isna(sma200) else bool(sma50 > sma200),
        "return_1m_pct": pct(last, close.iloc[-22]) if len(close) > 22 else None,
        "return_3m_pct": pct(last, close.iloc[-64]) if len(close) > 64 else None,
        "pct_from_52w_high": pct(last, window.max()),
        "pct_from_52w_low": pct(last, window.min()),
        "avg_volume_20d": int(df["Volume"].iloc[-20:].mean()),
        "volume_ratio_vs_20d": round(float(df["Volume"].iloc[-1] / df["Volume"].iloc[-20:].mean()), 2),
    }
