"""Network-facing data access (yfinance + Google News RSS). Every function raises DataUnavailable
instead of returning garbage, so agents can report failures and the Orchestrator can replan."""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from functools import lru_cache

import pandas as pd
import yfinance as yf


class DataUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=256)
def _history(ticker: str, as_of: str | None, years: int) -> pd.DataFrame:
    end = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    try:
        df = yf.Ticker(ticker).history(start=start, end=end + pd.Timedelta(days=1), auto_adjust=True)
    except Exception as exc:  # network, rate limit, bad symbol
        raise DataUnavailable(f"price history for {ticker}: {exc}") from exc
    if df is None or df.empty or len(df) < 60:
        raise DataUnavailable(f"not enough price history for {ticker} up to {end.date()}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def price_history(ticker: str, as_of: str | None = None, years: int = 2) -> pd.DataFrame:
    """Daily OHLCV up to and including as_of (never later: no look-ahead)."""
    return _history(ticker, as_of, years).copy()


def forward_return(ticker: str, as_of: str, days: int) -> float | None:
    """Return from the close on as_of to the close `days` trading days later (for scoring)."""
    start = pd.Timestamp(as_of)
    try:
        df = yf.Ticker(ticker).history(start=start, end=start + pd.Timedelta(days=days * 2 + 10), auto_adjust=True)
    except Exception:
        return None
    if df is None or len(df) <= days:
        return None
    return float(df["Close"].iloc[days] / df["Close"].iloc[0] - 1)


@lru_cache(maxsize=128)
def fundamentals(ticker: str) -> dict:
    """Current fundamentals. NOTE: yfinance only gives today's values, so this is NOT
    point-in-time; backtests exclude the Fundamental agent for that reason."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as exc:
        raise DataUnavailable(f"fundamentals for {ticker}: {exc}") from exc
    keys = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
            "returnOnEquity", "returnOnAssets", "debtToEquity", "currentRatio", "revenueGrowth",
            "earningsGrowth", "profitMargins", "operatingMargins", "payoutRatio"]
    out = {k: info.get(k) for k in keys if info.get(k) is not None}
    if len(out) < 4:
        raise DataUnavailable(f"fundamental fields missing for {ticker}")
    return normalise_fundamentals(out, info.get("financialCurrency"), _annual_trend(t))


def normalise_fundamentals(out: dict, statements_currency: str | None, trend) -> dict:
    """Fix yfinance unit traps (unit-tested). Found by reviewing execution traces:
    - debtToEquity arrives as a percentage (9.5 means 0.095x), and was misread as 9.5x debt
    - financial statements may be in USD while marketCap is in INR (e.g. Infosys)"""
    out = dict(out)
    if "debtToEquity" in out:
        out["debtToEquity"] = round(out["debtToEquity"] / 100, 3)
    out["units"] = ("returnOnEquity, returnOnAssets, margins, revenueGrowth, earningsGrowth and "
                    "payoutRatio are fractions (0.32 = 32%). debtToEquity is a plain ratio "
                    "(0.10 = debt is 10% of equity). marketCap is in INR; annual_trend figures are in "
                    "statements_currency, which may differ, so never divide one by the other.")
    out["statements_currency"] = statements_currency or "unknown"
    out["annual_trend"] = trend
    return out


def _annual_trend(t) -> dict | str:
    try:
        inc = t.income_stmt
        trend = {}
        for row in ("Total Revenue", "Net Income"):
            if row in inc.index:
                trend[row] = {str(c.date()): float(v) for c, v in inc.loc[row].dropna().items()}
        return trend
    except Exception:
        return "unavailable"


def _yahoo_headlines(ticker: str, limit: int) -> list[dict]:
    out = []
    for it in (yf.Ticker(ticker).news or [])[:limit]:
        c = it.get("content", it)
        title = c.get("title")
        if not title:
            continue
        provider = c.get("provider")
        out.append({
            "title": title,
            "publisher": provider.get("displayName") if isinstance(provider, dict) else c.get("publisher"),
            "published": c.get("pubDate") or c.get("providerPublishTime"),
            "summary": (c.get("summary") or "")[:300],
            "feed": "yahoo",
        })
    return out


def parse_google_news_rss(xml_bytes: bytes, limit: int) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.iter("item"):
        title = item.findtext("title")
        if title:
            out.append({"title": title, "publisher": item.findtext("source"),
                        "published": item.findtext("pubDate"), "summary": "", "feed": "google_news"})
        if len(out) >= limit:
            break
    return out


def _google_news_headlines(query: str, limit: int) -> list[dict]:
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f'"{query}" when:7d', "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return parse_google_news_rss(resp.read(), limit)


def headlines(ticker: str, limit: int = 12) -> list[dict]:
    """Recent headlines: Yahoo first; Yahoo's NSE news is often empty, so Google News RSS
    (last 7 days, searched by full company name) is the fallback."""
    errors = []
    try:
        out = _yahoo_headlines(ticker, limit)
        if out:
            return out
        errors.append("yahoo: empty")
    except Exception as exc:
        errors.append(f"yahoo: {exc}")
    try:
        try:
            name = fundamentals(ticker).get("longName")
        except Exception:
            name = None
        out = _google_news_headlines(name or ticker.removesuffix(".NS"), limit)
        if out:
            return out
        errors.append("google news: empty")
    except Exception as exc:
        errors.append(f"google news: {exc}")
    raise DataUnavailable(f"no recent headlines for {ticker} ({'; '.join(errors)})")
