"""LangChain tools the agents can choose to call. Built per run so they are bound to the
analysis date (as_of) and can never see data from after it."""
from langchain_core.tools import tool

from quantcrew.config import BENCHMARK
from quantcrew.tools import market_data as md
from quantcrew.tools.indicators import technical_snapshot
from quantcrew.tools.risk_metrics import risk_snapshot


def _safe(fn):
    try:
        return fn()
    except md.DataUnavailable as exc:
        return {"error": str(exc)}


def make_tools(as_of: str | None) -> dict:
    @tool
    def get_technical_indicators(ticker: str) -> dict:
        """RSI(14), MACD, 50/200-day moving averages, recent returns, 52-week range and volume for a ticker."""
        return _safe(lambda: technical_snapshot(md.price_history(ticker, as_of)))

    @tool
    def get_price_window(ticker: str, days: int = 30) -> dict:
        """Last `days` daily closing prices (max 120) for a ticker, to inspect the recent path."""
        def run():
            closes = md.price_history(ticker, as_of)["Close"].iloc[-min(days, 120):]
            return {str(k.date()): round(float(v), 2) for k, v in closes.items()}
        return _safe(run)

    @tool
    def get_risk_metrics(ticker: str) -> dict:
        """1-year annualised volatility, max drawdown, 95% historical VaR/CVaR and beta vs Nifty 50."""
        def run():
            try:
                market = md.price_history(BENCHMARK, as_of)
            except md.DataUnavailable:
                market = None
            return risk_snapshot(md.price_history(ticker, as_of), market)
        return _safe(run)

    @tool
    def get_benchmark_risk() -> dict:
        """The same risk metrics for the Nifty 50 index itself, for comparison."""
        return _safe(lambda: risk_snapshot(md.price_history(BENCHMARK, as_of)))

    @tool
    def get_fundamentals(ticker: str) -> dict:
        """Valuation (P/E, P/B), profitability (ROE, margins), leverage, growth and multi-year revenue/net income."""
        return _safe(lambda: md.fundamentals(ticker))

    @tool
    def get_recent_headlines(ticker: str) -> dict:
        """Up to 12 recent news headlines with publisher and date."""
        return _safe(lambda: {"headlines": md.headlines(ticker)})

    return {t.name: t for t in [get_technical_indicators, get_price_window, get_risk_metrics,
                                get_benchmark_risk, get_fundamentals, get_recent_headlines]}
