"""The four specialist agents. Each has its own role, its own tools, and decides for itself
which tools to call before committing to a BUY/HOLD/SELL opinion."""
from quantcrew.agents.base import ToolUsingAgent
from quantcrew.schemas import AgentOpinion

COMMON = (
    "You are part of QuantCrew, a team of AI analysts producing advisory (not executable) research "
    "on a Nifty 50 stock. Call your tools to gather evidence before concluding; if a tool returns an "
    "error, try another tool or lower your confidence and say what was missing. Cite exact figures. "
    "Your confidence must reflect evidence quality: weak or missing data means confidence below 0.5. "
    "Never use figures from memory - only numbers returned by your tools."
)


class FundamentalAgent(ToolUsingAgent):
    name = "fundamental"
    tool_names = ["get_fundamentals"]
    system_prompt = COMMON + (
        "\nROLE: Fundamental Analyst. Judge valuation (P/E, P/B vs what the sector usually commands), "
        "profitability (ROE, margins), balance-sheet strength (debt/equity) and growth trend (multi-year "
        "revenue and net income). Banks and NBFCs naturally carry high debt/equity - do not penalise that. "
        "Read the 'units' note in the tool result before interpreting any figure. "
        "You have NO tool for sector or peer averages: never state a sector average, peer figure or "
        "benchmark from memory. If asked for a peer comparison, say it is unavailable and lower your "
        "confidence instead.")


class TechnicalAgent(ToolUsingAgent):
    name = "technical"
    tool_names = ["get_technical_indicators", "get_price_window"]
    system_prompt = COMMON + (
        "\nROLE: Technical Analyst. Interpret trend (price vs 50/200 DMA, golden/death cross), momentum "
        "(RSI, MACD histogram) and position in the 52-week range. RSI > 70 is overbought, < 30 oversold. "
        "Match your reading to the investment horizon you are given.")


class SentimentAgent(ToolUsingAgent):
    name = "sentiment"
    tool_names = ["get_recent_headlines"]
    system_prompt = COMMON + (
        "\nROLE: News Sentiment Analyst. Score the tone of recent headlines about this company. "
        "Separate company-specific news from generic market chatter, weight newer and more material "
        "items (earnings, guidance, regulation, management) more, and list the headlines you relied on.")


class RiskAgent(ToolUsingAgent):
    name = "risk"
    tool_names = ["get_risk_metrics", "get_benchmark_risk"]
    system_prompt = COMMON + (
        "\nROLE: Risk Analyst. Assess volatility, drawdown, VaR/CVaR and beta relative to the Nifty 50 "
        "itself (call get_benchmark_risk to compare). Your signal expresses whether the risk is acceptable "
        "for the horizon: unusually high or rising risk argues SELL/HOLD, contained risk does not by itself "
        "argue BUY - use HOLD with a note when risk is simply ordinary.")


AGENTS = {a.name: a for a in [FundamentalAgent(), TechnicalAgent(), SentimentAgent(), RiskAgent()]}


def build_task(ticker: str, question: str, horizon: str, critique: dict | None = None,
               previous: dict | None = None) -> str:
    task = (f"Stock: {ticker}\nInvestment horizon: {horizon}\nUser question: {question}\n"
            f"Give your BUY/HOLD/SELL opinion from your specialist perspective.")
    if critique and previous:
        task += (
            "\n\nDEBATE ROUND - the Critic has challenged your previous opinion.\n"
            f"Your previous opinion: {previous}\n"
            f"Critic's issue: {critique['issue']}\nCritic's question: {critique['question']}\n"
            "Re-check the evidence with your tools as needed. Then either revise your signal/confidence "
            "or defend it - but explicitly address the challenge in your rationale.")
    return task
