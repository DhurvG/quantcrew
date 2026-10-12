"""Pydantic schemas: the system's input/output contracts and each agent's structured output."""
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

Signal = Literal["BUY", "HOLD", "SELL"]
AgentName = Literal["fundamental", "technical", "sentiment", "risk"]
Horizon = Literal["short", "medium", "long"]


class AnalysisRequest(BaseModel):
    """System input."""
    ticker: str = Field(description="Nifty 50 symbol, e.g. 'INFY' or 'INFY.NS'")
    question: str = Field(default="Should I buy, hold or sell this stock?", max_length=500)
    horizon: Optional[Horizon] = Field(default=None, description="If omitted, the Orchestrator infers it")
    as_of: Optional[date] = Field(default=None, description="Analysis date; defaults to today. Used by backtests.")


class AgentOpinion(BaseModel):
    signal: Signal
    confidence: float = Field(ge=0, le=1, description="0 = guessing, 1 = certain")
    rationale: str = Field(description="2-4 sentences explaining the call")
    key_evidence: list[str] = Field(description="Specific numbers from tool results that support the call")


class Plan(BaseModel):
    agents: list[AgentName] = Field(description="Which specialists to activate")
    horizon: Horizon
    reasoning: str


class Challenge(BaseModel):
    agent: AgentName
    issue: str = Field(description="The specific weakness, contradiction or unsupported claim")
    question: str = Field(description="What the agent must answer or re-check in its revision")


class CriticReview(BaseModel):
    challenges: list[Challenge]
    needs_revision: bool = Field(description="True only if at least one challenge could change a signal")
    summary: str


class Narrative(BaseModel):
    summary: str
    bull_case: list[str]
    bear_case: list[str]
    key_risks: list[str]
    what_would_change_the_view: list[str]


DISCLAIMER = (
    "QuantCrew is an academic decision-support prototype. Its output is advisory only, "
    "is not investment advice, and the system never places trades."
)
