"""Shared LangGraph state (the 'blackboard' every agent reads from and writes to)."""
import operator
from typing import Annotated, Any, TypedDict


def merge_dict(left: dict | None, right: dict | None) -> dict:
    return {**(left or {}), **(right or {})}


class QCState(TypedDict, total=False):
    # input
    ticker: str
    question: str
    horizon: str | None
    as_of: str | None
    mode: str                     # "live" or "backtest"
    forced_agents: list[str] | None
    max_rounds: int
    trust_file: str | None
    # orchestrator
    plan: dict
    plan_error: str | None
    # specialists (parallel writers -> merged)
    opinions: Annotated[dict, merge_dict]
    errors: Annotated[dict, merge_dict]
    retries: Annotated[dict, merge_dict]
    pending_retry: list[str]
    # debate
    debate_round: int
    conflict: dict
    critiques: Annotated[list, operator.add]
    revisions: Annotated[list, operator.add]   # every opinion change, for the trace/UI
    # output
    final: dict
    trace: Annotated[list[dict[str, Any]], operator.add]
