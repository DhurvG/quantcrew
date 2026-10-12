"""Plans for ablations and backtests must be exactly controlled (no LLM, no policy additions)."""
from quantcrew.graph import make_plan

STATE = {"ticker": "INFY.NS", "question": "q", "horizon": "medium"}


def test_single_agent_ablation_is_not_padded_with_risk():
    plan = make_plan(STATE | {"forced_agents": ["technical"]}, ["technical", "risk"], ["fundamental", "sentiment"])
    assert plan["agents"] == ["technical"]


def test_backtest_uses_every_point_in_time_agent_without_llm():
    plan = make_plan(STATE | {"mode": "backtest"}, ["technical", "risk"], ["fundamental", "sentiment"])
    assert plan["agents"] == ["technical", "risk"]
    assert plan["unavailable"] == ["fundamental", "sentiment"]


def test_forced_unavailable_agent_is_dropped():
    plan = make_plan(STATE | {"forced_agents": ["fundamental"]}, ["technical", "risk"], ["fundamental"])
    assert plan["agents"] == []
