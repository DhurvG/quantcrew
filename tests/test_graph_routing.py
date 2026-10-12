"""Routing logic tested without any LLM or network calls."""
from langgraph.types import Send

from quantcrew.graph import route_after_critic, route_after_gate, route_after_plan

BASE = {"ticker": "INFY.NS", "question": "q", "as_of": None,
        "plan": {"agents": ["technical", "risk", "fundamental"], "horizon": "medium", "unavailable": []},
        "opinions": {}, "errors": {}, "debate_round": 0, "max_rounds": 2}


def test_plan_fans_out_to_every_selected_agent():
    sends = route_after_plan(BASE)
    assert {s.node for s in sends} == {"technical", "risk", "fundamental"}


def test_invalid_ticker_goes_straight_to_decide():
    assert route_after_plan(BASE | {"plan_error": "not nifty"}) == "decide"


def test_failed_agent_is_retried():
    sends = route_after_gate(BASE | {"pending_retry": ["fundamental"]})
    assert isinstance(sends[0], Send) and sends[0].node == "fundamental" and sends[0].arg["retry"]


def test_gate_goes_to_critic_then_stops_at_max_rounds():
    assert route_after_gate(BASE | {"pending_retry": []}) == "critic"
    assert route_after_gate(BASE | {"pending_retry": [], "debate_round": 2}) == "decide"


def test_gate_skips_debate_with_fewer_than_two_valid_agents():
    errs = {"technical": "x", "fundamental": "y"}
    assert route_after_gate(BASE | {"pending_retry": [], "errors": errs}) == "decide"


def test_critic_sends_only_challenged_agents_back():
    critique = {"needs_revision": True, "summary": "",
                "challenges": [{"agent": "technical", "issue": "i", "question": "q"}]}
    sends = route_after_critic(BASE | {"critiques": [critique], "debate_round": 1,
                                       "opinions": {"technical": {"signal": "BUY", "confidence": 0.9}}})
    assert [s.node for s in sends] == ["technical"]
    assert sends[0].arg["critique"]["issue"] == "i"
    assert sends[0].arg["previous"]["signal"] == "BUY"


def test_critic_satisfied_means_decide():
    critique = {"needs_revision": False, "summary": "", "challenges": []}
    assert route_after_critic(BASE | {"critiques": [critique], "debate_round": 1}) == "decide"


def test_multiple_challenges_to_one_agent_are_all_delivered():
    critique = {"needs_revision": True, "summary": "", "challenges": [
        {"agent": "risk", "issue": "i1", "question": "q1"},
        {"agent": "risk", "issue": "i2", "question": "q2"}]}
    sends = route_after_critic(BASE | {"critiques": [critique], "debate_round": 1,
                                       "opinions": {"risk": {"signal": "SELL", "confidence": 0.6}}})
    assert len(sends) == 1
    assert "i1" in sends[0].arg["critique"]["issue"] and "i2" in sends[0].arg["critique"]["issue"]


def test_failed_revision_keeps_previous_opinion(monkeypatch):
    from quantcrew import graph

    def boom(*args, **kwargs):
        raise RuntimeError("429 rate limit")

    monkeypatch.setattr(graph.AGENTS["risk"], "run", boom)
    node = graph.make_specialist_node("risk")
    out = node({"agent": "risk", "ticker": "INFY.NS", "question": "q", "horizon": "medium",
                "as_of": None, "round": 1, "critique": {"issue": "i", "question": "q"},
                "previous": {"signal": "SELL", "confidence": 0.6}, "retry": False})
    assert out["errors"] == {"risk": None}
    assert "opinions" not in out  # the earlier SELL opinion stays in the shared state
    assert out["trace"][-1]["type"] == "revision_failed"


def test_first_attempt_failure_is_reported_for_retry(monkeypatch):
    from quantcrew import graph

    monkeypatch.setattr(graph.AGENTS["risk"], "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    out = graph.make_specialist_node("risk")({"agent": "risk", "ticker": "INFY.NS", "question": "q",
                                              "horizon": "medium", "as_of": None, "round": 0,
                                              "critique": None, "previous": None, "retry": False})
    assert out["errors"]["risk"] and out["trace"][-1]["type"] == "failure"
