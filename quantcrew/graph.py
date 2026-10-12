"""LangGraph wiring. Pattern: hierarchical orchestrator-worker over a shared-state blackboard,
with a Critic-led debate loop and failure-driven replanning.

  START -> orchestrator --(Send, parallel)--> [selected specialists] -> review_gate
  review_gate -> (retry failed agents once) | critic | decide
  critic -> (Send challenged agents back for revision) | decide
  decide -> END
"""
import json
from datetime import date

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from quantcrew.agents.critic import CRITIC
from quantcrew.agents.specialists import AGENTS, build_task
from quantcrew.config import (MAX_CONCURRENCY, MAX_DEBATE_ROUNDS, NIFTY50_SYMBOLS,
                              RECOMMENDATIONS_FILE, SPECIALISTS, TRUST_FILE, normalise_ticker)
from quantcrew.decision import aggregate, detect_conflict
from quantcrew.llm import get_llm, structured_invoke
from quantcrew.schemas import DISCLAIMER, AgentOpinion, Narrative, Plan
from quantcrew.state import QCState
from quantcrew.tools import market_data as md
from quantcrew.trace import event
from quantcrew.trust import TrustStore

# In backtests there is no point-in-time news or fundamentals in yfinance, so those agents
# (and the matching critic tool) are unavailable to avoid look-ahead bias.
BACKTEST_UNAVAILABLE = {"fundamental": "get_fundamentals", "sentiment": "get_recent_headlines"}


# ---------------------------------------------------------------- orchestrator: plan
def make_plan(state: QCState, available: list[str], unavailable: list[str]) -> dict:
    """Decide which specialists run. Ablations and backtests use fixed plans (an evaluation harness
    must be controlled); live requests are planned by the LLM."""
    horizon = state.get("horizon") or "medium"
    if state.get("forced_agents"):  # ablation: exactly these agents, no policy additions
        agents = [a for a in dict.fromkeys(state["forced_agents"]) if a in available]
        return {"agents": agents, "horizon": horizon, "reasoning": "Agents fixed by ablation setting.",
                "unavailable": unavailable}
    if state.get("mode") == "backtest":
        return {"agents": available, "horizon": horizon,
                "reasoning": "Backtest: every point-in-time agent is used.", "unavailable": unavailable}

    prompt = (
        "You are the Orchestrator of QuantCrew, a multi-agent equity research team for Nifty 50 stocks. "
        "Decide which specialists to activate for this request and the investment horizon.\n"
        f"Specialists available right now: {available}\n"
        "- fundamental: valuation, profitability, growth (most useful for medium/long horizons)\n"
        "- technical: trend and momentum (most useful for short/medium horizons)\n"
        "- sentiment: recent news tone (most useful for short horizons or after news events)\n"
        "- risk: volatility, drawdown, VaR (always useful)\n"
        f"Unavailable this run: {unavailable or 'none'}\n"
        f"User question: {state['question']}\nStock: {state['ticker']}\n"
        f"User-specified horizon: {state.get('horizon') or 'not given - infer it from the question'}\n"
        "Activate at least two specialists, and only the ones that add signal for this question.")
    plan = structured_invoke(get_llm(), Plan, prompt)
    if state.get("horizon"):
        plan.horizon = state["horizon"]
    agents = [a for a in dict.fromkeys(plan.agents) if a in available]
    if "risk" in available and "risk" not in agents:   # policy: risk is always consulted
        agents.append("risk")
    if len(agents) < 2:
        agents = available[:]
    return {"agents": agents, "horizon": plan.horizon, "reasoning": plan.reasoning, "unavailable": unavailable}


def orchestrator_plan(state: QCState) -> dict:
    ticker = normalise_ticker(state["ticker"])
    state = {**state, "ticker": ticker}
    trace = [event("user", "orchestrator", "request",
                   {"ticker": ticker, "question": state["question"], "as_of": state.get("as_of")})]

    if ticker not in NIFTY50_SYMBOLS:
        msg = f"{ticker} is not in the configured Nifty 50 list. QuantCrew only covers Nifty 50 stocks."
        return {"ticker": ticker, "plan_error": msg, "trace": trace + [event("orchestrator", "user", "rejected", msg)]}
    try:  # cheap availability probe before spending LLM calls
        md.price_history(ticker, state.get("as_of"))
    except md.DataUnavailable as exc:
        msg = f"Market data unavailable for {ticker}: {exc}"
        return {"ticker": ticker, "plan_error": msg, "trace": trace + [event("orchestrator", "user", "rejected", msg)]}

    unavailable = list(BACKTEST_UNAVAILABLE) if state.get("mode") == "backtest" else []
    available = [a for a in SPECIALISTS if a not in unavailable]
    plan_d = make_plan(state, available, unavailable)
    if not plan_d["agents"]:
        msg = f"No requested agents are available in this mode (unavailable: {unavailable})."
        return {"ticker": ticker, "plan_error": msg, "trace": trace + [event("orchestrator", "user", "rejected", msg)]}
    trace.append(event("orchestrator", "orchestrator", "plan", plan_d))
    return {"ticker": ticker, "plan": plan_d, "plan_error": None, "debate_round": 0, "trace": trace}


def _payload(state: QCState, agent: str, critique=None, retry=False) -> dict:
    return {"agent": agent, "ticker": state["ticker"], "question": state["question"],
            "horizon": state["plan"]["horizon"], "as_of": state.get("as_of"),
            "round": state.get("debate_round", 0), "critique": critique,
            "previous": (state.get("opinions") or {}).get(agent), "retry": retry}


def route_after_plan(state: QCState):
    if state.get("plan_error"):
        return "decide"
    return [Send(a, _payload(state, a)) for a in state["plan"]["agents"]]


# ---------------------------------------------------------------- specialists
def make_specialist_node(name: str):
    def node(p: dict) -> dict:
        agent = AGENTS[name]
        kind = "challenge" if p.get("critique") else ("retry" if p.get("retry") else "task")
        sender = "critic" if p.get("critique") else "orchestrator"
        trace = [event(sender, name, kind, p.get("critique") or {"horizon": p["horizon"]})]
        try:
            op, events = agent.run(build_task(p["ticker"], p["question"], p["horizon"], p.get("critique"),
                                              p.get("previous")), AgentOpinion, p.get("as_of"))
        except Exception as exc:  # AgentFailed or LLM/API error: one agent failing must not crash the crew
            msg = f"{type(exc).__name__}: {exc}"
            if p.get("previous"):  # failed while revising: keep the opinion it already gave
                return {"errors": {name: None},
                        "trace": trace + [event(name, "orchestrator", "revision_failed",
                                                f"kept previous opinion; {msg[:300]}")]}
            return {"errors": {name: msg}, "trace": trace + [event(name, "orchestrator", "failure", msg)]}
        new = op.model_dump() | {"round": p["round"]}
        out = {"opinions": {name: new}, "errors": {name: None},
               "trace": trace + events + [event(name, "orchestrator", "opinion", new)]}
        if p.get("previous"):
            old = p["previous"]
            out["revisions"] = [{"agent": name, "round": p["round"], "from": old["signal"], "to": new["signal"],
                                 "confidence_from": old["confidence"], "confidence_to": new["confidence"],
                                 "changed": old["signal"] != new["signal"]}]
        return out

    return node


# ---------------------------------------------------------------- review gate (join point)
def review_gate(state: QCState) -> dict:
    planned = state["plan"]["agents"]
    errors = state.get("errors") or {}
    failed = [a for a in planned if errors.get(a)]
    retries = state.get("retries") or {}
    pending = [a for a in failed if retries.get(a, 0) < 1]
    live = {a: o for a, o in (state.get("opinions") or {}).items() if a in planned and not errors.get(a)}
    conflict = detect_conflict(live)
    trace = [event("orchestrator", "orchestrator", "review",
                   {"valid_opinions": list(live), "failed": failed, "retrying": pending, "conflict": conflict})]
    return {"pending_retry": pending, "retries": {a: retries.get(a, 0) + 1 for a in pending},
            "conflict": conflict, "trace": trace}


def route_after_gate(state: QCState):
    if state.get("pending_retry"):  # replan: give failed agents one more attempt
        return [Send(a, _payload(state, a, retry=True)) for a in state["pending_retry"]]
    valid = [a for a in state["plan"]["agents"] if not (state.get("errors") or {}).get(a)]
    max_rounds = state.get("max_rounds", MAX_DEBATE_ROUNDS)
    if len(valid) >= 2 and state.get("debate_round", 0) < max_rounds:
        return "critic"
    return "decide"


# ---------------------------------------------------------------- critic
def critic_node(state: QCState) -> dict:
    errors = state.get("errors") or {}
    live = {a: o for a, o in state["opinions"].items() if a in state["plan"]["agents"] and not errors.get(a)}
    round_no = state.get("debate_round", 0)
    unavailable_tools = [BACKTEST_UNAVAILABLE[a] for a in state["plan"].get("unavailable", [])]
    trace = [event("orchestrator", "critic", "review_request", {"round": round_no + 1, "agents": list(live)})]
    try:
        review, events = CRITIC.review(state["ticker"], state["plan"]["horizon"], live, state.get("conflict"),
                                       round_no, state.get("as_of"), unavailable_tools)
    except Exception as exc:
        msg = f"Critic failed ({exc}); proceeding without further debate."
        return {"debate_round": 99, "trace": trace + [event("critic", "orchestrator", "failure", msg)]}
    rd = review.model_dump() | {"round": round_no + 1}
    trace += events + [event("critic", "orchestrator", "critique", rd)]
    for ch in review.challenges:
        if ch.agent in live:
            trace.append(event("critic", ch.agent, "challenge_issued", ch.model_dump()))
    return {"critiques": [rd], "debate_round": round_no + 1, "trace": trace}


def route_after_critic(state: QCState):
    if not state.get("critiques") or state.get("debate_round", 0) >= 99:
        return "decide"
    latest = state["critiques"][-1]
    errors = state.get("errors") or {}
    targets = {}
    for c in latest["challenges"]:
        a = c["agent"]
        if a not in state["plan"]["agents"] or errors.get(a):
            continue
        if a in targets:  # several challenges for one agent: deliver all of them
            targets[a] = {"agent": a, "issue": f"{targets[a]['issue']} | {c['issue']}",
                          "question": f"{targets[a]['question']} | {c['question']}"}
        else:
            targets[a] = c
    if not latest["needs_revision"] or not targets:
        return "decide"
    return [Send(a, _payload(state, a, critique=c)) for a, c in targets.items()]


# ---------------------------------------------------------------- orchestrator: decide
EMPTY_NARRATIVE = {"bull_case": [], "bear_case": [], "key_risks": [], "what_would_change_the_view": []}


def decide(state: QCState) -> dict:
    if state.get("plan_error"):
        final = {"status": "rejected", "ticker": state["ticker"], "error": state["plan_error"], "disclaimer": DISCLAIMER}
        return {"final": final, "trace": [event("orchestrator", "user", "final", final)]}

    errors = state.get("errors") or {}
    planned = state["plan"]["agents"]
    live = {a: o for a, o in (state.get("opinions") or {}).items() if a in planned and not errors.get(a)}
    store = TrustStore(state.get("trust_file") or TRUST_FILE)
    weights = store.weights(live)
    agg = aggregate(live, weights)
    degraded = [a for a in planned if errors.get(a)]

    if state.get("mode") == "backtest":  # saves an LLM call per backtest step
        narrative = {"summary": "Backtest run: narrative skipped.", **EMPTY_NARRATIVE}
    else:
        try:
            narrative = structured_invoke(get_llm(), Narrative,
                "You are the QuantCrew Orchestrator writing the final research note. The recommendation has already "
                "been computed by trust-weighted vote; explain it, do not change it. Use only figures that appear "
                "in the agent opinions.\n"
                f"Stock: {state['ticker']}\nHorizon: {state['plan']['horizon']}\nQuestion: {state['question']}\n"
                f"Recommendation: {agg['recommendation']} (score {agg['score']}, confidence {agg['confidence']}, "
                f"lean {agg['lean']})\n"
                f"Agent opinions: {json.dumps(live)}\nCritic reviews: {json.dumps(state.get('critiques', []))}\n"
                f"Agents that failed: {degraded or 'none'}").model_dump()
        except Exception as exc:
            narrative = {"summary": f"Narrative unavailable ({type(exc).__name__}); see agent opinions.",
                         **EMPTY_NARRATIVE}

    final = {
        "status": "degraded" if degraded else "ok",
        "ticker": state["ticker"],
        "as_of": state.get("as_of") or str(date.today()),
        "horizon": state["plan"]["horizon"],
        "recommendation": agg["recommendation"],
        "confidence": agg["confidence"],
        "weighted_score": agg["score"],
        "lean": agg["lean"],
        "minority_call_overruled": agg["minority_call_overruled"],
        "agent_votes": agg["contributions"],
        "agent_opinions": live,
        "failed_agents": {a: errors[a] for a in degraded},
        "debate": {"rounds": len(state.get("critiques", [])), "critiques": state.get("critiques", []),
                   "revisions": state.get("revisions", [])},
        "trust": {a: store.stats(a) for a in live},
        "narrative": narrative,
        "disclaimer": DISCLAIMER,
    }
    if state.get("mode", "live") == "live":
        _log_recommendation(final)
    return {"final": final, "trace": [event("orchestrator", "user", "final",
                                            {k: final[k] for k in ("recommendation", "confidence",
                                                                   "weighted_score", "lean")})]}


def _log_recommendation(final: dict):
    """Stored so scripts/update_trust.py can later score each agent against what the price actually did."""
    RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ticker": final["ticker"], "as_of": final["as_of"], "horizon": final["horizon"],
           "recommendation": final["recommendation"],
           "agent_signals": {a: o["signal"] for a, o in final["agent_opinions"].items()}, "scored": False}
    with RECOMMENDATIONS_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- build
def build_graph():
    g = StateGraph(QCState)
    g.add_node("orchestrator", orchestrator_plan)
    for name in SPECIALISTS:
        g.add_node(name, make_specialist_node(name))
        g.add_edge(name, "review_gate")
    g.add_node("review_gate", review_gate)
    g.add_node("critic", critic_node)
    g.add_node("decide", decide)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", route_after_plan, SPECIALISTS + ["decide"])
    g.add_conditional_edges("review_gate", route_after_gate, SPECIALISTS + ["critic", "decide"])
    g.add_conditional_edges("critic", route_after_critic, SPECIALISTS + ["decide"])
    g.add_edge("decide", END)
    return g.compile()


GRAPH = None
RUN_CONFIG = {"recursion_limit": 60, "max_concurrency": MAX_CONCURRENCY}


def initial_state(ticker, question, horizon=None, as_of=None, mode="live", forced_agents=None,
                  max_rounds=None, trust_file=None) -> dict:
    return {"ticker": ticker, "question": question, "horizon": horizon,
            "as_of": str(as_of) if as_of else None, "mode": mode, "forced_agents": forced_agents,
            "max_rounds": MAX_DEBATE_ROUNDS if max_rounds is None else max_rounds,
            "trust_file": str(trust_file) if trust_file else None,
            "opinions": {}, "errors": {}, "retries": {}, "critiques": [], "revisions": [], "trace": []}


def _graph():
    global GRAPH
    GRAPH = GRAPH or build_graph()
    return GRAPH


def run_analysis(ticker: str, question: str = "Should I buy, hold or sell?", **kwargs) -> dict:
    return _graph().invoke(initial_state(ticker, question, **kwargs), RUN_CONFIG)


def stream_analysis(ticker: str, question: str, **kwargs):
    """Yields (node_name, update) as each agent finishes - used by the Streamlit UI."""
    for chunk in _graph().stream(initial_state(ticker, question, **kwargs), RUN_CONFIG, stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update
