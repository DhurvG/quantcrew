"""QuantCrew Streamlit front end. Shows each agent's work live as the graph runs."""
import json
import os

import streamlit as st

# Streamlit Cloud keeps keys in st.secrets; copy them into the environment before quantcrew reads config.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:  # no secrets file locally: .env is used instead
    pass

from quantcrew.config import NIFTY50, SPECIALISTS  # noqa: E402
from quantcrew.graph import stream_analysis  # noqa: E402
from quantcrew.schemas import DISCLAIMER  # noqa: E402
from quantcrew.trace import save_trace  # noqa: E402
from quantcrew.trust import TrustStore  # noqa: E402

st.set_page_config(page_title="QuantCrew", page_icon="📈", layout="wide")

LABELS = {"fundamental": "Fundamental analyst", "technical": "Technical analyst",
          "sentiment": "News sentiment", "risk": "Risk analyst", "critic": "Critic",
          "orchestrator": "Orchestrator", "review_gate": "Orchestrator review", "decide": "Final decision"}
SIGNAL_COLOUR = {"BUY": "green", "HOLD": "orange", "SELL": "red"}

st.title("QuantCrew")
st.caption("Five AI analysts debate a Nifty 50 stock, and an orchestrator weighs their views by track record.")
st.info(DISCLAIMER)

with st.sidebar:
    st.subheader("Ask the crew")
    picked = st.selectbox("Stock", NIFTY50, index=NIFTY50.index("INFY"))
    typed = st.text_input("…or type any NSE symbol", placeholder="symbols outside the Nifty 50 are rejected")
    ticker = typed.strip() or picked
    question = st.text_area("Question", "Is this a good buy for the next 6 months?", max_chars=500)
    horizon = st.selectbox("Horizon", ["Let the orchestrator decide", "short", "medium", "long"])
    run = st.button("Run analysis", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Agent trust")
    store = TrustStore()
    st.dataframe([{"agent": a, **store.stats(a)} for a in SPECIALISTS], hide_index=True,
                 column_order=["agent", "weight", "n", "posterior_accuracy", "active"])
    st.caption("Learned from scored past calls (backtest-seeded for Technical and Risk). "
               "Weights stay at 1.0 until an agent has 15 scored calls.")

if run and not question.strip():
    st.warning("Please type a question for the crew.")
elif run:
    state = {"ticker": ticker, "question": question, "trace": []}
    final = None
    try:
        with st.status("The crew is working… (about 1–3 minutes)", expanded=True) as status:
            for node, update in stream_analysis(ticker, question,
                                                horizon=None if horizon.startswith("Let") else horizon):
                update = update or {}
                state["trace"] += update.get("trace", [])
                if update.get("plan"):
                    state["plan"] = update["plan"]
                    st.write(f"**Orchestrator** activated {', '.join(update['plan']['agents'])} "
                             f"for a {update['plan']['horizon']} horizon. {update['plan']['reasoning']}")
                for agent, op in (update.get("opinions") or {}).items():
                    tag = " (revised after critique)" if op.get("round") else ""
                    st.write(f"**{LABELS[agent]}**{tag}: :{SIGNAL_COLOUR[op['signal']]}[{op['signal']}] "
                             f"at {op['confidence']:.0%} confidence. {op['rationale']}")
                for agent, err in (update.get("errors") or {}).items():
                    if err:
                        st.write(f"**{LABELS[agent]}** failed: {err[:200]}. "
                                 "The orchestrator will retry once, then continue without it.")
                for e in update.get("trace", []):
                    if e["type"] == "revision_failed":
                        st.write(f"**{LABELS[e['from']]}** could not revise; its earlier opinion is kept.")
                for c in update.get("critiques", []):
                    st.write(f"**Critic**, round {c['round']}: {c['summary']}")
                    for ch in c["challenges"]:
                        st.write(f"- challenges *{LABELS[ch['agent']]}*: {ch['issue']}")
                if update.get("final"):
                    final = update["final"]
            status.update(label="Analysis complete", state="complete")
    except Exception as exc:
        st.error(f"The run stopped: {exc}. Check the API key and rate limits, then try again.")

    if final and final.get("status") == "rejected":
        st.error(final["error"])
    elif final:
        state.update(final=final, debate_round=final["debate"]["rounds"])
        path = save_trace(state)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommendation", final["recommendation"])
        c2.metric("Confidence", f"{final['confidence']:.0%}")
        c3.metric("Panel lean", final.get("lean", "NONE"))
        c4.metric("Debate rounds", final["debate"]["rounds"])
        if final.get("minority_call_overruled"):
            st.caption("A BUY/SELL backed by less than half the panel's conviction was turned into HOLD.")
        if final["failed_agents"]:
            st.warning(f"Decided without: {', '.join(LABELS[a] for a in final['failed_agents'])}")
        n = final["narrative"]
        st.subheader("Research note")
        st.write(n["summary"])
        a, b = st.columns(2)
        a.markdown("**Bull case**\n" + "\n".join(f"- {x}" for x in n["bull_case"]))
        b.markdown("**Bear case**\n" + "\n".join(f"- {x}" for x in n["bear_case"]))
        st.markdown("**What would change the view**\n" + "\n".join(f"- {x}" for x in n["what_would_change_the_view"]))
        st.subheader("How each agent voted")
        st.dataframe([{"agent": LABELS[a], **v} for a, v in final["agent_votes"].items()], hide_index=True)
        if final["debate"]["revisions"]:
            st.subheader("What changed during the debate")
            st.dataframe(final["debate"]["revisions"], hide_index=True)
        with st.expander("Full execution trace"):
            st.json(state["trace"], expanded=False)
        st.download_button("Download trace (JSON)", path.read_text(), file_name=path.name)
        st.download_button("Download output (JSON)", json.dumps(final, indent=2, default=str),
                           file_name=f"output_{final['ticker']}.json")
