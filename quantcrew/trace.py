"""Execution-trace events. Every message between agents and every tool call is recorded,
then saved as JSON - this file is the CA3 'Agent Execution Trace' artifact."""
import json
from datetime import datetime, timezone
from pathlib import Path

from quantcrew.config import TRACE_DIR


def event(sender: str, receiver: str, kind: str, content) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "from": sender,
        "to": receiver,
        "type": kind,
        "content": content,
    }


def save_trace(state: dict, directory: Path = TRACE_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"run_{state.get('ticker', 'NA')}_{stamp}.json"
    payload = {
        "input": {k: state.get(k) for k in ("ticker", "question", "horizon", "as_of", "mode")},
        "plan": state.get("plan"),
        "debate_rounds": state.get("debate_round", 0),
        "final": state.get("final"),
        "events": sorted(state.get("trace", []), key=lambda e: e["ts"]),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
