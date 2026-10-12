"""Run one analysis from the command line and save the execution trace + output.

    python -m scripts.run_once --ticker INFY --question "Is Infosys a buy for the next 6 months?"

Use the files this writes as your real sample output and execution trace for CA3."""
import argparse
import json
from pathlib import Path

from quantcrew.graph import run_analysis
from quantcrew.trace import save_trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--question", default="Should I buy, hold or sell this stock?")
    ap.add_argument("--horizon", choices=["short", "medium", "long"])
    ap.add_argument("--out", default="docs/samples")
    a = ap.parse_args()

    state = run_analysis(a.ticker, a.question, horizon=a.horizon)
    trace_path = save_trace(state)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_output.json").write_text(json.dumps(state["final"], indent=2, default=str))
    (out / "sample_input.json").write_text(json.dumps(
        {"ticker": a.ticker, "question": a.question, "horizon": a.horizon, "as_of": None}, indent=2))
    f = state["final"]
    print(f"{f.get('ticker')}: {f.get('recommendation', f.get('error'))} "
          f"(confidence {f.get('confidence')}, debate rounds {f.get('debate', {}).get('rounds')})")
    print(f"trace  -> {trace_path}\noutput -> {out / 'sample_output.json'}")


if __name__ == "__main__":
    main()
