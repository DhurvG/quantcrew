"""Start the live app from real track records: copy the per-agent outcomes learned in a backtest
into the live trust store (data/trust.json).

    python -m scripts.seed_trust                       # from results/trust_crew.json
    python -m scripts.seed_trust --source results/trust_crew.json

Only agents present in the source are replaced (backtests cover Technical and Risk); Fundamental
and Sentiment keep whatever live record they have, starting neutral at weight 1.0."""
import argparse
import json
from pathlib import Path

from quantcrew.config import RESULTS_DIR, SPECIALISTS, TRUST_FILE
from quantcrew.trust import TrustStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(RESULTS_DIR / "trust_crew.json"))
    ap.add_argument("--dest", default=str(TRUST_FILE))
    a = ap.parse_args()
    src = Path(a.source)
    if not src.exists():
        print(f"{src} not found. Run the crew backtest first.")
        return
    seeded = json.loads(src.read_text())
    store = TrustStore(a.dest)
    for agent, records in seeded.items():
        store.records[agent] = [r | {"source": f"backtest:{src.stem}"} for r in records]
    store.save()
    for agent in SPECIALISTS:
        print(agent, store.stats(agent))


if __name__ == "__main__":
    main()
