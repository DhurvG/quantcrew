"""Score past live recommendations against what the price actually did, and update agent trust.

    python -m scripts.update_trust

Run it daily (e.g. a cron job or GitHub Action). Only recommendations whose horizon has fully
elapsed are scored; each agent's own signal is scored, not just the final call."""
import json
from datetime import date

import pandas as pd

from quantcrew.config import HORIZON_DAYS, RECOMMENDATIONS_FILE, SPECIALISTS
from quantcrew.tools.market_data import forward_return
from quantcrew.trust import TrustStore


def main():
    if not RECOMMENDATIONS_FILE.exists():
        print("No recommendations logged yet.")
        return
    recs = [json.loads(line) for line in RECOMMENDATIONS_FILE.read_text().splitlines() if line.strip()]
    store, scored = TrustStore(), 0
    for r in recs:
        if r.get("scored"):
            continue
        days = HORIZON_DAYS[r["horizon"]]
        # calendar-day check first (cheap), then the trading-day forward return
        if (pd.Timestamp(date.today()) - pd.Timestamp(r["as_of"])).days < days * 1.5:
            continue
        ret = forward_return(r["ticker"], r["as_of"], days)
        if ret is None:
            continue
        for agent, signal in r["agent_signals"].items():
            store.add_outcome(agent, r["as_of"], r["ticker"], signal, ret)
        r["scored"], r["fwd_return"] = True, round(ret, 4)
        scored += 1
    store.save()
    RECOMMENDATIONS_FILE.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"Scored {scored} recommendation(s).")
    for a in SPECIALISTS:
        print(a, store.stats(a))


if __name__ == "__main__":
    main()
