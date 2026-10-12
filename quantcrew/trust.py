"""Outcome-based trust weights, with guards against noisy short-term outcomes:
- dead band: moves smaller than TRUST_BAND are 'no move', directional calls on them are not scored
- Beta prior shrinkage: a few lucky/unlucky calls barely move the estimate
- minimum sample size: weight stays 1.0 until TRUST_MIN_SAMPLES scored calls exist
- rolling window: only the last TRUST_WINDOW calls count, so stale history ages out
- clamp: no agent is ever silenced or allowed to dominate"""
import json
from pathlib import Path

from quantcrew.config import (TRUST_BAND, TRUST_CLAMP, TRUST_FILE, TRUST_MIN_SAMPLES,
                              TRUST_PRIOR, TRUST_WINDOW)


def score_call(signal: str, fwd_return: float, band: float = TRUST_BAND) -> bool | None:
    """True = correct, False = wrong, None = not scoreable (too small a move to judge a direction)."""
    if signal == "HOLD":
        return abs(fwd_return) <= band
    if abs(fwd_return) <= band:
        return None
    return (fwd_return > 0) == (signal == "BUY")


class TrustStore:
    def __init__(self, path: Path | str = TRUST_FILE):
        self.path = Path(path)
        self.records: dict[str, list[dict]] = {}
        if self.path.exists():
            self.records = json.loads(self.path.read_text())

    def add_outcome(self, agent: str, date: str, ticker: str, signal: str, fwd_return: float) -> bool | None:
        hit = score_call(signal, fwd_return)
        if hit is not None:
            self.records.setdefault(agent, []).append(
                {"date": date, "ticker": ticker, "signal": signal, "fwd_return": round(fwd_return, 4), "hit": hit})
        return hit

    def stats(self, agent: str) -> dict:
        recent = self.records.get(agent, [])[-TRUST_WINDOW:]
        n, hits = len(recent), sum(r["hit"] for r in recent)
        posterior = (hits + TRUST_PRIOR) / (n + 2 * TRUST_PRIOR)
        if n < TRUST_MIN_SAMPLES:
            weight = 1.0
        else:
            lo, hi = TRUST_CLAMP
            weight = min(hi, max(lo, posterior / 0.5))
        return {"n": n, "hits": hits, "raw_accuracy": round(hits / n, 3) if n else None,
                "posterior_accuracy": round(posterior, 3), "weight": round(weight, 3),
                "active": n >= TRUST_MIN_SAMPLES}

    def weights(self, agents) -> dict:
        return {a: self.stats(a)["weight"] for a in agents}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2))
