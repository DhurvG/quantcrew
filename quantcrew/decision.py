"""Pure decision logic: conflict detection and trust-weighted aggregation (unit-tested)."""
SIGNAL_VALUE = {"BUY": 1, "HOLD": 0, "SELL": -1}
THRESHOLD = 0.4


def detect_conflict(opinions: dict) -> dict:
    """A conflict exists when one agent says BUY and another SELL, when the panel is split with no
    majority view, or when a confident (>=0.6) agent disagrees with the majority."""
    if len(opinions) < 2:
        return {"conflict": False, "split_panel": False, "opposed_pairs": [],
                "confident_dissenters": [], "signals": {}}
    signals = {a: o["signal"] for a, o in opinions.items()}
    opposed = [(a, b) for a in signals for b in signals
               if a < b and {signals[a], signals[b]} == {"BUY", "SELL"}]
    counts = {s: list(signals.values()).count(s) for s in set(signals.values())}
    top = max(counts.values())
    leaders = [s for s, n in counts.items() if n == top]
    split = len(leaders) > 1  # tie: no majority view exists
    confident_dissent = [] if split else [
        a for a, o in opinions.items() if o["signal"] != leaders[0] and o["confidence"] >= 0.6]
    return {
        "conflict": bool(opposed or confident_dissent or split),
        "split_panel": split,
        "opposed_pairs": opposed,
        "confident_dissenters": confident_dissent,
        "signals": signals,
    }


def aggregate(opinions: dict, weights: dict, threshold: float = THRESHOLD) -> dict:
    """Score = sum(w * c * s) / sum(w * c) with s in {-1, 0, 1}.
    BUY if score > threshold, SELL if < -threshold, else HOLD. A BUY/SELL also needs more than half
    of the weighted conviction behind it, so one loud agent cannot outvote a holding majority.
    Confidence: for BUY/SELL, the share of weighted conviction agreeing; for HOLD,
    1 - the largest share pushing BUY or SELL. 'lean' reports which way the panel tilts."""
    if not opinions:
        return {"recommendation": "HOLD", "score": 0.0, "confidence": 0.0, "lean": "NONE",
                "minority_call_overruled": False, "contributions": {}}
    contrib, total = {}, 0.0
    for agent, op in opinions.items():
        w = weights.get(agent, 1.0) * op["confidence"]
        contrib[agent] = {"weight": round(weights.get(agent, 1.0), 3), "signal": op["signal"],
                          "confidence": op["confidence"], "vote": round(w * SIGNAL_VALUE[op["signal"]], 3)}
        total += w
    score = sum(c["vote"] for c in contrib.values()) / total if total else 0.0
    rec = "BUY" if score > threshold else "SELL" if score < -threshold else "HOLD"

    def support(signal):
        return sum(weights.get(a, 1.0) * o["confidence"] for a, o in opinions.items() if o["signal"] == signal)

    overruled = bool(rec != "HOLD" and total and support(rec) / total < 0.5)
    if overruled:
        rec = "HOLD"
    if rec == "HOLD":
        conf = 1 - max(support("BUY"), support("SELL")) / total if total else 0.0
    else:
        conf = support(rec) / total if total else 0.0
    lean = "BUY" if score > 0.05 else "SELL" if score < -0.05 else "NONE"
    return {"recommendation": rec, "score": round(score, 3), "confidence": round(conf, 2),
            "lean": lean, "minority_call_overruled": overruled, "contributions": contrib}
