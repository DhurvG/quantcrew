from quantcrew.decision import aggregate, detect_conflict


def op(signal, conf):
    return {"signal": signal, "confidence": conf}


def test_buy_vs_sell_is_a_conflict():
    c = detect_conflict({"technical": op("BUY", 0.7), "risk": op("SELL", 0.6)})
    assert c["conflict"] and c["opposed_pairs"] == [("risk", "technical")]


def test_unanimous_panel_is_not_a_conflict():
    assert not detect_conflict({"a": op("BUY", 0.9), "b": op("BUY", 0.6)})["conflict"]


def test_low_confidence_dissent_is_not_a_conflict():
    c = detect_conflict({"a": op("BUY", 0.8), "b": op("BUY", 0.7), "c": op("HOLD", 0.3)})
    assert not c["conflict"]


def test_trust_weights_change_the_outcome():
    ops = {"technical": op("BUY", 0.8), "fundamental": op("SELL", 0.8)}
    assert aggregate(ops, {"technical": 1.0, "fundamental": 1.0})["recommendation"] == "HOLD"
    assert aggregate(ops, {"technical": 1.5, "fundamental": 0.5})["recommendation"] == "BUY"


def test_empty_panel_holds():
    assert aggregate({}, {})["recommendation"] == "HOLD"


def test_tied_panel_is_a_conflict():
    c = detect_conflict({"technical": op("HOLD", 0.58), "risk": op("SELL", 0.6)})
    assert c["conflict"] and c["split_panel"]


def test_one_confident_agent_cannot_outvote_a_holding_majority():
    ops = {"fundamental": op("BUY", 0.75), "technical": op("HOLD", 0.55), "risk": op("HOLD", 0.55)}
    result = aggregate(ops, {})
    assert result["recommendation"] == "HOLD" and result["minority_call_overruled"]


def test_split_panel_holds_with_meaningful_confidence_and_lean():
    ops = {"fundamental": op("BUY", 0.6), "technical": op("SELL", 0.55), "risk": op("SELL", 0.68)}
    r = aggregate(ops, {})
    assert r["recommendation"] == "HOLD" and r["lean"] == "SELL"
    assert 0.3 < r["confidence"] < 0.4


def test_unanimous_hold_is_fully_confident():
    r = aggregate({"a": op("HOLD", 0.5), "b": op("HOLD", 0.7)}, {})
    assert r["recommendation"] == "HOLD" and r["confidence"] == 1.0 and r["lean"] == "NONE"
