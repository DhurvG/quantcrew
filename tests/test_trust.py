from quantcrew.config import TRUST_MIN_SAMPLES
from quantcrew.trust import TrustStore, score_call


def test_small_moves_are_not_scored_for_directional_calls():
    assert score_call("BUY", 0.005) is None
    assert score_call("HOLD", 0.005) is True
    assert score_call("SELL", -0.05) is True
    assert score_call("BUY", -0.05) is False


def test_weight_stays_neutral_until_min_samples(tmp_path):
    s = TrustStore(tmp_path / "t.json")
    for i in range(TRUST_MIN_SAMPLES - 1):
        s.add_outcome("technical", f"2024-01-{i+1:02d}", "INFY.NS", "BUY", 0.05)
    assert s.stats("technical")["weight"] == 1.0


def test_consistently_right_agent_gains_weight_and_wrong_one_loses(tmp_path):
    s = TrustStore(tmp_path / "t.json")
    for i in range(40):
        s.add_outcome("good", str(i), "X", "BUY", 0.05)
        s.add_outcome("bad", str(i), "X", "BUY", -0.05)
    assert s.stats("good")["weight"] > 1.3
    assert s.stats("bad")["weight"] <= 0.6
    s.save()
    assert TrustStore(tmp_path / "t.json").stats("good")["n"] == 40


def test_prior_shrinks_a_short_lucky_streak(tmp_path):
    s = TrustStore(tmp_path / "t.json")
    for i in range(TRUST_MIN_SAMPLES):
        s.add_outcome("lucky", str(i), "X", "BUY", 0.05)
    # 15/15 raw accuracy, but the Beta(5,5) prior keeps the estimate well below 1
    assert s.stats("lucky")["posterior_accuracy"] < 0.85
