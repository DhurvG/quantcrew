import numpy as np
import pandas as pd

from quantcrew.tools.risk_metrics import beta, historical_var, max_drawdown


def test_max_drawdown_simple_path():
    assert max_drawdown(pd.Series([100, 120, 90, 130])) == -0.25


def test_var_is_positive_loss():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    assert 0.01 < historical_var(r) < 0.025


def test_beta_of_series_with_itself_is_one():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0, 0.01, 200))
    assert abs(beta(r, r) - 1) < 1e-9
    assert abs(beta(2 * r, r) - 2) < 1e-9
