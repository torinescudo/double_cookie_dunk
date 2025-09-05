import pandas as pd
from whitelight.portfolio import compute_metrics


def test_metrics_alpha_beta():
    bench = pd.Series([100, 101, 103, 102])
    asset = pd.Series([100, 102, 106, 104])
    m = compute_metrics(asset, bench)
    assert abs(m.beta - 2) < 0.1
    assert abs(m.alpha) < 0.1
    assert m.sharpe > 0
