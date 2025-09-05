import pandas as pd
from whitelight.synthetic import make_leveraged_series


def test_make_leveraged_series():
    rets = pd.Series([0.01, -0.02, 0.03])
    synth = make_leveraged_series(rets, leverage=3.0, fee_annual=0, borrow_cost_annual=0)
    expected = (1 + 3 * rets).cumprod()
    pd.testing.assert_series_equal(synth, expected, check_names=False)
