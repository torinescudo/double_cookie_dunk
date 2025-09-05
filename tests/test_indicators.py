import pandas as pd
from whitelight.indicators import ma, bollinger


def test_ma():
    s = pd.Series([1, 2, 3, 4, 5])
    result = ma(s, 2)
    expected = pd.Series([float("nan"), 1.5, 2.5, 3.5, 4.5])
    pd.testing.assert_series_equal(result, expected)


def test_bollinger():
    s = pd.Series([1, 2, 3, 4, 5])
    m, up, low, z = bollinger(s, 2, 2)
    assert m.iloc[-1] == 4.5
    assert round(z.iloc[-1], 6) == round((5 - 4.5) / s.iloc[-2:].std(), 6)
