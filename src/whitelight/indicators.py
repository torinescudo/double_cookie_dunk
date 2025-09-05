"""Indicator functions used by the strategy."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(length).mean()


def bollinger(
    series: pd.Series, length: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands and z-score."""
    m = ma(series, length)
    s = series.rolling(length).std()
    upper = m + k * s
    lower = m - k * s
    z = (series - m) / s
    return m, upper, lower, z


def atr(series: pd.Series, length: int = 14) -> pd.Series:
    """Average true range using simple method.

    Expects a close price series; returns rolling standard deviation
    scaled by square-root of time as a crude proxy.
    """
    return series.pct_change().rolling(length).std() * np.sqrt(length)
