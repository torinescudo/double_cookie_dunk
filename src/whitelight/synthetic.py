"""Synthetic leveraged series for TQQQ/SQQQ.

The real ETFs only have history from 2010. For earlier periods we
construct synthetic paths based on NASDAQ-100 index returns with daily
rebalancing and explicit fee/borrow costs.
"""
from __future__ import annotations

import pandas as pd


def make_leveraged_series(
    index_returns: pd.Series,
    leverage: float = 3.0,
    fee_annual: float = 0.0095,
    borrow_cost_annual: float = 0.0,
    daily_rebalance: bool = True,
) -> pd.Series:
    """Create a leveraged return series from unlevered ``index_returns``.

    Parameters
    ----------
    index_returns : pd.Series
        Daily return series of the underlying index.
    leverage : float, default 3.0
        Leverage factor, negative for inverse exposure.
    fee_annual : float, default 0.0095
        Annual management fee applied daily.
    borrow_cost_annual : float, default 0.0
        Additional annual borrowing cost for short/inverse products.
    daily_rebalance : bool, default True
        Keep leverage constant via daily rebalancing.
    """
    days = index_returns.shape[0]
    fee_daily = (1 + fee_annual) ** (1 / 252) - 1
    borrow_daily = (1 + borrow_cost_annual) ** (1 / 252) - 1
    if daily_rebalance:
        levered_returns = leverage * index_returns
    else:
        levered_returns = (1 + index_returns).cumprod().pct_change() * leverage
    levered_returns = levered_returns - fee_daily - borrow_daily
    prices = (1 + levered_returns).cumprod()
    return prices.rename("synthetic")
