"""Core strategy logic.

The strategy derives an exposure between -1 and +1 based on NASDAQ-100
price relative to moving averages and Bollinger bands.
"""
from __future__ import annotations

import pandas as pd

from .indicators import bollinger, ma
from .position_sizing import compute_target_exposure


DEFAULT_PARAMS = {
    "ma_fast": 20,
    "ma_slow": 250,
    "bb_len": 20,
    "bb_k": 2.0,
    "w1": 0.6,
    "w2": 0.4,
    "clip_long": 1.0,
    "clip_short": 1.0,
    "momentum_thresh": 1.0,
    "boost_factor": 1.5,
    "slippage_bps": 2,
    "commission_bps": 1,
}


def generate_signals(prices_ndx: pd.Series, params: dict | None = None) -> pd.Series:
    p = {**DEFAULT_PARAMS, **(params or {})}
    ma20 = ma(prices_ndx, p["ma_fast"])
    ma250 = ma(prices_ndx, p["ma_slow"])
    _, _, _, z = bollinger(prices_ndx, p["bb_len"], p["bb_k"])
    exposure = compute_target_exposure(
        prices_ndx,
        ma20,
        ma250,
        z,
        p["w1"],
        p["w2"],
        p["clip_long"],
        p["clip_short"],
        "linear",
        p["momentum_thresh"],
        p["boost_factor"],
    )
    return exposure.dropna()


def backtest(df: pd.DataFrame, params: dict | None = None) -> pd.Series:
    """Run a simple backtest given price data.

    ``df`` must contain columns ``ndx``, ``tqqq`` and ``sqqq`` representing
    price series of the index and ETFs respectively.
    """
    exposure = generate_signals(df["ndx"], params)
    tqqq_ret = df["tqqq"].pct_change().reindex(exposure.index).fillna(0)
    sqqq_ret = df["sqqq"].pct_change().reindex(exposure.index).fillna(0)
    target_ret = exposure.clip(lower=0) * tqqq_ret + exposure.clip(upper=0) * sqqq_ret
    # costs on turnover
    turns = exposure.diff().abs().fillna(0)
    p = {**DEFAULT_PARAMS, **(params or {})}
    slippage = turns * (p["slippage_bps"] / 10000)
    commission = turns * (p["commission_bps"] / 10000)
    net_ret = target_ret - slippage - commission
    equity = (1 + net_ret).cumprod()
    return equity
