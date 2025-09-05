"""Portfolio statistics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class Metrics:
    cagr: float
    vol: float
    sharpe: float
    sortino: float
    mdd: float
    calmar: float
    skew: float
    kurtosis: float
    trades: int
    avg_trade: float
    hit_rate: float
    alpha: float
    beta: float


def compute_drawdown(equity: pd.Series) -> pd.Series:
    cummax = equity.cummax()
    return equity / cummax - 1


def compute_metrics(
    equity: pd.Series,
    bench: pd.Series | None = None,
) -> Metrics:
    returns = equity.pct_change().dropna()
    ann_factor = 252
    cagr = equity.iloc[-1] ** (ann_factor / len(equity)) - 1
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = returns.mean() / returns.std() * np.sqrt(ann_factor)
    downside = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = returns.mean() / downside if downside != 0 else np.nan
    dd = compute_drawdown(equity)
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    skew = returns.skew()
    kurt = returns.kurtosis()
    trades = (returns != 0).sum()
    avg_trade = returns.mean()
    hit_rate = (returns > 0).mean()
    alpha = beta = np.nan
    if bench is not None:
        aligned = pd.concat([returns, bench.pct_change().reindex(returns.index).dropna()], axis=1)
        aligned.columns = ["r", "b"]
        X = sm.add_constant(aligned["b"])
        model = sm.OLS(aligned["r"], X).fit()
        alpha = model.params["const"] * ann_factor
        beta = model.params["b"]
    return Metrics(cagr, vol, sharpe, sortino, mdd, calmar, skew, kurt, trades, avg_trade, hit_rate, alpha, beta)


def metrics_to_json(metrics: Metrics) -> str:
    return json.dumps(metrics.__dict__, indent=2)
