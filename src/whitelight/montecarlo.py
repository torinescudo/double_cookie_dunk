"""Monte Carlo hyper-parameter search."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Dict

from joblib import Parallel, delayed

from .portfolio import compute_metrics
from .registry import Registry
from .strategy import backtest, DEFAULT_PARAMS


@dataclass
class Result:
    model_id: str
    params: Dict[str, float]
    metrics: Dict[str, float]


def _sample(seed: int) -> Dict[str, float]:
    r = random.Random(seed)
    return {
        "ma_fast": r.randint(10, 40),
        "ma_slow": r.randint(150, 300),
        "bb_len": r.randint(10, 40),
        "bb_k": r.uniform(1.5, 3.0),
    }


def _evaluate(df, seed: int) -> Result:
    params = {**DEFAULT_PARAMS, **_sample(seed)}
    eq = backtest(df, params)
    m = compute_metrics(eq)
    return Result(str(seed), params, m.__dict__)


def run_montecarlo(df, n: int, registry: Registry) -> None:
    def worker(seed: int):
        res = _evaluate(df, seed)
        registry.insert(res.model_id, res.params, res.metrics)

    Parallel(n_jobs=-1)(delayed(worker)(s) for s in range(n))
