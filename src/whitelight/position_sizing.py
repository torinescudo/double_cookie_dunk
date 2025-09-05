"""Position sizing helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def size_by_ma_distance(
    d20: pd.Series,
    d250: pd.Series,
    w1: float = 0.6,
    w2: float = 0.4,
    clip_long: float = 1.0,
    clip_short: float = 1.0,
    mode: str = "linear",
) -> pd.Series:
    """Combine MA distances into a raw exposure signal."""
    raw = w1 * d20 + w2 * d250
    if mode == "tanh":
        raw = np.tanh(raw)
    return raw.clip(-clip_short, clip_long)


def momentum_boost(
    exposure: pd.Series,
    z: pd.Series,
    thresh: float = 1.0,
    boost_factor: float = 1.5,
) -> pd.Series:
    """Amplify exposure when |z| exceeds threshold."""
    boosted = exposure.copy()
    mask = z.abs() > thresh
    boosted[mask] = boosted[mask] * boost_factor
    return boosted.clip(-1, 1)


def compute_target_exposure(
    price: pd.Series,
    ma20: pd.Series,
    ma250: pd.Series,
    z: pd.Series,
    w1: float = 0.6,
    w2: float = 0.4,
    clip_long: float = 1.0,
    clip_short: float = 1.0,
    mode: str = "linear",
    momentum_thresh: float = 1.0,
    boost_factor: float = 1.5,
) -> pd.Series:
    d20 = price / ma20 - 1.0
    d250 = price / ma250 - 1.0
    base = size_by_ma_distance(d20, d250, w1, w2, clip_long, clip_short, mode)
    return momentum_boost(base, z, momentum_thresh, boost_factor)
