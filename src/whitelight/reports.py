"""Reporting helpers using matplotlib."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .portfolio import compute_drawdown


def plot_equity(equity: pd.Series, path: str) -> None:
    fig, ax = plt.subplots()
    equity.plot(ax=ax)
    ax.set_title("Equity Curve")
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown(equity: pd.Series, path: str) -> None:
    dd = compute_drawdown(equity)
    fig, ax = plt.subplots()
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    fig.savefig(path)
    plt.close(fig)
