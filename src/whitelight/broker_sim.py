"""Broker simulation and adapter interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BrokerAdapter(Protocol):
    """Minimal interface for future real-broker connections."""

    def submit_order(self, symbol: str, qty: int, side: str) -> None:
        ...

    def get_position(self, symbol: str) -> float:
        ...


@dataclass
class Fill:
    symbol: str
    qty: int
    price: float
    side: str  # 'buy' or 'sell'


def simulate_fill(price: float, slippage_bps: float, commission_bps: float, side: str) -> float:
    """Return execution price including slippage/commission."""
    slip = price * slippage_bps / 10000
    comm = price * commission_bps / 10000
    adj = price + slip if side == "buy" else price - slip
    return adj + comm
