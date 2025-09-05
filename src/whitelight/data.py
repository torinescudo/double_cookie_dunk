"""Data utilities for Whitelight.

Handles download of market data from `yfinance` with retry logic and
alignment to the US trading calendar. Results are cached locally using
`diskcache` to avoid repeated downloads.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Tuple

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential
from dateutil.relativedelta import relativedelta

CACHE = Cache(".cache")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch(ticker: str, start: date, end: date) -> pd.DataFrame:
    return yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )


def download_price_data(
    ticker: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Download daily data for ``ticker`` between ``start`` and ``end``.

    Data are aligned to the NYSE trading calendar and cached locally. If
    less than ~20 years of data are available a warning is emitted but the
    function still returns the maximum history available.
    """
    today = date.today()
    if end is None:
        end = today
    if start is None:
        twenty_five = today - relativedelta(years=25)
        start = max(date(1999, 1, 1), twenty_five)
    key = f"{ticker}-{start}-{end}"
    if key in CACHE:
        df = CACHE[key]
    else:
        raw = _fetch(ticker, start, end)
        df = raw[["Close"]].rename(columns={"Close": "close"})
        cal = mcal.get_calendar("XNYS")
        sched = cal.schedule(start_date=start, end_date=end)
        all_days = mcal.date_range(sched, frequency="1D")
        df = df.reindex(all_days, method="ffill")
        CACHE[key] = df
    if len(df) < 252 * 20:
        import warnings

        warnings.warn(
            f"Historical data for {ticker} less than 20 years ({len(df)} days)",
            RuntimeWarning,
        )
    return df


def split_is_oos(data: pd.DataFrame, ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into in-sample and out-of-sample segments."""
    idx = int(len(data) * ratio)
    return data.iloc[:idx].copy(), data.iloc[idx:].copy()
