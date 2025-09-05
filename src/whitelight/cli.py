"""Command line interface using Typer."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from . import data, synthetic, strategy, portfolio, registry, reports

app = typer.Typer(help="Whitelight backtesting toolkit")


def _extract_close(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the ``close`` column as a Series indexed by date.

    Depending on caching or upstream changes, ``data.download_price_data`` may
    yield DataFrames with a ``Ticker`` level in the index or with an extra
    column dimension. This helper normalises the structure so callers receive a
    plain ``Series`` indexed only by date and named according to ``name``.
    """

    series = df["close"]
    if isinstance(series, pd.DataFrame):
        # Drop any redundant column level
        series = series.iloc[:, 0]
    if isinstance(series.index, pd.MultiIndex) and "Ticker" in series.index.names:
        series = series.droplevel("Ticker")
    return series.rename(name)


@app.command()
def download(start: str = "1999-01-01", end: Optional[str] = None) -> None:
    """Download core datasets and cache them locally."""
    tickers = ["^NDX", "^GSPC", "TQQQ", "SQQQ"]
    for t in tickers:
        df = data.download_price_data(
            t, pd.to_datetime(start).date(), pd.to_datetime(end).date() if end else None
        )
        typer.echo(f"Downloaded {t}: {len(df)} rows")


@app.command("make-synthetic")
def make_synthetic() -> None:
    ndx = data.download_price_data("^NDX")
    rets = ndx["close"].pct_change().dropna()
    t = synthetic.make_leveraged_series(rets, 3.0)
    s = synthetic.make_leveraged_series(rets, -3.0)
    out = pd.DataFrame({"tqqq": t, "sqqq": s})
    out.to_parquet("synthetic.parquet")
    typer.echo("Saved synthetic series to synthetic.parquet")


@app.command()
def backtest(config: Optional[Path] = None) -> None:
    """Run backtest using cached data."""
    ndx = data.download_price_data("^NDX")
    tqqq = data.download_price_data("TQQQ")
    sqqq = data.download_price_data("SQQQ")
    df = pd.concat(
        [
            _extract_close(ndx, "ndx"),
            _extract_close(tqqq, "tqqq"),
            _extract_close(sqqq, "sqqq"),
        ],
        axis=1,
        join="inner",
    )
    equity = strategy.backtest(df)
    m = portfolio.compute_metrics(equity)
    reports.plot_equity(equity, "equity.png")
    reports.plot_drawdown(equity, "drawdown.png")
    typer.echo(portfolio.metrics_to_json(m))


@app.command()
def montecarlo(n: int = 100) -> None:
    ndx = data.download_price_data("^NDX")
    tqqq = data.download_price_data("TQQQ")
    sqqq = data.download_price_data("SQQQ")
    df = pd.concat(
        [
            _extract_close(ndx, "ndx"),
            _extract_close(tqqq, "tqqq"),
            _extract_close(sqqq, "sqqq"),
        ],
        axis=1,
        join="inner",
    )
    reg = registry.Registry()
    from . import montecarlo as mc

    best = mc.run_montecarlo(df, n, reg)
    typer.echo("Monte Carlo completed")
    eq_best = strategy.backtest(df, best.params)
    reports.plot_equity(eq_best, "best_equity.png")
    typer.echo(
        f"Best model {best.model_id} Sharpe {best.metrics.get('sharpe', 0):.2f}"
        " saved to best_equity.png"
    )


@app.command()
def report(model_id: str) -> None:
    reg = registry.Registry()
    df = reg.top("sharpe")
    row = df[df["model_id"] == model_id]
    if row.empty:
        typer.echo("model not found")
    else:
        typer.echo(row.to_string(index=False))


@app.command()
def top(metric: str = "sharpe", k: int = 20) -> None:
    reg = registry.Registry()
    df = reg.top(metric, k)
    typer.echo(df.to_string(index=False))


if __name__ == "__main__":
    app()
