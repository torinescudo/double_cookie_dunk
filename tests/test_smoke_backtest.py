import numpy as np
import pandas as pd

from whitelight import synthetic, strategy, portfolio, registry


def test_smoke_backtest(tmp_path):
    np.random.seed(0)
    rets = pd.Series(np.random.normal(0, 0.01, 300))
    ndx = 100 * (1 + rets).cumprod()
    tqqq = synthetic.make_leveraged_series(rets, 3.0)
    sqqq = synthetic.make_leveraged_series(rets, -3.0)
    df = pd.DataFrame({"ndx": ndx, "tqqq": tqqq, "sqqq": sqqq})
    eq = strategy.backtest(df)
    assert eq.iloc[-1] != 0
    m = portfolio.compute_metrics(eq, ndx)
    reg = registry.Registry(path=str(tmp_path / "reg.sqlite"))
    reg.insert("model", {}, m.__dict__)
    top = reg.top("sharpe")
    assert "model" in top["model_id"].values
