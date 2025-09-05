# Whitelight

Backtester y Monte Carlo para estrategias sistemáticas TQQQ/SQQQ. Este paquete descarga datos de `yfinance`, genera series apalancadas sintéticas y ejecuta un backtest EOD totalmente programático. Incluye ejemplos de uso en la CLI y disclaimers de riesgo.

## Uso rápido

```bash
pip install -e .
whitelight download --start 1999-01-01 --end 2024-12-31
whitelight make-synthetic
whitelight backtest
whitelight montecarlo --n 100
```

`download` obtiene los precios diarios de `yfinance` con reintentos y los alinea al calendario de cierre de NYSE. `make-synthetic` construye series ±3× a partir del NASDAQ‑100 con rebalanceo diario, fees y *volatility drag*. `backtest` ejecuta la estrategia base sobre TQQQ/SQQQ reales o sintéticos y produce curvas y métricas. `montecarlo` realiza una búsqueda aleatoria de hiperparámetros y guarda cada iteración en `registry.sqlite`.

**Descargo de responsabilidad:** "CAGR backtest ~80% y MDD ~49% observados históricamente no garantizan resultados futuros".
