"""SQLite registry for Monte Carlo runs."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd


class Registry:
    def __init__(self, path: str = "registry.sqlite") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (model_id TEXT PRIMARY KEY, params TEXT, metrics TEXT)"
        )

    def insert(self, model_id: str, params: Dict, metrics: Dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
            (model_id, json.dumps(params, default=float), json.dumps(metrics, default=float)),
        )
        self.conn.commit()

    def top(self, metric: str, k: int = 20) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM runs", self.conn)
        df[metric] = df["metrics"].apply(lambda x: json.loads(x).get(metric, 0))
        return df.sort_values(metric, ascending=False).head(k)
