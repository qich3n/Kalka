"""
DuckDB persistence layer.

Stores raw Binance candles, funding rates, open interest, engineered features,
training labels, model predictions, and backtest results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import config


class Database:
    """Thin wrapper around a local DuckDB file."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or config.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        self._migrate_candles_schema()
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                timestamp TIMESTAMP PRIMARY KEY,
                funding_rate DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                timestamp TIMESTAMP PRIMARY KEY,
                open_interest DOUBLE
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS brti_ticks (
                timestamp TIMESTAMP PRIMARY KEY,
                price DOUBLE,
                source VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_samples (
                id INTEGER PRIMARY KEY,
                window_start TIMESTAMP,
                observation_time TIMESTAMP,
                strike DOUBLE,
                label INTEGER,
                features JSON,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS training_samples_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                btc_price DOUBLE,
                strike DOUBLE,
                minutes_remaining DOUBLE,
                model_probability DOUBLE,
                kalshi_implied_prob DOUBLE,
                edge DOUBLE,
                recommendation VARCHAR,
                top_factors JSON,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS predictions_id_seq START 1
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY,
                run_id VARCHAR,
                timestamp TIMESTAMP,
                strike DOUBLE,
                model_probability DOUBLE,
                kalshi_implied_prob DOUBLE,
                actual_label INTEGER,
                recommendation VARCHAR,
                pnl DOUBLE,
                created_at TIMESTAMP DEFAULT current_timestamp
            )
        """)
        self.conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS backtest_results_id_seq START 1
        """)

    def _migrate_candles_schema(self) -> None:
        """Ensure candles table supports multiple exchanges."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                exchange VARCHAR NOT NULL DEFAULT 'binance',
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (exchange, timestamp)
            )
        """)
        # Migrate legacy single-exchange schema (timestamp-only PK)
        cols = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'candles'"
        ).fetchall()
        col_names = {c[0] for c in cols}
        if col_names and "exchange" not in col_names:
            self.conn.execute("""
                CREATE TABLE candles_migrated (
                    exchange VARCHAR NOT NULL DEFAULT 'binance',
                    timestamp TIMESTAMP NOT NULL,
                    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
                    PRIMARY KEY (exchange, timestamp)
                )
            """)
            self.conn.execute("""
                INSERT INTO candles_migrated (exchange, timestamp, open, high, low, close, volume)
                SELECT 'binance', timestamp, open, high, low, close, volume FROM candles
            """)
            self.conn.execute("DROP TABLE candles")
            self.conn.execute("ALTER TABLE candles_migrated RENAME TO candles")

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def upsert_candles(self, df: pd.DataFrame, exchange: str = "binance") -> int:
        """Insert or replace candle rows for a given exchange."""
        if df.empty:
            return 0
        df = df.copy()
        df["exchange"] = exchange
        self.conn.execute(
            "DELETE FROM candles WHERE exchange = ? AND timestamp IN (SELECT timestamp FROM df)",
            [exchange],
        )
        self.conn.execute(
            "INSERT INTO candles SELECT exchange, timestamp, open, high, low, close, volume FROM df"
        )
        return len(df)

    def get_candles(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        exchange: str = "binance",
    ) -> pd.DataFrame:
        """Fetch candles for one exchange, ordered by timestamp ascending."""
        query = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE exchange = ?"
        params: list = [exchange]
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        query += " ORDER BY timestamp ASC"
        if limit:
            query += f" LIMIT {limit}"
        return self.conn.execute(query, params).df()

    def get_all_exchange_candles(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch candles for all exchanges."""
        result = {}
        for ex in config.EXCHANGES:
            df = self.get_candles(start=start, end=end, limit=limit, exchange=ex)
            if not df.empty:
                result[ex] = df
        return result

    def get_latest_candle_time(self, exchange: str = "binance") -> datetime | None:
        """Return the most recent candle timestamp for an exchange."""
        row = self.conn.execute(
            "SELECT MAX(timestamp) FROM candles WHERE exchange = ?",
            [exchange],
        ).fetchone()
        return row[0] if row and row[0] else None

    # ------------------------------------------------------------------
    # Funding & open interest
    # ------------------------------------------------------------------

    def upsert_funding_rates(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.conn.execute(
            "DELETE FROM funding_rates WHERE timestamp IN (SELECT timestamp FROM df)"
        )
        self.conn.execute("INSERT INTO funding_rates SELECT * FROM df")
        return len(df)

    def get_latest_funding_rate(self) -> float | None:
        row = self.conn.execute(
            "SELECT funding_rate FROM funding_rates ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def upsert_open_interest(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.conn.execute(
            "DELETE FROM open_interest WHERE timestamp IN (SELECT timestamp FROM df)"
        )
        self.conn.execute("INSERT INTO open_interest SELECT * FROM df")
        return len(df)

    def get_latest_open_interest(self) -> float | None:
        row = self.conn.execute(
            "SELECT open_interest FROM open_interest ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # BRTI ticks (CF Benchmarks settlement index)
    # ------------------------------------------------------------------

    def upsert_brti_tick(
        self, timestamp: datetime, price: float, source: str
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO brti_ticks (timestamp, price, source)
            VALUES (?, ?, ?)
            """,
            [timestamp, price, source],
        )

    def get_brti_ticks(self, limit: int = 120) -> pd.DataFrame:
        return self.conn.execute(
            f"SELECT * FROM brti_ticks ORDER BY timestamp DESC LIMIT {limit}"
        ).df()

    def get_brti_ticks_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch BRTI ticks in a time range, ordered ascending."""
        query = "SELECT timestamp, price, source FROM brti_ticks"
        clauses: list[str] = []
        if start:
            clauses.append(f"timestamp >= '{start.isoformat()}'")
        if end:
            clauses.append(f"timestamp <= '{end.isoformat()}'")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp ASC"
        return self.conn.execute(query).df()

    def get_brti_60s_average(self) -> float | None:
        """Average of BRTI ticks stored in the last 60 seconds."""
        row = self.conn.execute("""
            SELECT AVG(price) FROM brti_ticks
            WHERE timestamp >= current_timestamp - INTERVAL 60 SECOND
        """).fetchone()
        return float(row[0]) if row and row[0] else None

    def get_latest_brti(self) -> tuple[float, str] | None:
        row = self.conn.execute(
            "SELECT price, source FROM brti_ticks ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return (row[0], row[1]) if row else None

    # ------------------------------------------------------------------
    # Training samples
    # ------------------------------------------------------------------

    def save_training_samples(self, samples: list[dict[str, Any]]) -> int:
        """Persist labeled feature vectors for model training."""
        if not samples:
            return 0
        for sample in samples:
            self.conn.execute(
                """
                INSERT INTO training_samples (id, window_start, observation_time, strike, label, features)
                VALUES (
                    nextval('training_samples_id_seq'),
                    ?, ?, ?, ?, ?
                )
                """,
                [
                    sample["window_start"],
                    sample["observation_time"],
                    sample["strike"],
                    sample["label"],
                    json.dumps(sample["features"]),
                ],
            )
        return len(samples)

    def get_training_data(self) -> pd.DataFrame:
        """Load all training samples as a flat DataFrame."""
        rows = self.conn.execute(
            "SELECT window_start, observation_time, strike, label, features FROM training_samples"
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        records = []
        for window_start, obs_time, strike, label, features_json in rows:
            features = json.loads(features_json)
            features["window_start"] = window_start
            features["observation_time"] = obs_time
            features["strike"] = strike
            features["label"] = label
            records.append(features)
        return pd.DataFrame(records)

    def clear_training_samples(self) -> None:
        self.conn.execute("DELETE FROM training_samples")

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def save_prediction(self, record: dict[str, Any]) -> None:
        """Store a live prediction for historical tracking."""
        self.conn.execute(
            """
            INSERT INTO predictions (
                id, timestamp, btc_price, strike, minutes_remaining,
                model_probability, kalshi_implied_prob, edge,
                recommendation, top_factors
            ) VALUES (
                nextval('predictions_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                record.get("timestamp", datetime.now(timezone.utc)),
                record["btc_price"],
                record["strike"],
                record["minutes_remaining"],
                record["model_probability"],
                record["kalshi_implied_prob"],
                record["edge"],
                record["recommendation"],
                json.dumps(record["top_factors"]),
            ],
        )

    def get_predictions(self, limit: int = 100) -> pd.DataFrame:
        return self.conn.execute(
            f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}"
        ).df()

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def save_backtest_results(self, run_id: str, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0
        for r in results:
            self.conn.execute(
                """
                INSERT INTO backtest_results (
                    id, run_id, timestamp, strike, model_probability,
                    kalshi_implied_prob, actual_label, recommendation, pnl
                ) VALUES (
                    nextval('backtest_results_id_seq'), ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    run_id,
                    r["timestamp"],
                    r["strike"],
                    r["model_probability"],
                    r.get("kalshi_implied_prob"),
                    r["actual_label"],
                    r["recommendation"],
                    r.get("pnl", 0.0),
                ],
            )
        return len(results)

    def get_backtest_summary(self, run_id: str) -> dict[str, Any]:
        df = self.conn.execute(
            "SELECT * FROM backtest_results WHERE run_id = ?", [run_id]
        ).df()
        if df.empty:
            return {}
        trades = df[df["recommendation"] != "NO TRADE"]
        wins = trades[trades["pnl"] > 0]
        return {
            "run_id": run_id,
            "total_predictions": len(df),
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) if len(trades) else 0.0,
            "total_pnl": float(trades["pnl"].sum()) if len(trades) else 0.0,
            "avg_edge": float(df["model_probability"].mean() - 0.5),
        }

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args) -> None:
        self.close()
