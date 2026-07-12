"""
Walk-forward backtesting engine.

Simulates trading decisions on historical 15-minute windows using the
trained XGBoost model. Computes win rate, PnL, and calibration metrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from src.db.database import Database
from src.data.aggregator import ExchangeAggregator
from src.features.engineering import FEATURE_COLUMNS, FeatureEngineer
from src.models.trainer import ModelTrainer


class BacktestEngine:
    """
    Backtest the model on historical candle data.

    For each 15-minute window, generates features at a fixed observation
    offset and evaluates whether the model's recommendation would have
    been profitable.
    """

    def __init__(self, db: Database | None = None) -> None:
        self.db = db or Database()
        self.engineer = FeatureEngineer()
        self.trainer = ModelTrainer(self.db)

    def run(
        self,
        candles: pd.DataFrame | None = None,
        observation_offset: int = 7,
        save: bool = True,
    ) -> dict:
        """
        Execute a backtest and return summary statistics.

        PnL assumes $1 contracts: BUY YES wins (1 - entry_price) on correct
        calls, loses entry_price on incorrect calls. Simplified for research.
        """
        self.trainer.load_model()

        if candles is None:
            candles = self.db.get_candles()
        if candles.empty:
            raise ValueError("No candle data available for backtesting")

        candles = candles.sort_values("timestamp").reset_index(drop=True)
        all_candles = self.db.get_all_exchange_candles()
        composite = ExchangeAggregator.build_composite_candles(all_candles)
        brti_ticks = self.db.get_brti_ticks_range()
        samples = self.engineer.generate_training_samples(
            candles,
            brti_ticks=brti_ticks if not brti_ticks.empty else None,
            composite_candles=composite if not composite.empty else None,
            observation_offsets=[observation_offset],
        )

        if not samples:
            raise ValueError("No backtest samples generated")

        run_id = str(uuid.uuid4())[:8]
        results = []

        for sample in samples:
            features = sample["features"]
            X = self.engineer.features_to_array(features).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)
            prob = float(self.trainer.predict_probability(X)[0])
            label = sample["label"]

            if prob > config.YES_THRESHOLD:
                rec = "BUY YES"
                # Simplified PnL: win = +(1 - prob), lose = -prob
                pnl = (1 - prob) if label == 1 else -prob
            elif prob < config.NO_THRESHOLD:
                rec = "BUY NO"
                pnl = (1 - prob) if label == 0 else -prob
            else:
                rec = "NO TRADE"
                pnl = 0.0

            results.append({
                "timestamp": sample["observation_time"],
                "strike": sample["strike"],
                "model_probability": prob,
                "kalshi_implied_prob": None,
                "actual_label": label,
                "recommendation": rec,
                "pnl": pnl,
            })

        if save:
            self.db.save_backtest_results(run_id, results)

        trades = [r for r in results if r["recommendation"] != "NO TRADE"]
        wins = [r for r in trades if r["pnl"] > 0]
        yes_trades = [r for r in trades if r["recommendation"] == "BUY YES"]
        no_trades = [r for r in trades if r["recommendation"] == "BUY NO"]

        probs = [r["model_probability"] for r in results]
        labels = [r["actual_label"] for r in results]

        summary = {
            "run_id": run_id,
            "total_windows": len(results),
            "total_trades": len(trades),
            "yes_trades": len(yes_trades),
            "no_trades": len(no_trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "total_pnl": sum(r["pnl"] for r in trades),
            "avg_probability": float(np.mean(probs)),
            "actual_positive_rate": float(np.mean(labels)),
            "calibration_gap": abs(float(np.mean(probs)) - float(np.mean(labels))),
        }

        return summary

    @staticmethod
    def format_output(summary: dict) -> str:
        """Format backtest summary for terminal display."""
        lines = [
            "====================================",
            "  BACKTEST RESULTS",
            "====================================",
            "",
            f"Run ID                  {summary.get('run_id', 'N/A')}",
            f"Total Windows           {summary.get('total_windows', 0)}",
            f"Total Trades            {summary.get('total_trades', 0)}",
            f"  BUY YES               {summary.get('yes_trades', 0)}",
            f"  BUY NO                {summary.get('no_trades', 0)}",
            f"Win Rate                {summary.get('win_rate', 0) * 100:.1f}%",
            f"Total PnL (units)       {summary.get('total_pnl', 0):.4f}",
            f"Avg Model Probability   {summary.get('avg_probability', 0) * 100:.1f}%",
            f"Actual Positive Rate    {summary.get('actual_positive_rate', 0) * 100:.1f}%",
            f"Calibration Gap         {summary.get('calibration_gap', 0) * 100:.1f}%",
            "",
            "====================================",
        ]
        return "\n".join(lines)
