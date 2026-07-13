"""
Walk-forward backtesting engine.

Simulates ENTER-gated trades on historical 15-minute windows using the
trained model, net edge after spread/fees, and ask-based contract PnL.
"""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd

import config
from src.db.database import Database
from src.data.aggregator import ExchangeAggregator
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.trading.logic import (
    brti_price_from_features,
    contract_pnl,
    estimate_kalshi_quotes,
    get_entry_signal,
    get_recommendation,
    net_executable_edges,
)


class BacktestEngine:
    """
    Backtest ENTER signals on historical candle data.

    Uses the same entry gates as live prediction (conviction, net edge,
    BRTI alignment, 5–10 min remaining). Persistence is skipped because
    backtest samples are single observations per window.
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

        PnL uses real ask prices (estimated when historical Kalshi data
        is unavailable) and Kalshi fee on winning contracts.
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
            features = dict(sample["features"])
            strike = sample["strike"]
            label = sample["label"]
            minutes_remaining = config.WINDOW_MINUTES - observation_offset

            quotes = estimate_kalshi_quotes(features)
            features["kalshi_yes_mid"] = quotes["kalshi_yes_mid"]
            features["kalshi_yes_ask"] = quotes["kalshi_yes_ask"]
            features["kalshi_no_ask"] = quotes["kalshi_no_ask"]
            features["kalshi_yes_spread"] = quotes["kalshi_yes_spread"]

            X = self.engineer.features_to_array(features).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0)
            prob = float(self.trainer.predict_probability(X)[0])

            yes_ask = quotes["kalshi_yes_ask"]
            no_ask = quotes["kalshi_no_ask"]
            yes_bid = quotes["kalshi_yes_bid"]
            edge_yes, edge_no = net_executable_edges(
                prob, yes_ask, no_ask, yes_bid=yes_bid, no_bid=quotes["kalshi_no_bid"]
            )
            conviction = max(prob, 1 - prob)
            rec = get_recommendation(prob)
            brti_price = brti_price_from_features(features, strike)

            entry_signal, entry_reason = get_entry_signal(
                rec,
                edge_yes,
                edge_no,
                conviction,
                brti_price,
                strike,
                minutes_remaining,
                skip_persistence=True,
            )

            pnl = contract_pnl(entry_signal, yes_ask, no_ask, label)

            results.append({
                "timestamp": sample["observation_time"],
                "strike": strike,
                "minutes_remaining": minutes_remaining,
                "model_probability": prob,
                "kalshi_implied_prob": quotes["kalshi_yes_mid"],
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "net_edge_yes": edge_yes,
                "net_edge_no": edge_no,
                "actual_label": label,
                "recommendation": rec,
                "entry_signal": entry_signal,
                "entry_reason": entry_reason,
                "pnl": pnl,
            })

        if save:
            self.db.save_backtest_results(run_id, results)

        enters = [r for r in results if r["entry_signal"].startswith("ENTER")]
        wins = [r for r in enters if r["pnl"] > 0]
        yes_enters = [r for r in enters if r["entry_signal"] == "ENTER YES"]
        no_enters = [r for r in enters if r["entry_signal"] == "ENTER NO"]
        rec_trades = [r for r in results if r["recommendation"] != "NO TRADE"]

        probs = [r["model_probability"] for r in results]
        labels = [r["actual_label"] for r in results]

        summary = {
            "run_id": run_id,
            "observation_offset": observation_offset,
            "minutes_remaining": config.WINDOW_MINUTES - observation_offset,
            "total_windows": len(results),
            "total_recommendations": len(rec_trades),
            "total_enters": len(enters),
            "yes_enters": len(yes_enters),
            "no_enters": len(no_enters),
            "win_rate": len(wins) / len(enters) if enters else 0.0,
            "total_pnl": sum(r["pnl"] for r in enters),
            "avg_pnl_per_enter": (
                sum(r["pnl"] for r in enters) / len(enters) if enters else 0.0
            ),
            "avg_probability": float(np.mean(probs)),
            "actual_positive_rate": float(np.mean(labels)),
            "calibration_gap": abs(float(np.mean(probs)) - float(np.mean(labels))),
        }

        return summary

    @staticmethod
    def format_output(summary: dict) -> str:
        """Format backtest summary for terminal display."""
        mins = summary.get("minutes_remaining", "?")
        lines = [
            "====================================",
            "  BACKTEST RESULTS (ENTER gates)",
            "====================================",
            "",
            f"Run ID                  {summary.get('run_id', 'N/A')}",
            f"Observation Offset      {summary.get('observation_offset', '?')} min into window",
            f"Minutes Remaining       {mins} min (entry window {config.ENTRY_MIN_MINUTES_REMAINING}–{config.ENTRY_MAX_MINUTES_REMAINING})",
            f"Total Windows           {summary.get('total_windows', 0)}",
            f"Recommendations         {summary.get('total_recommendations', 0)}",
            f"ENTER Trades            {summary.get('total_enters', 0)}",
            f"  ENTER YES             {summary.get('yes_enters', 0)}",
            f"  ENTER NO              {summary.get('no_enters', 0)}",
            f"Win Rate (ENTER only)   {summary.get('win_rate', 0) * 100:.1f}%",
            f"Total PnL (per $1)      {summary.get('total_pnl', 0):.4f}",
            f"Avg PnL per ENTER       {summary.get('avg_pnl_per_enter', 0):.4f}",
            f"Avg Model Probability   {summary.get('avg_probability', 0) * 100:.1f}%",
            f"Actual Positive Rate    {summary.get('actual_positive_rate', 0) * 100:.1f}%",
            f"Calibration Gap         {summary.get('calibration_gap', 0) * 100:.1f}%",
            "",
            "PnL: buy at ask, pay fee on wins, spread/fees in edge gate.",
            "",
            "====================================",
        ]
        return "\n".join(lines)
