#!/usr/bin/env python3
"""
Backtesting entry point.

Evaluates the trained model on historical 15-minute windows and reports
win rate, PnL, and calibration metrics.

Usage:
    python backtest.py
    python backtest.py --offset 5
"""

from __future__ import annotations

import argparse
import sys

import config
from src.backtest.engine import BacktestEngine
from src.models.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the prediction model")
    parser.add_argument(
        "--offset", type=int, default=7,
        help="Minutes into window to make prediction (default: 7)",
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Days of data to backtest (default: 30)",
    )
    args = parser.parse_args()

    if not config.MODEL_PATH.exists():
        print("No trained model found. Run: python train.py")
        sys.exit(1)

    # Ensure we have candle data
    trainer = ModelTrainer()
    try:
        print(f"Loading {args.days} days of candle data...")
        trainer.build_training_data_from_candles(days=args.days, save=False)
    finally:
        trainer.db.close()

    engine = BacktestEngine()
    try:
        print(f"Running backtest (observation offset = {args.offset} min)...")
        summary = engine.run(observation_offset=args.offset)
        print(BacktestEngine.format_output(summary))
    finally:
        engine.db.close()


if __name__ == "__main__":
    main()
