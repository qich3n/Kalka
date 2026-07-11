#!/usr/bin/env python3
"""
Live prediction entry point.

Fetches current market data, runs the trained XGBoost model, and prints
an explainable recommendation with probability, edge, and top factors.

Usage:
    python predict.py
    python predict.py --strike 97500 --minutes 8
"""

from __future__ import annotations

import argparse
import sys

from src.models.predictor import Predictor
from src.models.trainer import ModelTrainer
import config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Kalshi 15-minute BTC YES/NO probability"
    )
    parser.add_argument(
        "--strike", type=float, default=None,
        help="Override strike price (default: from Kalshi)",
    )
    parser.add_argument(
        "--minutes", type=float, default=None,
        help="Override minutes remaining (default: from Kalshi)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train model first if none exists",
    )
    args = parser.parse_args()

    # Auto-train if no model exists
    if not config.MODEL_PATH.exists():
        if args.train:
            print("No model found. Training...")
            trainer = ModelTrainer()
            trainer.build_training_data_from_candles(days=14)
            trainer.train()
        else:
            print(
                "No trained model found. Run:\n"
                "  python train.py\n"
                "or:\n"
                "  python predict.py --train"
            )
            sys.exit(1)

    predictor = Predictor()
    try:
        result = predictor.predict(
            strike=args.strike,
            minutes_remaining=args.minutes,
        )
        print(Predictor.format_output(result))
    finally:
        predictor.close()


if __name__ == "__main__":
    main()
