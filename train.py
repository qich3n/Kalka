#!/usr/bin/env python3
"""
Model training entry point.

Fetches historical Binance data, generates labeled training samples,
trains an XGBoost classifier, and reports validation metrics.

Usage:
    python train.py
    python train.py --days 60
"""

from __future__ import annotations

import argparse

from src.models.trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XGBoost prediction model")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Days of historical data to use (default: 30)",
    )
    args = parser.parse_args()

    trainer = ModelTrainer()
    try:
        print(f"Building training data from {args.days} days of candles...")
        df = trainer.build_training_data_from_candles(days=args.days)

        print(f"\nTraining XGBoost on {len(df)} samples...")
        metrics = trainer.train(df)

        print("\nTop feature importances:")
        sorted_imp = sorted(
            trainer.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for feat, imp in sorted_imp[:10]:
            print(f"  {feat:30s} {imp:.4f}")

        print("\nModel saved to:", config.MODEL_PATH)
    finally:
        trainer.db.close()


if __name__ == "__main__":
    import config  # noqa: E402 — used in print above
    main()
