"""
Reusable feature engineering for BTC 15-minute Kalshi markets.

Computes technical indicators (EMA, VWAP, ATR, RSI, MACD), volume and
volatility metrics, momentum, distance from strike, time remaining, funding
rate, and open interest. All features are designed to be interpretable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config

# Columns used as model inputs (order matters for persistence)
FEATURE_COLUMNS = [
    "ema_fast",
    "ema_slow",
    "ema_ratio",
    "vwap",
    "price_vs_vwap",
    "atr",
    "atr_pct",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "volume",
    "volume_ratio",
    "volatility",
    "momentum",
    "distance_from_strike",
    "distance_from_strike_pct",
    "minutes_remaining",
    "minutes_elapsed_pct",
    "funding_rate",
    "open_interest",
    "order_imbalance",
]


class FeatureEngineer:
    """
    Stateless feature calculator.

  Given a candle DataFrame and market context (strike, time remaining),
  returns a single feature vector suitable for model inference or training.
    """

    def __init__(self) -> None:
        self.ema_fast = config.EMA_FAST
        self.ema_slow = config.EMA_SLOW
        self.rsi_period = config.RSI_PERIOD
        self.atr_period = config.ATR_PERIOD
        self.vol_window = config.VOLATILITY_WINDOW
        self.mom_window = config.MOMENTUM_WINDOW

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(self.atr_period).mean()

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = self._ema(close, config.MACD_FAST)
        ema_slow = self._ema(close, config.MACD_SLOW)
        macd = ema_fast - ema_slow
        signal = self._ema(macd, config.MACD_SIGNAL)
        hist = macd - signal
        return macd, signal, hist

    # ------------------------------------------------------------------
    # Full indicator DataFrame
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicator columns to a candle DataFrame.

        Expects columns: open, high, low, close, volume.
        """
        out = df.copy()
        close = out["close"]

        out["ema_fast"] = self._ema(close, self.ema_fast)
        out["ema_slow"] = self._ema(close, self.ema_slow)
        out["ema_ratio"] = out["ema_fast"] / out["ema_slow"].replace(0, np.nan)

        out["vwap"] = self._compute_vwap(out)
        out["price_vs_vwap"] = (close - out["vwap"]) / out["vwap"].replace(0, np.nan)

        out["atr"] = self._compute_atr(out)
        out["atr_pct"] = out["atr"] / close.replace(0, np.nan)

        out["rsi"] = self._compute_rsi(close)

        macd, signal, hist = self._compute_macd(close)
        out["macd"] = macd
        out["macd_signal"] = signal
        out["macd_hist"] = hist

        out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean().replace(0, np.nan)

        returns = close.pct_change()
        out["volatility"] = returns.rolling(self.vol_window).std()

        out["momentum"] = close.pct_change(self.mom_window)

        return out

    # ------------------------------------------------------------------
    # Single observation vector
    # ------------------------------------------------------------------

    def build_features(
        self,
        candles: pd.DataFrame,
        strike: float,
        minutes_remaining: float,
        funding_rate: float = 0.0,
        open_interest: float = 0.0,
        order_imbalance: float = 0.0,
    ) -> dict[str, float]:
        """
        Build a feature dict from the latest candle window.

        Uses the last row of computed indicators plus market context.
        """
        if candles.empty:
            raise ValueError("Cannot build features from empty candle data")

        indicators = self.compute_indicators(candles)
        latest = indicators.iloc[-1]
        price = float(latest["close"])

        distance = price - strike
        distance_pct = distance / strike if strike else 0.0
        minutes_elapsed_pct = 1.0 - (minutes_remaining / config.WINDOW_MINUTES)

        features = {
            "ema_fast": float(latest["ema_fast"]),
            "ema_slow": float(latest["ema_slow"]),
            "ema_ratio": float(latest["ema_ratio"]) if pd.notna(latest["ema_ratio"]) else 1.0,
            "vwap": float(latest["vwap"]) if pd.notna(latest["vwap"]) else price,
            "price_vs_vwap": float(latest["price_vs_vwap"]) if pd.notna(latest["price_vs_vwap"]) else 0.0,
            "atr": float(latest["atr"]) if pd.notna(latest["atr"]) else 0.0,
            "atr_pct": float(latest["atr_pct"]) if pd.notna(latest["atr_pct"]) else 0.0,
            "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else 50.0,
            "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else 0.0,
            "macd_signal": float(latest["macd_signal"]) if pd.notna(latest["macd_signal"]) else 0.0,
            "macd_hist": float(latest["macd_hist"]) if pd.notna(latest["macd_hist"]) else 0.0,
            "volume": float(latest["volume"]),
            "volume_ratio": float(latest["volume_ratio"]) if pd.notna(latest["volume_ratio"]) else 1.0,
            "volatility": float(latest["volatility"]) if pd.notna(latest["volatility"]) else 0.0,
            "momentum": float(latest["momentum"]) if pd.notna(latest["momentum"]) else 0.0,
            "distance_from_strike": distance,
            "distance_from_strike_pct": distance_pct,
            "minutes_remaining": minutes_remaining,
            "minutes_elapsed_pct": minutes_elapsed_pct,
            "funding_rate": funding_rate,
            "open_interest": open_interest,
            "order_imbalance": order_imbalance,
        }
        return features

    def features_to_array(self, features: dict[str, float]) -> np.ndarray:
        """Convert feature dict to ordered numpy array for the model."""
        return np.array([features[col] for col in FEATURE_COLUMNS], dtype=np.float64)

    # ------------------------------------------------------------------
    # Training sample generation
    # ------------------------------------------------------------------

    def generate_training_samples(
        self,
        candles: pd.DataFrame,
        window_minutes: int = config.WINDOW_MINUTES,
        observation_offsets: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate labeled training samples from historical candles.

        Kalshi KXBTC15M settlement rule:
          YES if avg(BRTI, 60s before close) >= avg(BRTI, 60s before open)

        Without licensed historical BRTI, we proxy labels using 1-minute candles:
          - reference = close of minute before window open
          - settlement = close of last minute in window
          - label = 1 if settlement >= reference

        Microstructure features still come from Binance (BRTI constituents).
        """
        if observation_offsets is None:
            observation_offsets = [3, 7, 12]

        candles = candles.sort_values("timestamp").reset_index(drop=True)
        candles["minute"] = candles["timestamp"].dt.minute
        candles["hour"] = candles["timestamp"].dt.hour

        samples: list[dict[str, Any]] = []
        n = len(candles)

        # Find window starts at :00, :15, :30, :45
        window_starts = candles[
            candles["minute"] % window_minutes == 0
        ]["timestamp"].tolist()

        for ws in window_starts:
            we = ws + pd.Timedelta(minutes=window_minutes)
            window = candles[
                (candles["timestamp"] >= ws) & (candles["timestamp"] < we)
            ]
            if len(window) < window_minutes:
                continue

            # Opening reference: minute before window open (proxy for pre-open 60s BRTI avg)
            pre_open = candles[candles["timestamp"] == ws - pd.Timedelta(minutes=1)]
            if not pre_open.empty:
                reference = float(pre_open.iloc[0]["close"])
            else:
                reference = float(window.iloc[0]["open"])

            # Settlement: last minute close (proxy for pre-close 60s BRTI avg)
            settlement = float(window.iloc[-1]["close"])
            label = 1 if settlement >= reference else 0

            for offset in observation_offsets:
                obs_time = ws + pd.Timedelta(minutes=offset)
                history = candles[candles["timestamp"] <= obs_time]
                if len(history) < 30:
                    continue

                minutes_remaining = window_minutes - offset
                try:
                    features = self.build_features(
                        history.tail(120),
                        strike=reference,
                        minutes_remaining=minutes_remaining,
                    )
                except (ValueError, KeyError):
                    continue

                samples.append({
                    "window_start": ws.to_pydatetime() if hasattr(ws, "to_pydatetime") else ws,
                    "observation_time": obs_time.to_pydatetime() if hasattr(obs_time, "to_pydatetime") else obs_time,
                    "strike": reference,
                    "label": label,
                    "features": features,
                })

        return samples

    # ------------------------------------------------------------------
    # Human-readable factor explanations
    # ------------------------------------------------------------------

    FACTOR_DESCRIPTIONS: dict[str, tuple[str, str, callable]] = {
        "momentum": (
            "Positive momentum",
            "Negative momentum",
            lambda v: v > 0,
        ),
        "price_vs_vwap": (
            "Above VWAP",
            "Below VWAP",
            lambda v: v > 0,
        ),
        "order_imbalance": (
            "Order imbalance bullish",
            "Order imbalance bearish",
            lambda v: v > 0,
        ),
        "volatility": (
            "High volatility",
            "Low volatility",
            lambda v: v > 0.001,
        ),
        "rsi": (
            "Bullish RSI",
            "Bearish RSI",
            lambda v: v > 55,
        ),
        "ema_ratio": (
            "EMA bullish crossover",
            "EMA bearish crossover",
            lambda v: v > 1.0,
        ),
        "distance_from_strike_pct": (
            "Above reference",
            "Below reference",
            lambda v: v > 0,
        ),
        "macd_hist": (
            "MACD bullish",
            "MACD bearish",
            lambda v: v > 0,
        ),
        "funding_rate": (
            "Positive funding (longs pay)",
            "Negative funding (shorts pay)",
            lambda v: v > 0,
        ),
        "volume_ratio": (
            "Elevated volume",
            "Low volume",
            lambda v: v > 1.2,
        ),
    }

    def explain_factors(
        self,
        features: dict[str, float],
        importances: dict[str, float],
        top_n: int = 5,
    ) -> list[str]:
        """
        Return human-readable explanations for the top contributing features.

        Ranks features by XGBoost importance weighted by directional signal.
        """
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        explanations: list[str] = []

        for feat, _imp in ranked:
            if feat not in self.FACTOR_DESCRIPTIONS:
                continue
            bullish, bearish, condition = self.FACTOR_DESCRIPTIONS[feat]
            value = features.get(feat, 0)
            desc = bullish if condition(value) else bearish
            if desc not in explanations:
                explanations.append(desc)
            if len(explanations) >= top_n:
                break

        # Fallback if importances don't map cleanly
        if len(explanations) < top_n:
            for feat, (bullish, bearish, condition) in self.FACTOR_DESCRIPTIONS.items():
                if feat in features:
                    desc = bullish if condition(features[feat]) else bearish
                    if desc not in explanations:
                        explanations.append(desc)
                if len(explanations) >= top_n:
                    break

        return explanations[:top_n]
