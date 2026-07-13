"""
Shared trading logic: edge (after costs), entry gates, and contract PnL.

Used by live prediction and backtesting so both evaluate the same ENTER rules.
"""

from __future__ import annotations

import pandas as pd

import config


def gross_executable_edges(
    probability: float,
    yes_ask: float,
    no_ask: float,
) -> tuple[float, float]:
    """Raw edge vs ask prices (before spread buffer / fees)."""
    edge_yes = probability - yes_ask
    edge_no = (1.0 - probability) - no_ask
    return edge_yes, edge_no


def net_executable_edges(
    probability: float,
    yes_ask: float,
    no_ask: float,
    yes_bid: float | None = None,
    no_bid: float | None = None,
) -> tuple[float, float]:
    """
    Conservative edge after half-spread buffer and expected fee on winnings.

    Crossing the ask already pays the spread; we subtract half-spread again as
    slippage buffer plus fee-adjusted edge for threshold checks.
    """
    yes_spread = (
        max(yes_ask - yes_bid, 0.0)
        if yes_bid is not None
        else config.ASSUMED_KALSHI_SPREAD
    )
    no_spread = (
        max(no_ask - no_bid, 0.0)
        if no_bid is not None
        else config.ASSUMED_KALSHI_SPREAD
    )
    fee = config.KALSHI_FEE_ON_PROFITS

    edge_yes, edge_no = gross_executable_edges(probability, yes_ask, no_ask)
    edge_yes -= yes_spread / 2 + fee * max(1.0 - yes_ask, 0.0)
    edge_no -= no_spread / 2 + fee * max(1.0 - no_ask, 0.0)
    return edge_yes, edge_no


def count_persisted_signals(
    recent: pd.DataFrame,
    recommendation: str,
    min_conviction: float,
) -> int:
    """Count recent predictions matching direction with sufficient conviction."""
    if recent.empty:
        return 0
    count = 0
    for _, row in recent.iterrows():
        prob = float(row["model_probability"])
        conviction = max(prob, 1 - prob)
        if row["recommendation"] == recommendation and conviction >= min_conviction:
            count += 1
        else:
            break
    return count


def get_recommendation(probability: float) -> str:
    if probability > config.YES_THRESHOLD:
        return "BUY YES"
    if probability < config.NO_THRESHOLD:
        return "BUY NO"
    return "NO TRADE"


def get_entry_signal(
    recommendation: str,
    edge_yes: float,
    edge_no: float,
    conviction: float,
    brti_price: float,
    reference_price: float,
    minutes_remaining: float,
    persisted_samples: int = 0,
    *,
    skip_persistence: bool = False,
) -> tuple[str, str]:
    """
    Decide whether to enter based on conviction, net edge, BRTI alignment,
    entry time window, and optional signal persistence.
    """
    min_edge = config.MIN_EDGE_THRESHOLD
    min_edge_pct = min_edge * 100
    min_conviction = config.ENTRY_CONVICTION_THRESHOLD
    conviction_pct = conviction * 100
    min_conviction_pct = min_conviction * 100
    brti_vs_ref = brti_price - reference_price

    if minutes_remaining < config.ENTRY_MIN_MINUTES_REMAINING:
        return (
            "SKIP",
            f"Too close to expiry ({minutes_remaining:.1f} min left, min {config.ENTRY_MIN_MINUTES_REMAINING})",
        )
    if minutes_remaining > config.ENTRY_MAX_MINUTES_REMAINING:
        return (
            "SKIP",
            f"Too early in window ({minutes_remaining:.1f} min left, max {config.ENTRY_MAX_MINUTES_REMAINING})",
        )

    if recommendation == "NO TRADE":
        return "SKIP", f"Conviction {conviction_pct:.1f}% below recommendation threshold"

    if conviction < min_conviction:
        return (
            "SKIP",
            f"Conviction {conviction_pct:.1f}% below {min_conviction_pct:.0f}% entry minimum",
        )

    if recommendation == "BUY YES":
        if config.REQUIRE_BRTI_ALIGNMENT and brti_price < reference_price:
            return (
                "SKIP",
                f"BRTI ${brti_vs_ref:+,.2f} below reference — wait for YES alignment",
            )
        if edge_yes < min_edge:
            return (
                "SKIP",
                f"Conviction {conviction_pct:.1f}%, net edge too low ({edge_yes * 100:+.1f}% vs {min_edge_pct:.0f}% minimum)",
            )
        if not skip_persistence and persisted_samples + 1 < config.ENTRY_PERSISTENCE_SAMPLES:
            need = config.ENTRY_PERSISTENCE_SAMPLES - persisted_samples - 1
            return (
                "SKIP",
                f"Signal not stable yet — run {need} more time(s) with consistent BUY YES",
            )
        return (
            "ENTER YES",
            f"Conviction {conviction_pct:.1f}%, net edge {edge_yes * 100:+.1f}%, BRTI aligned, in entry window",
        )

    if recommendation == "BUY NO":
        if config.REQUIRE_BRTI_ALIGNMENT and brti_price > reference_price:
            return (
                "SKIP",
                f"BRTI ${brti_vs_ref:+,.2f} above reference — wait for NO alignment",
            )
        if edge_no < min_edge:
            return (
                "SKIP",
                f"Conviction {conviction_pct:.1f}%, net edge too low ({edge_no * 100:+.1f}% vs {min_edge_pct:.0f}% minimum)",
            )
        if not skip_persistence and persisted_samples + 1 < config.ENTRY_PERSISTENCE_SAMPLES:
            need = config.ENTRY_PERSISTENCE_SAMPLES - persisted_samples - 1
            return (
                "SKIP",
                f"Signal not stable yet — run {need} more time(s) with consistent BUY NO",
            )
        return (
            "ENTER NO",
            f"Conviction {conviction_pct:.1f}%, net edge {edge_no * 100:+.1f}%, BRTI aligned, in entry window",
        )

    return "SKIP", "No actionable signal"


def contract_pnl(
    entry_signal: str,
    yes_ask: float,
    no_ask: float,
    actual_label: int,
) -> float:
    """
    Realized PnL per $1 contract after entry at ask, including fee on profits.

    actual_label: 1 if YES won (settlement >= reference), else 0.
    """
    fee = config.KALSHI_FEE_ON_PROFITS

    if entry_signal == "ENTER YES":
        if actual_label == 1:
            return (1.0 - yes_ask) * (1.0 - fee)
        return -yes_ask

    if entry_signal == "ENTER NO":
        if actual_label == 0:
            return (1.0 - no_ask) * (1.0 - fee)
        return -no_ask

    return 0.0


def estimate_kalshi_quotes(features: dict[str, float]) -> dict[str, float]:
    """
    Estimate Kalshi quotes for historical backtest when tick data is unavailable.

    Uses settlement proxy distance to reference as a fair-value estimate.
    """
    proxy_pct = features.get("settlement_proxy_vs_reference_pct", 0.0)
    dist_pct = features.get("distance_from_strike_pct", 0.0)
    signal = proxy_pct if proxy_pct != 0.0 else dist_pct

    yes_mid = max(0.02, min(0.98, 0.5 + signal * 15.0))
    half_spread = config.ASSUMED_KALSHI_SPREAD / 2
    yes_ask = min(0.99, yes_mid + half_spread)
    no_mid = 1.0 - yes_mid
    no_ask = min(0.99, no_mid + half_spread)
    yes_bid = max(0.01, yes_mid - half_spread)
    no_bid = max(0.01, no_mid - half_spread)

    return {
        "kalshi_yes_mid": yes_mid,
        "kalshi_yes_ask": yes_ask,
        "kalshi_no_ask": no_ask,
        "kalshi_yes_bid": yes_bid,
        "kalshi_no_bid": no_bid,
        "kalshi_yes_spread": yes_ask - yes_bid,
    }


def brti_price_from_features(features: dict[str, float], reference: float) -> float:
    """Infer BRTI spot from strike distance feature."""
    return reference + features.get("distance_from_strike", 0.0)
