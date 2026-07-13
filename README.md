# Kalka

A quantitative research assistant that predicts the probability of Bitcoin finishing **up** in Kalshi 15-minute YES/NO markets.

## Settlement: CF Benchmarks BRTI

Kalshi **KXBTC15M** does **not** settle on Binance spot. It uses the [CF Benchmarks Bitcoin Real-Time Index (BRTI)](https://www.cfbenchmarks.com/data/indices/BRTI):

- **YES** if the 60-second BRTI average before window close ≥ the 60-second BRTI average before window open
- The Kalshi `floor_strike` / "Target Price" is the **opening reference** (pre-open 60s BRTI avg)

Kalka uses **BRTI for index price and reference distance**. Microstructure features come from **Binance** candles; **Coinbase** and **Kraken** provide cross-exchange basis and composite BRTI proxy labels.

## Data Sources

| Source | Role |
|--------|------|
| **CF Benchmarks BRTI** | Settlement index (Kalshi resolution) |
| **Kalshi** | Market context, reference price, implied probability |
| **Binance** | Primary microstructure candles, funding, open interest |
| **Coinbase** | Cross-exchange basis, composite BRTI proxy |
| **Kraken** | Cross-exchange basis, composite BRTI proxy |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py          # fetch data + train model (~2 min)
python predict.py        # live prediction
python backtest.py       # evaluate on historical windows
```

On macOS you may need `brew install libomp` for XGBoost.

## Optional: Authenticated BRTI Feed

For the best live BRTI data, set Kalshi API credentials (enables the CF Benchmarks passthrough):

```bash
export KALSHI_API_KEY_ID="your-key-id"
export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi-private-key.pem"
```

Without keys, Kalka falls back to the public CF Benchmarks index page (typically 5–30+ seconds delayed). Output shows quote age; values marked `[STALE]` are too old for reliable edge. For live 1-second BRTI, set Kalshi API credentials:

## What It Does

Kalka answers one question: **"Should I buy YES or NO right now?"**

It does not place trades. Every prediction includes:

- Model probability P(end BRTI avg ≥ opening reference)
- Kalshi implied probability
- Estimated edge
- Confidence level
- BRTI price vs Binance basis
- Top 5 explanatory factors
- Trading recommendation

## Architecture

```
predict.py / train.py / backtest.py   ← entry points
config.py                             ← shared configuration
src/
  data/       BRTI + Binance + Coinbase + Kraken + Kalshi clients
  db/         DuckDB persistence
  features/   EMA, VWAP, ATR, RSI, MACD, momentum, etc.
  models/     XGBoost training + live prediction
  backtest/   Walk-forward backtesting
data/
  kalka.duckdb                        ← local database
  models/xgboost_model.json           ← trained model
```

## Trading Logic

| Condition | Recommendation |
|-----------|---------------|
| Probability > 55% | BUY YES |
| Probability < 45% | BUY NO |
| Otherwise | NO TRADE |

If confidence (max of prob, 1-prob) is below 55%, the prediction label shows **NO TRADE** regardless of direction.

**Entry Signal** is stricter than recommendation — all must pass:

| Gate | Requirement |
|------|-------------|
| Time window | 5–10 minutes remaining in the window |
| Conviction | ≥ 70% (`ENTRY_CONVICTION_THRESHOLD`) |
| Net edge | ≥ 5% after spread buffer + Kalshi fee on expected profit |
| BRTI alignment | YES only if BRTI ≥ reference; NO only if BRTI ≤ reference |
| Persistence | Same signal on 2 consecutive runs within 3 minutes (live only) |

Tune thresholds in `config.py`. **Net edge** subtracts half-spread slippage and fee-adjusted profit from gross edge vs Kalshi **ask** prices.

Training includes **Kalshi market features** (`kalshi_yes_mid`, `kalshi_yes_ask`, `kalshi_no_ask`, `kalshi_yes_spread`). Live runs use real quotes; historical training/backtest estimate quotes from settlement proxy distance.

## Manual Overrides

```bash
python predict.py --strike 97500 --minutes 8
```

`--strike` sets the opening BRTI reference price. Useful when Kalshi markets are closed or for testing.

## Retraining

```bash
python train.py --days 60
```

Training labels prefer **stored BRTI ticks** (true 60-second settlement windows). When enough BRTI-labeled samples exist, training automatically uses them only. Force with `python train.py --brti-only`, or include candle proxies with `--all-labels`.

The model includes **BRTI-native features** (60s average vs reference, momentum, volatility, composite basis, settlement proxy) and **Kalshi quote features**. **Retrain after feature updates:** `python train.py --days 60`.

## Backtesting

```bash
python backtest.py --offset 7 --days 30
```

Backtest uses the same **ENTER gates** as live (conviction, net edge, BRTI alignment, 5–10 min window). PnL buys at estimated ask prices and applies Kalshi fees on wins. Default `--offset 7` observes 8 minutes before expiry (sweet spot). Persistence is skipped (single sample per window).

## Disclaimer

This is a research tool, not financial advice. Training uses Binance proxies for historical BRTI settlement when licensed tick data is unavailable. Past backtest performance does not guarantee future results.
