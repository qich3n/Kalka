# Kalka

A quantitative research assistant that predicts the probability of Bitcoin finishing **up** in Kalshi 15-minute YES/NO markets.

## Settlement: CF Benchmarks BRTI

Kalshi **KXBTC15M** does **not** settle on Binance spot. It uses the [CF Benchmarks Bitcoin Real-Time Index (BRTI)](https://www.cfbenchmarks.com/data/indices/BRTI):

- **YES** if the 60-second BRTI average before window close ≥ the 60-second BRTI average before window open
- The Kalshi `floor_strike` / "Target Price" is the **opening reference** (pre-open 60s BRTI avg)

Kalka uses **BRTI for index price and reference distance**. Binance data is used only for microstructure features (EMA, volume, order flow, etc.).

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

Without keys, Kalka falls back to the public CF Benchmarks index page.

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
  data/       BRTI + Binance + Kalshi API clients
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

## Manual Overrides

```bash
python predict.py --strike 97500 --minutes 8
```

`--strike` sets the opening BRTI reference price. Useful when Kalshi markets are closed or for testing.

## Retraining

```bash
python train.py --days 60
```

Training labels proxy BRTI settlement using 1-minute candle data. Retrain after collecting local BRTI ticks for better calibration.

## Backtesting

```bash
python backtest.py --offset 7 --days 30
```

## Disclaimer

This is a research tool, not financial advice. Training uses Binance proxies for historical BRTI settlement when licensed tick data is unavailable. Past backtest performance does not guarantee future results.
