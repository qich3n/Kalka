# Kalka

A quantitative research assistant that predicts the probability of Bitcoin finishing above strike in Kalshi 15-minute YES/NO markets.

## Quick Start

```bash
pip install -r requirements.txt
python train.py          # fetch data + train model (~2 min)
python predict.py        # live prediction
python backtest.py       # evaluate on historical windows
```

## What It Does

Kalka answers one question: **"Should I buy YES or NO right now?"**

It does not place trades. Every prediction includes:

- Model probability (P(BTC finishes above strike))
- Kalshi implied probability
- Estimated edge
- Confidence level
- Top 5 explanatory factors
- Trading recommendation

## Architecture

```
predict.py / train.py / backtest.py   ← entry points
config.py                             ← shared configuration
src/
  data/       Binance + Kalshi API clients
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

## Features

- EMA (9/21), VWAP, ATR, RSI, MACD
- Volume ratio, volatility, momentum
- Distance from strike (absolute and %)
- Time remaining / elapsed
- Funding rate, open interest, order book imbalance

## Manual Overrides

```bash
python predict.py --strike 97500 --minutes 8
```

Useful when Kalshi markets are closed or for testing.

## Retraining

```bash
python train.py --days 60
```

## Backtesting

```bash
python backtest.py --offset 7 --days 30
```

## Disclaimer

This is a research tool, not financial advice. Past performance in backtests does not guarantee future results. Always verify predictions against your own analysis before trading.
