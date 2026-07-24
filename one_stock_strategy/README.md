# One-Stock Multi-Timeframe RSI/MACD/Volume Backtester

This system downloads **one-minute data for one stock**, calculates the base strategy, and builds completed **2-minute, 5-minute, and 15-minute candles** internally.

## Bull signal logic

The base setup requires:

1. One-minute MACD crosses above its signal line.
2. One-minute RSI is at or below `--rsi-buy`.
3. One-minute relative volume is at least `--volume-spike`.

Before creating the final buy signal, the 2m, 5m, and 15m confirmations check:

- RSI is above its timeframe threshold and rising.
- Relative volume is above its timeframe threshold and rising.
- MACD histogram is positive (or above the supplied threshold) and rising.

By default, all three timeframes must confirm. Use `--required-confirmations 2` to require any two.

The 30-session volume comparison is made against the **same clock-time candle** on prior trading sessions. The current candle is excluded from its own average.

## Install

```bash
python -m pip install -r requirements.txt
export MASSIVE_API_KEY="YOUR_API_KEY"
```

## Example

```bash
python one_stock_rsi_macd_volume_backtest.py \
  --ticker AAPL \
  --start 2026-05-01 \
  --end 2026-06-30 \
  --timespan minute \
  --multiplier 1 \
  --rsi-buy 35 \
  --volume-spike 1.5 \
  --rsi-2m-min 45 \
  --rsi-5m-min 48 \
  --rsi-15m-min 50 \
  --volume-2m-min 1.20 \
  --volume-5m-min 1.15 \
  --volume-15m-min 1.10 \
  --required-confirmations 3
```

Signals are generated from completed candles and executed at the next one-minute candle open. This is a research/backtesting example and does not place brokerage orders.

## Optimize 1,000+ configurations for one company

The optimizer downloads the stock once, caches the bars, precomputes all
indicators, and random-tests thresholds and stop/take-profit settings. It uses
the earlier portion as training data and the latest 30% as validation data.

```bash
python optimize_one_stock_strategy.py \
  --ticker AAPL \
  --start 2025-01-01 \
  --end 2026-06-30 \
  --trials 1000 \
  --validation-fraction 0.30 \
  --minimum-trades 10
```

Outputs:

- Every tested configuration and metrics
- Top 25 configurations
- Best configuration as JSON
- Reusable raw stock-data cache, avoiding repeated API downloads

For more reliable results, use many months of data and test the winning
configuration on a completely separate date range that was not used by the
optimizer.
