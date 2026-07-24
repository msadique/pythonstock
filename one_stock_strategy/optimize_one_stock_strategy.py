#!/usr/bin/env python3
"""Fast random-search optimizer for a one-stock 5-minute strategy.

Key performance choices:
- Downloads native 5-minute bars instead of one-minute bars.
- Precomputes indicators only once.
- Converts indicator columns to compact NumPy arrays once.
- Uses a Numba-compiled backtest loop when Numba is installed.
- Falls back to the same array loop in normal Python if Numba is unavailable.
- Uses chronological train/validation splits to reduce overfitting.

Entry signal (evaluated on a completed 5-minute candle and executed at the
next candle open):
- 5-minute MACD bullish crossover
- 5-minute RSI within a configurable range
- 5-minute same-clock-time relative volume above a configurable threshold
- Optional rising RSI/MACD histogram/relative volume
- Optional completed 15-minute bullish confirmation

Exit signal:
- 5-minute MACD bearish crossover, RSI exit threshold, stop loss, or target.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from one_stock_rsi_macd_volume_backtest import (
    add_macd,
    add_rsi,
    add_same_time_volume_baseline,
    download_massive_bars,
    filter_regular_session,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback remains functional
    njit = None


@dataclass(frozen=True)
class Candidate:
    rsi_buy_min: float
    rsi_buy_max: float
    rsi_sell: float
    relative_volume_min: float
    macd_hist_min: float
    require_rsi_rising: bool
    require_volume_rising: bool
    require_macd_rising: bool
    use_15m_confirmation: bool
    rsi_15m_min: float
    relative_volume_15m_min: float
    macd_hist_15m_min: float
    required_15m_rising: bool
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class EngineSettings:
    initial_cash: float
    commission: float
    slippage_bps: float
    allocation_pct: float = 1.0


def sample_candidate(rng: random.Random) -> Candidate:
    """Generate practical 5-minute configurations with enough signal variety."""
    rsi_low = rng.uniform(25.0, 50.0)
    rsi_high = rng.uniform(max(rsi_low + 3.0, 40.0), 68.0)
    return Candidate(
        rsi_buy_min=rsi_low,
        rsi_buy_max=rsi_high,
        rsi_sell=rng.uniform(max(rsi_high + 3.0, 58.0), 82.0),
        relative_volume_min=rng.uniform(0.85, 2.00),
        macd_hist_min=rng.choice([0.0, 0.0, 0.0, -0.01]),
        require_rsi_rising=rng.choice([True, False, False]),
        require_volume_rising=rng.choice([True, False, False]),
        require_macd_rising=rng.choice([True, True, False]),
        use_15m_confirmation=rng.choice([True, False, False]),
        rsi_15m_min=rng.uniform(40.0, 62.0),
        relative_volume_15m_min=rng.uniform(0.75, 1.50),
        macd_hist_15m_min=rng.choice([0.0, 0.0, -0.01]),
        required_15m_rising=rng.choice([True, False, False]),
        stop_loss_pct=rng.choice([0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025]),
        take_profit_pct=rng.choice([0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
    )


def _completed_15m_indicators(df_5m: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Attach completed 15-minute indicators to each 5-minute row without lookahead."""
    local = df_5m.tz_convert(timezone)
    bars_15m = (
        local.resample("15min", label="right", closed="right", origin="start_day", offset="30min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    bars_15m = add_rsi(bars_15m, 14)
    bars_15m = add_macd(bars_15m, 12, 26, 9)
    bars_15m = add_same_time_volume_baseline(bars_15m, 30, timezone)
    bars_15m["rsi_rising_15m"] = bars_15m["rsi"].diff() > 0
    bars_15m["volume_rising_15m"] = bars_15m["relative_volume"].diff() > 0
    bars_15m["macd_rising_15m"] = bars_15m["macd_histogram"].diff() > 0

    selected = bars_15m[[
        "rsi", "relative_volume", "macd_histogram",
        "rsi_rising_15m", "volume_rising_15m", "macd_rising_15m",
    ]].rename(columns={
        "rsi": "rsi_15m",
        "relative_volume": "relative_volume_15m",
        "macd_histogram": "macd_histogram_15m",
    })

    # A 15-minute bar timestamp is its completion time. Forward fill only from
    # completed bars, so a 5-minute row never sees an unfinished 15-minute bar.
    joined = local.join(selected, how="left").ffill()
    return joined.tz_convert("UTC")


def build_indicator_cache(raw: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = filter_regular_session(raw, timezone)
    df = add_rsi(df, 14)
    df = add_macd(df, 12, 26, 9)
    df = add_same_time_volume_baseline(df, 30, timezone)
    df["rsi_rising"] = df["rsi"].diff() > 0
    df["volume_rising"] = df["relative_volume"].diff() > 0
    df["macd_rising"] = df["macd_histogram"].diff() > 0
    df["macd_cross_up"] = (
        (df["macd"] > df["macd_signal"])
        & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    )
    df["macd_cross_down"] = (
        (df["macd"] < df["macd_signal"])
        & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    )
    return _completed_15m_indicators(df, timezone)


def split_indices(df: pd.DataFrame, validation_fraction: float, timezone: str) -> tuple[int, str]:
    dates = np.asarray(df.index.tz_convert(timezone).date)
    sessions = np.unique(dates)
    if len(sessions) < 10:
        raise ValueError("Use at least 10 trading sessions for optimization.")
    split_session_index = max(1, min(len(sessions) - 1, int(len(sessions) * (1 - validation_fraction))))
    validation_start = sessions[split_session_index]
    split_row = int(np.searchsorted(dates, validation_start, side="left"))
    return split_row, str(validation_start)


def to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    # Normalize explicitly to nanoseconds. Recent Pandas versions can preserve
    # a microsecond-resolution DatetimeIndex, so index.view("int64") is not
    # guaranteed to be nanoseconds. The backtest duration math divides by
    # 60_000_000_000, which requires true Unix nanoseconds.
    timestamps_ns = (
        df.index.tz_convert("UTC")
        .tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    return (
        timestamps_ns,
        df["open"].to_numpy(np.float64),
        df["high"].to_numpy(np.float64),
        df["low"].to_numpy(np.float64),
        df["close"].to_numpy(np.float64),
        df["rsi"].to_numpy(np.float64),
        df["relative_volume"].to_numpy(np.float64),
        df["macd_histogram"].to_numpy(np.float64),
        df["macd_cross_up"].to_numpy(np.bool_),
        df["macd_cross_down"].to_numpy(np.bool_),
        df["rsi_rising"].to_numpy(np.bool_),
        df["volume_rising"].to_numpy(np.bool_),
        df["macd_rising"].to_numpy(np.bool_),
        df["rsi_15m"].to_numpy(np.float64),
        df["relative_volume_15m"].to_numpy(np.float64),
        df["macd_histogram_15m"].to_numpy(np.float64),
        df["rsi_rising_15m"].to_numpy(np.bool_),
        df["volume_rising_15m"].to_numpy(np.bool_),
        df["macd_rising_15m"].to_numpy(np.bool_),
    )


def _fast_backtest_impl(
    timestamps_ns: np.ndarray,
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    rsi: np.ndarray, rel_volume: np.ndarray, macd_hist: np.ndarray,
    cross_up: np.ndarray, cross_down: np.ndarray,
    rsi_rising: np.ndarray, volume_rising: np.ndarray, macd_rising: np.ndarray,
    rsi_15m: np.ndarray, rel_volume_15m: np.ndarray, macd_hist_15m: np.ndarray,
    rsi_rising_15m: np.ndarray, volume_rising_15m: np.ndarray, macd_rising_15m: np.ndarray,
    start: int, end: int,
    rsi_buy_min: float, rsi_buy_max: float, rsi_sell: float,
    rel_volume_min: float, macd_hist_min: float,
    require_rsi_rising: bool, require_volume_rising: bool, require_macd_rising: bool,
    use_15m: bool, rsi_15m_min: float, rel_volume_15m_min: float,
    macd_hist_15m_min: float, required_15m_rising: bool,
    stop_loss_pct: float, take_profit_pct: float,
    initial_cash: float, commission: float, slippage_bps: float, allocation_pct: float,
) -> tuple[float, float, int, float, float, float, float, float, float, float]:
    cash = initial_cash
    quantity = 0
    entry_price = 0.0
    entry_total_cost = 0.0
    peak_equity = initial_cash
    max_drawdown = 0.0
    trades = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    return_sum = 0.0
    entry_index = -1
    total_holding_minutes = 0.0
    min_holding_minutes = 1.0e30
    max_holding_minutes = 0.0
    total_holding_bars = 0
    slip = slippage_bps / 10000.0

    for i in range(max(start + 1, 1), end):
        p = i - 1

        # Exit from previous completed candle's indicator signal.
        if quantity > 0 and (cross_down[p] or rsi[p] >= rsi_sell):
            exit_price = open_[i] * (1.0 - slip)
            proceeds = quantity * exit_price - commission
            pnl = proceeds - entry_total_cost
            cash += proceeds
            trades += 1
            if pnl > 0:
                wins += 1
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += -pnl
            return_sum += (exit_price / entry_price - 1.0) * 100.0
            if entry_index >= 0:
                holding_minutes = (timestamps_ns[i] - timestamps_ns[entry_index]) / 60_000_000_000.0
                holding_bars = i - entry_index
                total_holding_minutes += holding_minutes
                total_holding_bars += holding_bars
                if holding_minutes < min_holding_minutes:
                    min_holding_minutes = holding_minutes
                if holding_minutes > max_holding_minutes:
                    max_holding_minutes = holding_minutes
            quantity = 0
            entry_index = -1

        # Enter from previous completed candle's signal.
        if quantity == 0:
            signal = (
                cross_up[p]
                and rsi[p] >= rsi_buy_min
                and rsi[p] <= rsi_buy_max
                and rel_volume[p] >= rel_volume_min
                and macd_hist[p] >= macd_hist_min
            )
            if require_rsi_rising:
                signal = signal and rsi_rising[p]
            if require_volume_rising:
                signal = signal and volume_rising[p]
            if require_macd_rising:
                signal = signal and macd_rising[p]
            if use_15m:
                signal = (
                    signal
                    and rsi_15m[p] >= rsi_15m_min
                    and rel_volume_15m[p] >= rel_volume_15m_min
                    and macd_hist_15m[p] >= macd_hist_15m_min
                )
                if required_15m_rising:
                    signal = signal and rsi_rising_15m[p] and macd_rising_15m[p]

            if signal:
                entry_price = open_[i] * (1.0 + slip)
                usable = max(cash * allocation_pct - commission, 0.0)
                quantity = int(usable // entry_price)
                if quantity > 0:
                    entry_total_cost = quantity * entry_price + commission
                    cash -= entry_total_cost
                    entry_index = i

        # Intrabar risk exits. Stop first if both are touched.
        if quantity > 0:
            stop_price = entry_price * (1.0 - stop_loss_pct)
            target_price = entry_price * (1.0 + take_profit_pct)
            raw_exit = 0.0
            if low[i] <= stop_price:
                raw_exit = stop_price
            elif high[i] >= target_price:
                raw_exit = target_price

            if raw_exit > 0.0:
                exit_price = raw_exit * (1.0 - slip)
                proceeds = quantity * exit_price - commission
                pnl = proceeds - entry_total_cost
                cash += proceeds
                trades += 1
                if pnl > 0:
                    wins += 1
                    gross_profit += pnl
                elif pnl < 0:
                    gross_loss += -pnl
                return_sum += (exit_price / entry_price - 1.0) * 100.0
                if entry_index >= 0:
                    holding_minutes = (timestamps_ns[i] - timestamps_ns[entry_index]) / 60_000_000_000.0
                    holding_bars = i - entry_index
                    total_holding_minutes += holding_minutes
                    total_holding_bars += holding_bars
                    if holding_minutes < min_holding_minutes:
                        min_holding_minutes = holding_minutes
                    if holding_minutes > max_holding_minutes:
                        max_holding_minutes = holding_minutes
                quantity = 0
                entry_index = -1

        equity = cash + quantity * close[i]
        if equity > peak_equity:
            peak_equity = equity
        if peak_equity > 0:
            dd = equity / peak_equity - 1.0
            if dd < max_drawdown:
                max_drawdown = dd

    if quantity > 0 and end > start:
        exit_price = close[end - 1] * (1.0 - slip)
        proceeds = quantity * exit_price - commission
        pnl = proceeds - entry_total_cost
        cash += proceeds
        trades += 1
        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += -pnl
        return_sum += (exit_price / entry_price - 1.0) * 100.0
        if entry_index >= 0:
            exit_index = end - 1
            holding_minutes = (timestamps_ns[exit_index] - timestamps_ns[entry_index]) / 60_000_000_000.0
            holding_bars = exit_index - entry_index
            total_holding_minutes += holding_minutes
            total_holding_bars += holding_bars
            if holding_minutes < min_holding_minutes:
                min_holding_minutes = holding_minutes
            if holding_minutes > max_holding_minutes:
                max_holding_minutes = holding_minutes

    total_return_pct = (cash / initial_cash - 1.0) * 100.0
    max_drawdown_pct = max_drawdown * 100.0
    win_rate_pct = (wins / trades * 100.0) if trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (5.0 if gross_profit > 0 else 0.0)
    avg_trade_return_pct = (return_sum / trades) if trades else 0.0
    avg_holding_minutes = (total_holding_minutes / trades) if trades else 0.0
    avg_holding_bars = (total_holding_bars / trades) if trades else 0.0
    if trades == 0:
        min_holding_minutes = 0.0
    return (
        total_return_pct, max_drawdown_pct, trades, win_rate_pct, profit_factor,
        avg_trade_return_pct, avg_holding_minutes, min_holding_minutes,
        max_holding_minutes, avg_holding_bars,
    )


fast_backtest: Callable[..., tuple[float, float, int, float, float, float, float, float, float, float]]
if njit is not None:
    fast_backtest = njit(cache=True)(_fast_backtest_impl)
else:
    fast_backtest = _fast_backtest_impl


def evaluate(arrays: tuple[np.ndarray, ...], start: int, end: int, c: Candidate, e: EngineSettings) -> dict[str, float]:
    values = fast_backtest(
        *arrays, start, end,
        c.rsi_buy_min, c.rsi_buy_max, c.rsi_sell,
        c.relative_volume_min, c.macd_hist_min,
        c.require_rsi_rising, c.require_volume_rising, c.require_macd_rising,
        c.use_15m_confirmation, c.rsi_15m_min, c.relative_volume_15m_min,
        c.macd_hist_15m_min, c.required_15m_rising,
        c.stop_loss_pct, c.take_profit_pct,
        e.initial_cash, e.commission, e.slippage_bps, e.allocation_pct,
    )
    return {
        "total_return_pct": float(values[0]),
        "max_drawdown_pct": float(values[1]),
        "number_of_trades": int(values[2]),
        "win_rate_pct": float(values[3]),
        "profit_factor": float(values[4]),
        "average_trade_return_pct": float(values[5]),
        "average_holding_minutes": float(values[6]),
        "minimum_holding_minutes": float(values[7]),
        "maximum_holding_minutes": float(values[8]),
        "average_holding_bars": float(values[9]),
    }



def collect_trades(
    df: pd.DataFrame, arrays: tuple[np.ndarray, ...], start: int, end: int,
    c: Candidate, e: EngineSettings, timezone: str,
) -> pd.DataFrame:
    """Replay one configuration and return a detailed trade ledger.

    This is intentionally run only once for the best configuration, so the
    optimizer keeps using the fast compiled metrics-only backtest.
    """
    (
        timestamps_ns, open_, high, low, close, rsi, rel_volume, macd_hist,
        cross_up, cross_down, rsi_rising, volume_rising, macd_rising,
        rsi_15m, rel_volume_15m, macd_hist_15m,
        rsi_rising_15m, volume_rising_15m, macd_rising_15m,
    ) = arrays

    cash = e.initial_cash
    quantity = 0
    entry_price = 0.0
    entry_total_cost = 0.0
    entry_index = -1
    slip = e.slippage_bps / 10000.0
    records: list[dict[str, Any]] = []

    def close_trade(exit_index: int, raw_exit_price: float, reason: str) -> None:
        nonlocal cash, quantity, entry_price, entry_total_cost, entry_index
        exit_price = raw_exit_price * (1.0 - slip)
        proceeds = quantity * exit_price - e.commission
        pnl = proceeds - entry_total_cost
        pnl_pct = (pnl / entry_total_cost * 100.0) if entry_total_cost else 0.0
        holding_minutes = (timestamps_ns[exit_index] - timestamps_ns[entry_index]) / 60_000_000_000.0
        cash += proceeds
        records.append({
            "trade_number": len(records) + 1,
            "entry_time": df.index[entry_index].tz_convert(timezone),
            "exit_time": df.index[exit_index].tz_convert(timezone),
            "holding_minutes": holding_minutes,
            "holding_bars": exit_index - entry_index,
            "quantity": quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_entry_value": quantity * entry_price,
            "gross_exit_value": quantity * exit_price,
            "profit_loss": pnl,
            "profit_loss_pct": pnl_pct,
            "exit_reason": reason,
            "cash_after_trade": cash,
        })
        quantity = 0
        entry_index = -1

    for i in range(max(start + 1, 1), end):
        p = i - 1

        if quantity > 0 and (cross_down[p] or rsi[p] >= c.rsi_sell):
            reason = "MACD cross down" if cross_down[p] else "RSI sell threshold"
            close_trade(i, open_[i], reason)

        if quantity == 0:
            signal = (
                cross_up[p]
                and rsi[p] >= c.rsi_buy_min
                and rsi[p] <= c.rsi_buy_max
                and rel_volume[p] >= c.relative_volume_min
                and macd_hist[p] >= c.macd_hist_min
            )
            if c.require_rsi_rising:
                signal = signal and rsi_rising[p]
            if c.require_volume_rising:
                signal = signal and volume_rising[p]
            if c.require_macd_rising:
                signal = signal and macd_rising[p]
            if c.use_15m_confirmation:
                signal = (
                    signal
                    and rsi_15m[p] >= c.rsi_15m_min
                    and rel_volume_15m[p] >= c.relative_volume_15m_min
                    and macd_hist_15m[p] >= c.macd_hist_15m_min
                )
                if c.required_15m_rising:
                    signal = signal and rsi_rising_15m[p] and macd_rising_15m[p]

            if signal:
                entry_price = open_[i] * (1.0 + slip)
                usable = max(cash * e.allocation_pct - e.commission, 0.0)
                quantity = int(usable // entry_price)
                if quantity > 0:
                    entry_total_cost = quantity * entry_price + e.commission
                    cash -= entry_total_cost
                    entry_index = i

        if quantity > 0:
            stop_price = entry_price * (1.0 - c.stop_loss_pct)
            target_price = entry_price * (1.0 + c.take_profit_pct)
            if low[i] <= stop_price:
                close_trade(i, stop_price, "Stop loss")
            elif high[i] >= target_price:
                close_trade(i, target_price, "Take profit")

    if quantity > 0 and end > start:
        close_trade(end - 1, close[end - 1], "End of validation")

    return pd.DataFrame(records)

def score(stats: dict[str, float], minimum_trades: int) -> float:
    trades = int(stats["number_of_trades"])
    if trades < minimum_trades:
        # Mild enough to show progress, severe enough not to select tiny samples.
        return -10_000.0 - (minimum_trades - trades) * 100.0
    return (
        stats["total_return_pct"]
        - 1.25 * abs(stats["max_drawdown_pct"])
        + 1.5 * min(stats["profit_factor"], 5.0)
        + 0.02 * stats["win_rate_pct"]
        + min(trades, 100) * 0.01
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast 5-minute optimizer for one stock.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.30)
    p.add_argument("--minimum-trades", type=int, default=10)
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--commission", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--output-dir", default="optimization_output")
    p.add_argument("--cache-csv", default=None, help="Optional existing native 5-minute cache CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if not 0.10 <= args.validation_fraction <= 0.50:
        raise SystemExit("--validation-fraction must be between 0.10 and 0.50")

    ticker = args.ticker.upper()
    timezone = "America/New_York"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_csv) if args.cache_csv else output / f"{ticker}_5minute_raw_bars.csv"

    if cache_path.exists():
        print(f"Loading cached native 5-minute bars: {cache_path}")
        raw = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
        raw.index = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index.tz_convert("UTC")
    else:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise SystemExit("Set MASSIVE_API_KEY or provide --cache-csv.")
        warmup_start = (date.fromisoformat(args.start) - timedelta(days=75)).isoformat()
        raw = download_massive_bars(
            api_key=api_key,
            ticker=ticker,
            start_date=warmup_start,
            end_date=args.end,
            multiplier=5,
            timespan="minute",
        )
        raw.to_csv(cache_path, index_label="timestamp")
        print(f"Saved native 5-minute cache: {cache_path}")

    print("Precomputing 5-minute indicators and completed 15-minute confirmation...")
    df = build_indicator_cache(raw, timezone)
    local_index = df.index.tz_convert(timezone)
    requested_start = pd.Timestamp(args.start, tz=timezone)
    requested_end = pd.Timestamp(args.end, tz=timezone) + pd.Timedelta(days=1)
    df = df.loc[(local_index >= requested_start) & (local_index < requested_end)].copy()
    required = [
        "rsi", "relative_volume", "macd_histogram", "rsi_15m",
        "relative_volume_15m", "macd_histogram_15m",
    ]
    df = df.dropna(subset=required)
    if len(df) < 100:
        raise SystemExit("Not enough usable 5-minute rows after indicator warmup.")

    split_row, validation_start = split_indices(df, args.validation_fraction, timezone)
    arrays = to_arrays(df)
    engine = EngineSettings(args.initial_cash, args.commission, args.slippage_bps)
    print(
        f"Train rows: {split_row:,}; validation rows: {len(df)-split_row:,}; "
        f"validation starts {validation_start}"
    )
    print("Backtest engine: " + ("Numba JIT" if njit is not None else "Python array fallback"))

    # Trigger JIT compilation once before timing/progress output.
    warmup_candidate = sample_candidate(random.Random(args.seed))
    evaluate(arrays, 0, min(split_row, 500), warmup_candidate, engine)

    rng = random.Random(args.seed)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    trial = 0
    while trial < args.trials:
        candidate = sample_candidate(rng)
        signature = json.dumps(asdict(candidate), sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        trial += 1

        train_stats = evaluate(arrays, 0, split_row, candidate, engine)
        validation_stats = evaluate(arrays, split_row, len(df), candidate, engine)
        train_score = score(train_stats, args.minimum_trades)
        validation_score = score(validation_stats, args.minimum_trades)
        degradation = max(0.0, train_stats["total_return_pct"] - validation_stats["total_return_pct"])
        robust_score = validation_score + 0.25 * train_score - 0.20 * degradation

        row: dict[str, Any] = {
            "trial": trial,
            "robust_score": robust_score,
            "train_score": train_score,
            "validation_score": validation_score,
        }
        row.update({f"param_{k}": v for k, v in asdict(candidate).items()})
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"validation_{k}": v for k, v in validation_stats.items()})
        rows.append(row)

        if trial == 1 or trial % 100 == 0 or trial == args.trials:
            best = max(rows, key=lambda x: x["robust_score"])
            print(
                f"Trial {trial:,}/{args.trials:,} | best score={best['robust_score']:.2f} | "
                f"validation return={best['validation_total_return_pct']:.2f}% | "
                f"validation trades={int(best['validation_number_of_trades'])}"
            )

    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "validation_total_return_pct"], ascending=False
    )
    results_path = output / f"{ticker}_{args.trials}_5minute_optimization_results.csv"
    top_path = output / f"{ticker}_top_25_5minute_configurations.csv"
    best_path = output / f"{ticker}_best_5minute_configuration.json"
    results.to_csv(results_path, index=False)
    results.head(25).to_csv(top_path, index=False)

    best = results.iloc[0]
    parameters = {c.removeprefix("param_"): best[c] for c in results.columns if c.startswith("param_")}
    best_path.write_text(json.dumps({
        "ticker": ticker,
        "bar_size": "5 minutes",
        "optimization_start": args.start,
        "optimization_end": args.end,
        "validation_start": validation_start,
        "trials": len(results),
        "parameters": parameters,
        "train_metrics": {c.removeprefix("train_"): best[c] for c in results.columns if c.startswith("train_")},
        "validation_metrics": {c.removeprefix("validation_"): best[c] for c in results.columns if c.startswith("validation_")},
        "robust_score": best["robust_score"],
    }, indent=2, default=float), encoding="utf-8")

    print("\nBEST 5-MINUTE CONFIGURATION")
    print("-" * 72)
    for key, value in parameters.items():
        print(f"{key:32s}: {value}")
    print(f"Validation return:               {best['validation_total_return_pct']:.2f}%")
    print(f"Validation max drawdown:         {best['validation_max_drawdown_pct']:.2f}%")
    print(f"Validation trades:               {int(best['validation_number_of_trades'])}")
    print(f"Validation win rate:             {best['validation_win_rate_pct']:.2f}%")
    print(f"Validation profit factor:        {best['validation_profit_factor']:.2f}")
    avg_minutes = float(best['validation_average_holding_minutes'])
    min_minutes = float(best['validation_minimum_holding_minutes'])
    max_minutes = float(best['validation_maximum_holding_minutes'])
    print(f"Validation average hold:         {avg_minutes:.1f} min ({avg_minutes / 60.0:.2f} hr)")
    print(f"Validation minimum hold:         {min_minutes:.1f} min")
    print(f"Validation maximum hold:         {max_minutes:.1f} min ({max_minutes / 60.0:.2f} hr)")
    print(f"Validation average hold bars:    {best['validation_average_holding_bars']:.1f} x 5-min bars")

    # pandas/numpy may load booleans as numpy.bool_; normalize all fields explicitly.
    best_candidate = Candidate(
        rsi_buy_min=float(parameters["rsi_buy_min"]),
        rsi_buy_max=float(parameters["rsi_buy_max"]),
        rsi_sell=float(parameters["rsi_sell"]),
        relative_volume_min=float(parameters["relative_volume_min"]),
        macd_hist_min=float(parameters["macd_hist_min"]),
        require_rsi_rising=bool(parameters["require_rsi_rising"]),
        require_volume_rising=bool(parameters["require_volume_rising"]),
        require_macd_rising=bool(parameters["require_macd_rising"]),
        use_15m_confirmation=bool(parameters["use_15m_confirmation"]),
        rsi_15m_min=float(parameters["rsi_15m_min"]),
        relative_volume_15m_min=float(parameters["relative_volume_15m_min"]),
        macd_hist_15m_min=float(parameters["macd_hist_15m_min"]),
        required_15m_rising=bool(parameters["required_15m_rising"]),
        stop_loss_pct=float(parameters["stop_loss_pct"]),
        take_profit_pct=float(parameters["take_profit_pct"]),
    )
    trades = collect_trades(df, arrays, split_row, len(df), best_candidate, engine, timezone)
    trades_path = output / f"{ticker}_best_validation_trades.csv"
    trades.to_csv(trades_path, index=False)

    print("\nVALIDATION TRADES")
    print("-" * 132)
    if trades.empty:
        print("No validation trades.")
    else:
        display = trades.copy()
        display["entry_time"] = display["entry_time"].map(lambda x: x.strftime("%Y-%m-%d %H:%M %Z"))
        display["exit_time"] = display["exit_time"].map(lambda x: x.strftime("%Y-%m-%d %H:%M %Z"))
        for row in display.itertuples(index=False):
            outcome = "PROFIT" if row.profit_loss >= 0 else "LOSS"
            print(
                f"#{row.trade_number:02d} | {row.entry_time} -> {row.exit_time} | "
                f"hold {row.holding_minutes:.0f} min | qty {row.quantity} | "
                f"${row.entry_price:.2f} -> ${row.exit_price:.2f} | "
                f"{outcome}: ${row.profit_loss:,.2f} ({row.profit_loss_pct:+.2f}%) | {row.exit_reason}"
            )
        print("-" * 132)
        print(f"Net validation P/L: ${trades['profit_loss'].sum():,.2f}")

    print("\nSaved:")
    print(f"  All trials:       {results_path}")
    print(f"  Top 25:           {top_path}")
    print(f"  Best JSON:        {best_path}")
    print(f"  Validation trades:{trades_path}")


if __name__ == "__main__":
    main()
