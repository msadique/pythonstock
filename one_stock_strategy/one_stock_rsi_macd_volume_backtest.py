#!/usr/bin/env python3
"""
One-stock intraday RSI + MACD + relative-volume backtester using Massive.com.

Core behavior:
- Downloads OHLCV aggregate bars for ONE ticker.
- Supports minute/hour/day-style Massive aggregate intervals.
- Computes RSI and MACD.
- Computes a 30-session average volume for the SAME intraday time slot.
  Example: today's 10:15 bar is compared with the previous 30 trading
  sessions' 10:15 bars.
- Detects volume spikes using current_volume / prior_30_session_avg_volume.
- Creates BUY/SELL signals.
- Backtests long-only trades over a requested date range.
- Saves enriched bars, trades, and an equity curve to CSV.

This is a research/backtesting example, not investment advice and not a
production live-trading engine.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


MASSIVE_AGGS_URL = (
    "https://api.massive.com/v2/aggs/ticker/{ticker}/range/"
    "{multiplier}/{timespan}/{start}/{end}"
)


@dataclass(frozen=True)
class StrategyConfig:
    ticker: str
    start_date: str
    end_date: str
    multiplier: int = 1
    timespan: str = "minute"

    # Indicator settings
    rsi_period: int = 14
    rsi_buy_level: float = 35.0
    rsi_sell_level: float = 65.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Relative-volume settings
    volume_lookback_sessions: int = 30
    volume_spike_multiple: float = 1.5

    # Multi-timeframe bullish confirmation (2m, 5m, 15m).
    confirmation_timeframes: tuple[int, ...] = (2, 5, 15)
    required_confirmations: int = 3
    rsi_2m_min: float = 45.0
    rsi_5m_min: float = 48.0
    rsi_15m_min: float = 50.0
    volume_2m_min: float = 1.20
    volume_5m_min: float = 1.15
    volume_15m_min: float = 1.10
    macd_hist_2m_min: float = 0.0
    macd_hist_5m_min: float = 0.0
    macd_hist_15m_min: float = 0.0
    require_rising_rsi: bool = True
    require_rising_volume: bool = True
    require_rising_macd: bool = True

    # Backtest settings
    initial_cash: float = 100_000.0
    allocation_pct: float = 1.0
    commission_per_order: float = 0.0
    slippage_bps: float = 2.0
    stop_loss_pct: float | None = 0.03
    take_profit_pct: float | None = 0.06

    # Data/session settings
    regular_hours_only: bool = True
    timezone: str = "America/New_York"


@dataclass
class Position:
    quantity: int
    entry_time: pd.Timestamp
    entry_price: float
    entry_fee: float
    entry_reason: str


def validate_config(config: StrategyConfig) -> None:
    try:
        start = date.fromisoformat(config.start_date)
        end = date.fromisoformat(config.end_date)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format.") from exc

    if end < start:
        raise ValueError("end_date must be on or after start_date.")
    if config.multiplier < 1:
        raise ValueError("multiplier must be at least 1.")
    if config.timespan not in {
        "second", "minute", "hour", "day", "week", "month", "quarter", "year"
    }:
        raise ValueError("Unsupported timespan.")
    if not 0 < config.allocation_pct <= 1:
        raise ValueError("allocation_pct must be greater than 0 and at most 1.")
    if config.macd_fast >= config.macd_slow:
        raise ValueError("macd_fast must be smaller than macd_slow.")
    if config.volume_lookback_sessions < 1:
        raise ValueError("volume_lookback_sessions must be at least 1.")
    if config.volume_spike_multiple <= 0:
        raise ValueError("volume_spike_multiple must be positive.")
    if config.required_confirmations < 1 or config.required_confirmations > len(config.confirmation_timeframes):
        raise ValueError("required_confirmations must be between 1 and the number of confirmation timeframes.")
    if config.timespan != "minute" or config.multiplier != 1:
        raise ValueError(
            "Multi-timeframe 2m/5m/15m confirmation requires --timespan minute --multiplier 1."
        )


def chunks_by_calendar_days(
    start: date,
    end: date,
    days_per_chunk: int,
) -> Iterable[tuple[date, date]]:
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days_per_chunk - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def request_json(
    url: str,
    params: dict[str, Any],
    timeout_seconds: int = 30,
    max_retries: int = 5,
) -> dict[str, Any]:
    retryable_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout_seconds)

            if response.status_code in retryable_statuses:
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"Massive API returned {response.status_code}; "
                    f"retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            payload = response.json()

            if payload.get("status") == "ERROR":
                raise RuntimeError(
                    f"Massive API error: {payload.get('error', payload)}"
                )
            return payload

        except (requests.RequestException, ValueError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Massive API request failed: {exc}") from exc
            wait_seconds = min(2 ** attempt, 30)
            print(f"Request failed; retrying in {wait_seconds}s: {exc}")
            time.sleep(wait_seconds)

    raise RuntimeError("Unexpected request failure.")


def download_massive_bars(
    api_key: str,
    ticker: str,
    start_date: str,
    end_date: str,
    multiplier: int,
    timespan: str,
) -> pd.DataFrame:
    """
    Download aggregate bars in chunks and follow Massive's next_url pagination.

    Smaller chunks are used for second/minute bars to reduce the chance that
    a response hits the endpoint result limit.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    if timespan == "second":
        chunk_days = 2
    elif timespan == "minute":
        chunk_days = 20
    elif timespan == "hour":
        chunk_days = 120
    else:
        chunk_days = 3650

    records: list[dict[str, Any]] = []

    for chunk_start, chunk_end in chunks_by_calendar_days(start, end, chunk_days):
        url = MASSIVE_AGGS_URL.format(
            ticker=ticker.upper(),
            multiplier=multiplier,
            timespan=timespan,
            start=chunk_start.isoformat(),
            end=chunk_end.isoformat(),
        )
        params: dict[str, Any] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50_000,
            "apiKey": api_key,
        }

        while url:
            payload = request_json(url, params=params)
            records.extend(payload.get("results", []))

            next_url = payload.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": api_key}
            else:
                url = ""

        print(
            f"Downloaded {ticker.upper()} bars through "
            f"{chunk_end.isoformat()}"
        )

    if not records:
        raise RuntimeError(
            "No aggregate bars were returned. Check the ticker, date range, "
            "subscription entitlement, and API key."
        )

    df = pd.DataFrame.from_records(records)
    rename_map = {
        "t": "timestamp_ms",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "transactions",
    }
    df = df.rename(columns=rename_map)

    required = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"Massive response is missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"], unit="ms", utc=True
    )
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")
    )

    numeric_columns = [
        column
        for column in [
            "open", "high", "low", "close", "volume", "vwap", "transactions"
        ]
        if column in df.columns
    ]
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    return df


def filter_regular_session(
    df: pd.DataFrame,
    timezone: str,
) -> pd.DataFrame:
    """
    Keep U.S. regular trading hours: 09:30 <= timestamp < 16:00 Eastern.
    """
    local_index = df.index.tz_convert(timezone)
    minute_of_day = local_index.hour * 60 + local_index.minute
    mask = (minute_of_day >= 9 * 60 + 30) & (minute_of_day < 16 * 60)
    return df.loc[mask].copy()


def add_rsi(
    df: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    result = df.copy()
    delta = result["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder-style smoothing uses alpha = 1 / period.
    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)
    result["rsi"] = 100 - (100 / (1 + relative_strength))

    # Explicitly handle one-sided windows.
    result.loc[
        (average_loss == 0) & (average_gain > 0), "rsi"
    ] = 100.0
    result.loc[
        (average_gain == 0) & (average_loss > 0), "rsi"
    ] = 0.0
    result.loc[
        (average_gain == 0) & (average_loss == 0), "rsi"
    ] = 50.0
    return result


def add_macd(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    result = df.copy()
    fast_ema = result["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = result["close"].ewm(span=slow, adjust=False).mean()

    result["macd"] = fast_ema - slow_ema
    result["macd_signal"] = (
        result["macd"].ewm(span=signal, adjust=False).mean()
    )
    result["macd_histogram"] = (
        result["macd"] - result["macd_signal"]
    )
    return result


def add_same_time_volume_baseline(
    df: pd.DataFrame,
    lookback_sessions: int,
    timezone: str,
) -> pd.DataFrame:
    """
    Calculate prior-N-session mean volume for each intraday time slot.

    Important anti-lookahead detail:
    The rolling mean is shifted by one observation within each slot, so the
    current bar's volume is NOT included in its own baseline.
    """
    result = df.copy()
    local_index = result.index.tz_convert(timezone)

    result["session_date"] = pd.Index(local_index.date)
    result["time_slot"] = (
        local_index.hour * 60
        + local_index.minute
        + local_index.second / 60.0
    )

    # This works for any bar interval because the slot comes from each bar's
    # actual local timestamp.
    result["volume_avg_30_sessions"] = (
        result.groupby("time_slot", sort=False)["volume"]
        .transform(
            lambda values: values.shift(1).rolling(
                window=lookback_sessions,
                min_periods=lookback_sessions,
            ).mean()
        )
    )

    result["relative_volume"] = (
        result["volume"] / result["volume_avg_30_sessions"]
    )
    return result


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Create completed N-minute OHLCV candles from one-minute bars."""
    rule = f"{minutes}min"
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "transactions" in df.columns:
        aggregation["transactions"] = "sum"

    # right/right means a 09:30-09:32 candle is labeled 09:32.  When joined
    # backward to the one-minute stream, only a completed higher-timeframe
    # candle is visible.
    result = df.resample(rule, label="right", closed="right").agg(aggregation)
    return result.dropna(subset=["open", "high", "low", "close"])


def add_multi_timeframe_confirmation(
    df: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Attach 2m/5m/15m indicator state and compute bullish confirmations."""
    result = df.copy().sort_index()
    thresholds = {
        2: (config.rsi_2m_min, config.volume_2m_min, config.macd_hist_2m_min),
        5: (config.rsi_5m_min, config.volume_5m_min, config.macd_hist_5m_min),
        15: (config.rsi_15m_min, config.volume_15m_min, config.macd_hist_15m_min),
    }

    confirmation_columns: list[str] = []

    for minutes in config.confirmation_timeframes:
        higher = resample_ohlcv(result, minutes)
        higher = add_rsi(higher, config.rsi_period)
        higher = add_macd(
            higher,
            config.macd_fast,
            config.macd_slow,
            config.macd_signal,
        )
        higher = add_same_time_volume_baseline(
            higher,
            config.volume_lookback_sessions,
            config.timezone,
        )

        suffix = f"_{minutes}m"
        selected = higher[[
            "close", "rsi", "macd", "macd_signal", "macd_histogram",
            "volume", "volume_avg_30_sessions", "relative_volume",
        ]].rename(columns=lambda column: f"{column}{suffix}")

        # Completed-bar values are carried forward until the next completed
        # candle. This avoids using an unfinished 5m/15m candle.
        result = result.join(selected, how="left").ffill()

        result[f"rsi_rising{suffix}"] = (
            result[f"rsi{suffix}"] > result[f"rsi{suffix}"].shift(1)
        )
        result[f"volume_rising{suffix}"] = (
            result[f"relative_volume{suffix}"]
            > result[f"relative_volume{suffix}"].shift(1)
        )
        result[f"macd_rising{suffix}"] = (
            result[f"macd_histogram{suffix}"]
            > result[f"macd_histogram{suffix}"].shift(1)
        )

        rsi_min, volume_min, macd_hist_min = thresholds[minutes]
        condition = (
            (result[f"rsi{suffix}"] >= rsi_min)
            & (result[f"relative_volume{suffix}"] >= volume_min)
            & (result[f"macd_histogram{suffix}"] >= macd_hist_min)
        )
        if config.require_rising_rsi:
            condition &= result[f"rsi_rising{suffix}"]
        if config.require_rising_volume:
            condition &= result[f"volume_rising{suffix}"]
        if config.require_rising_macd:
            condition &= result[f"macd_rising{suffix}"]

        confirmation_column = f"bull_confirm{suffix}"
        result[confirmation_column] = condition.fillna(False)
        confirmation_columns.append(confirmation_column)

    result["bull_confirmation_count"] = result[confirmation_columns].sum(axis=1)
    result["multi_timeframe_bullish"] = (
        result["bull_confirmation_count"] >= config.required_confirmations
    )
    return result


def add_signals(
    df: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Create long-only signals with 2m/5m/15m bullish confirmation."""
    result = df.copy()

    macd_cross_up = (
        (result["macd"] > result["macd_signal"])
        & (result["macd"].shift(1) <= result["macd_signal"].shift(1))
    )
    macd_cross_down = (
        (result["macd"] < result["macd_signal"])
        & (result["macd"].shift(1) >= result["macd_signal"].shift(1))
    )

    result["volume_spike"] = (
        result["relative_volume"] >= config.volume_spike_multiple
    )

    result["base_bull_setup"] = (
        macd_cross_up
        & (result["rsi"] <= config.rsi_buy_level)
        & result["volume_spike"]
    ).fillna(False)

    # A final bull signal is emitted only after enough higher-timeframe
    # confirmations indicate that momentum and participation are increasing.
    result["buy_signal"] = (
        result["base_bull_setup"]
        & result["multi_timeframe_bullish"]
    ).fillna(False)

    result["sell_signal"] = (
        macd_cross_down
        | (result["rsi"] >= config.rsi_sell_level)
    ).fillna(False)

    return result


def apply_buy_slippage(price: float, slippage_bps: float) -> float:
    return price * (1 + slippage_bps / 10_000.0)


def apply_sell_slippage(price: float, slippage_bps: float) -> float:
    return price * (1 - slippage_bps / 10_000.0)


def close_position(
    *,
    cash: float,
    position: Position,
    exit_time: pd.Timestamp,
    raw_exit_price: float,
    exit_reason: str,
    commission_per_order: float,
    slippage_bps: float,
) -> tuple[float, dict[str, Any]]:
    exit_price = apply_sell_slippage(raw_exit_price, slippage_bps)
    exit_fee = commission_per_order
    proceeds = position.quantity * exit_price - exit_fee
    updated_cash = cash + proceeds

    cost_basis = (
        position.quantity * position.entry_price + position.entry_fee
    )
    net_pnl = proceeds - cost_basis
    return_pct = (
        net_pnl / cost_basis if cost_basis > 0 else np.nan
    )

    trade = {
        "entry_time": position.entry_time,
        "exit_time": exit_time,
        "quantity": position.quantity,
        "entry_price": position.entry_price,
        "exit_price": exit_price,
        "entry_reason": position.entry_reason,
        "exit_reason": exit_reason,
        "entry_fee": position.entry_fee,
        "exit_fee": exit_fee,
        "net_pnl": net_pnl,
        "return_pct": return_pct,
        "holding_minutes": (
            exit_time - position.entry_time
        ).total_seconds() / 60.0,
    }
    return updated_cash, trade


def backtest(
    df: pd.DataFrame,
    config: StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Backtest with next-bar-open signal execution.

    Stop-loss and take-profit are checked against each later bar's high/low.
    If both are touched in the same candle, the stop is assumed to occur first
    (conservative handling of unknown intrabar path).
    """
    if len(df) < 2:
        raise ValueError("At least two bars are required for backtesting.")

    cash = config.initial_cash
    position: Position | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for i in range(1, len(df)):
        previous = df.iloc[i - 1]
        current = df.iloc[i]
        current_time = df.index[i]

        # 1. Execute previous bar's exit signal at current open.
        if position is not None and bool(previous["sell_signal"]):
            cash, trade = close_position(
                cash=cash,
                position=position,
                exit_time=current_time,
                raw_exit_price=float(current["open"]),
                exit_reason="indicator_exit",
                commission_per_order=config.commission_per_order,
                slippage_bps=config.slippage_bps,
            )
            trades.append(trade)
            position = None

        # 2. Execute previous bar's entry signal at current open.
        if position is None and bool(previous["buy_signal"]):
            entry_price = apply_buy_slippage(
                float(current["open"]), config.slippage_bps
            )
            capital_to_use = cash * config.allocation_pct
            affordable_cash = max(
                capital_to_use - config.commission_per_order,
                0.0,
            )
            quantity = int(affordable_cash // entry_price)

            if quantity > 0:
                entry_fee = config.commission_per_order
                cash -= quantity * entry_price + entry_fee
                position = Position(
                    quantity=quantity,
                    entry_time=current_time,
                    entry_price=entry_price,
                    entry_fee=entry_fee,
                    entry_reason="rsi_macd_relative_volume_multitimeframe",
                )

        # 3. Check stop/take-profit within current candle after entry/hold.
        if position is not None:
            stop_price = (
                position.entry_price * (1 - config.stop_loss_pct)
                if config.stop_loss_pct is not None
                else None
            )
            target_price = (
                position.entry_price * (1 + config.take_profit_pct)
                if config.take_profit_pct is not None
                else None
            )

            stop_hit = (
                stop_price is not None
                and float(current["low"]) <= stop_price
            )
            target_hit = (
                target_price is not None
                and float(current["high"]) >= target_price
            )

            raw_exit_price: float | None = None
            exit_reason: str | None = None

            if stop_hit:
                # Conservative if stop and target both occur in one bar.
                raw_exit_price = float(stop_price)
                exit_reason = "stop_loss"
            elif target_hit:
                raw_exit_price = float(target_price)
                exit_reason = "take_profit"

            if raw_exit_price is not None and exit_reason is not None:
                cash, trade = close_position(
                    cash=cash,
                    position=position,
                    exit_time=current_time,
                    raw_exit_price=raw_exit_price,
                    exit_reason=exit_reason,
                    commission_per_order=config.commission_per_order,
                    slippage_bps=config.slippage_bps,
                )
                trades.append(trade)
                position = None

        market_value = (
            position.quantity * float(current["close"])
            if position is not None
            else 0.0
        )
        equity_rows.append(
            {
                "timestamp": current_time,
                "cash": cash,
                "market_value": market_value,
                "equity": cash + market_value,
                "position_quantity": (
                    position.quantity if position is not None else 0
                ),
            }
        )

    # Liquidate any open position at the final bar close.
    if position is not None:
        final_time = df.index[-1]
        final_close = float(df.iloc[-1]["close"])
        cash, trade = close_position(
            cash=cash,
            position=position,
            exit_time=final_time,
            raw_exit_price=final_close,
            exit_reason="end_of_backtest",
            commission_per_order=config.commission_per_order,
            slippage_bps=config.slippage_bps,
        )
        trades.append(trade)
        position = None

        if equity_rows:
            equity_rows[-1]["cash"] = cash
            equity_rows[-1]["market_value"] = 0.0
            equity_rows[-1]["equity"] = cash
            equity_rows[-1]["position_quantity"] = 0

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("timestamp")

    if equity_df.empty:
        final_equity = config.initial_cash
        max_drawdown = 0.0
    else:
        final_equity = float(equity_df["equity"].iloc[-1])
        running_peak = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / running_peak - 1.0
        max_drawdown = float(drawdown.min())

    total_return = final_equity / config.initial_cash - 1.0

    if trades_df.empty:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade_return = 0.0
    else:
        wins = trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"]
        losses = trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"]
        win_rate = float((trades_df["net_pnl"] > 0).mean())
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        average_trade_return = float(
            trades_df["return_pct"].mean()
        )

    stats = {
        "initial_cash": float(config.initial_cash),
        "final_equity": final_equity,
        "net_profit": final_equity - config.initial_cash,
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "number_of_trades": float(len(trades_df)),
        "win_rate_pct": win_rate * 100.0,
        "profit_factor": profit_factor,
        "average_trade_return_pct": average_trade_return * 100.0,
    }

    return trades_df, equity_df, stats


def prepare_dataset(
    raw_df: pd.DataFrame,
    config: StrategyConfig,
) -> pd.DataFrame:
    df = raw_df.copy()

    if config.regular_hours_only and config.timespan in {
        "second", "minute", "hour"
    }:
        df = filter_regular_session(df, config.timezone)

    df = add_rsi(df, config.rsi_period)
    df = add_macd(
        df,
        config.macd_fast,
        config.macd_slow,
        config.macd_signal,
    )
    df = add_same_time_volume_baseline(
        df,
        config.volume_lookback_sessions,
        config.timezone,
    )
    df = add_multi_timeframe_confirmation(df, config)
    df = add_signals(df, config)
    return df


def print_stats(stats: dict[str, float]) -> None:
    print("\nBACKTEST RESULTS")
    print("-" * 55)
    for key, value in stats.items():
        if key in {
            "total_return_pct",
            "max_drawdown_pct",
            "win_rate_pct",
            "average_trade_return_pct",
        }:
            print(f"{key:30s}: {value:,.2f}%")
        elif key == "number_of_trades":
            print(f"{key:30s}: {int(value)}")
        else:
            print(f"{key:30s}: {value:,.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest one stock using RSI, MACD, and same-time "
            "30-session relative volume."
        )
    )
    parser.add_argument("--ticker", required=True, help="Example: AAPL")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument(
        "--timespan",
        choices=[
            "second", "minute", "hour", "day",
            "week", "month", "quarter", "year",
        ],
        default="minute",
    )
    parser.add_argument("--rsi-buy", type=float, default=35.0)
    parser.add_argument("--rsi-sell", type=float, default=65.0)
    parser.add_argument("--rsi-2m-min", type=float, default=45.0)
    parser.add_argument("--rsi-5m-min", type=float, default=48.0)
    parser.add_argument("--rsi-15m-min", type=float, default=50.0)
    parser.add_argument("--volume-2m-min", type=float, default=1.20)
    parser.add_argument("--volume-5m-min", type=float, default=1.15)
    parser.add_argument("--volume-15m-min", type=float, default=1.10)
    parser.add_argument("--macd-hist-2m-min", type=float, default=0.0)
    parser.add_argument("--macd-hist-5m-min", type=float, default=0.0)
    parser.add_argument("--macd-hist-15m-min", type=float, default=0.0)
    parser.add_argument(
        "--required-confirmations", type=int, default=3,
        help="How many of 2m/5m/15m must be bullish (1-3).",
    )
    parser.add_argument(
        "--volume-spike",
        type=float,
        default=1.5,
        help="1.5 means current volume is 150%% of its prior average.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--allocation", type=float, default=1.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--stop-loss", type=float, default=0.03)
    parser.add_argument("--take-profit", type=float, default=0.06)
    parser.add_argument(
        "--include-extended-hours",
        action="store_true",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set the MASSIVE_API_KEY environment variable first."
        )

    config = StrategyConfig(
        ticker=args.ticker.upper(),
        start_date=args.start,
        end_date=args.end,
        multiplier=args.multiplier,
        timespan=args.timespan,
        rsi_buy_level=args.rsi_buy,
        rsi_sell_level=args.rsi_sell,
        volume_spike_multiple=args.volume_spike,
        required_confirmations=args.required_confirmations,
        rsi_2m_min=args.rsi_2m_min,
        rsi_5m_min=args.rsi_5m_min,
        rsi_15m_min=args.rsi_15m_min,
        volume_2m_min=args.volume_2m_min,
        volume_5m_min=args.volume_5m_min,
        volume_15m_min=args.volume_15m_min,
        macd_hist_2m_min=args.macd_hist_2m_min,
        macd_hist_5m_min=args.macd_hist_5m_min,
        macd_hist_15m_min=args.macd_hist_15m_min,
        initial_cash=args.initial_cash,
        allocation_pct=args.allocation,
        commission_per_order=args.commission,
        slippage_bps=args.slippage_bps,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        regular_hours_only=not args.include_extended_hours,
    )
    validate_config(config)

    # Fetch extra history before the requested test period so the first test
    # day can already have a 30-session volume baseline and warm indicators.
    # 60 calendar days usually covers at least 30 U.S. trading sessions.
    warmup_calendar_days = max(
        config.volume_lookback_sessions * 2,
        60,
    )
    download_start = (
        date.fromisoformat(config.start_date)
        - timedelta(days=warmup_calendar_days)
    ).isoformat()

    print(
        f"Downloading {config.ticker}: {download_start} "
        f"through {config.end_date}"
    )
    raw_df = download_massive_bars(
        api_key=api_key,
        ticker=config.ticker,
        start_date=download_start,
        end_date=config.end_date,
        multiplier=config.multiplier,
        timespan=config.timespan,
    )
    enriched_df = prepare_dataset(raw_df, config)

    # Only process/backtest the user-requested range after all indicators and
    # historical baselines have been calculated from the warmup data.
    start_ts = pd.Timestamp(config.start_date, tz=config.timezone)
    end_exclusive = (
        pd.Timestamp(config.end_date, tz=config.timezone)
        + pd.Timedelta(days=1)
    )
    local_index = enriched_df.index.tz_convert(config.timezone)
    test_mask = (
        (local_index >= start_ts)
        & (local_index < end_exclusive)
    )
    test_df = enriched_df.loc[test_mask].copy()

    if test_df.empty:
        raise RuntimeError(
            "No bars remain inside the requested backtest range."
        )

    trades_df, equity_df, stats = backtest(test_df, config)
    print_stats(stats)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = (
        f"{config.ticker}_{config.multiplier}{config.timespan}_"
        f"{config.start_date}_{config.end_date}"
    )
    bars_path = output_dir / f"{prefix}_bars.csv"
    trades_path = output_dir / f"{prefix}_trades.csv"
    equity_path = output_dir / f"{prefix}_equity.csv"
    stats_path = output_dir / f"{prefix}_stats.csv"

    test_df.to_csv(bars_path)
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    print("\nSaved files:")
    print(f"  Bars:   {bars_path}")
    print(f"  Trades: {trades_path}")
    print(f"  Equity: {equity_path}")
    print(f"  Stats:  {stats_path}")

    latest = test_df.iloc[-1]
    print("\nLATEST BAR")
    print("-" * 55)
    print(f"Timestamp:        {test_df.index[-1]}")
    print(f"Close:            {latest['close']:.4f}")
    print(f"RSI:              {latest['rsi']:.2f}")
    print(f"MACD:             {latest['macd']:.6f}")
    print(f"MACD signal:      {latest['macd_signal']:.6f}")
    print(f"Volume:           {latest['volume']:,.0f}")
    print(
        "30-session avg:  "
        f"{latest['volume_avg_30_sessions']:,.0f}"
    )
    print(f"Relative volume:  {latest['relative_volume']:.2f}x")
    print(f"Volume spike:     {bool(latest['volume_spike'])}")
    for minutes in config.confirmation_timeframes:
        suffix = f"_{minutes}m"
        print(
            f"{minutes:>2}m confirmation: "
            f"RSI={latest[f'rsi{suffix}']:.2f}, "
            f"RVOL={latest[f'relative_volume{suffix}']:.2f}x, "
            f"MACD hist={latest[f'macd_histogram{suffix}']:.6f}, "
            f"bull={bool(latest[f'bull_confirm{suffix}'])}"
        )
    print(
        f"Bull confirmations: {int(latest['bull_confirmation_count'])}/"
        f"{len(config.confirmation_timeframes)}"
    )
    print(f"Buy signal:       {bool(latest['buy_signal'])}")
    print(f"Sell signal:      {bool(latest['sell_signal'])}")


if __name__ == "__main__":
    main()
