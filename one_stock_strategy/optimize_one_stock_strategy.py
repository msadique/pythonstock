#!/usr/bin/env python3
"""Random-search optimizer for the one-stock RSI/MACD/volume strategy.

Downloads one-minute data once, precomputes indicators once, then evaluates
1,000+ threshold/risk configurations. Uses chronological train/validation
splits to reduce (not eliminate) overfitting.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from one_stock_rsi_macd_volume_backtest import (
    StrategyConfig,
    add_macd,
    add_multi_timeframe_confirmation,
    add_rsi,
    add_same_time_volume_baseline,
    backtest,
    download_massive_bars,
    filter_regular_session,
    validate_config,
)


def sample_config(base: StrategyConfig, rng: random.Random) -> StrategyConfig:
    """Sample a practical configuration. Edit ranges here for wider searches."""
    stop = rng.choice([0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03])
    target = rng.choice([0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06])

    return replace(
        base,
        rsi_buy_level=rng.uniform(25.0, 45.0),
        rsi_sell_level=rng.uniform(55.0, 80.0),
        volume_spike_multiple=rng.uniform(1.05, 2.50),
        rsi_2m_min=rng.uniform(38.0, 62.0),
        rsi_5m_min=rng.uniform(38.0, 65.0),
        rsi_15m_min=rng.uniform(40.0, 68.0),
        volume_2m_min=rng.uniform(0.85, 2.00),
        volume_5m_min=rng.uniform(0.85, 1.80),
        volume_15m_min=rng.uniform(0.80, 1.60),
        # MACD histogram depends on stock price, so zero is generally a more
        # portable threshold; momentum strength is captured by the rising test.
        macd_hist_2m_min=0.0,
        macd_hist_5m_min=0.0,
        macd_hist_15m_min=0.0,
        required_confirmations=rng.choice([1, 2, 3]),
        require_rising_rsi=rng.choice([True, True, False]),
        require_rising_volume=rng.choice([True, True, False]),
        require_rising_macd=rng.choice([True, True, True, False]),
        stop_loss_pct=stop,
        take_profit_pct=target,
    )


def build_indicator_cache(raw_df: pd.DataFrame, base: StrategyConfig) -> pd.DataFrame:
    """Calculate price/volume indicators only once for every optimization run."""
    df = filter_regular_session(raw_df, base.timezone)
    df = add_rsi(df, base.rsi_period)
    df = add_macd(df, base.macd_fast, base.macd_slow, base.macd_signal)
    df = add_same_time_volume_baseline(
        df, base.volume_lookback_sessions, base.timezone
    )

    # This attaches raw 2m/5m/15m indicator values and rising flags. The
    # confirmation booleans created here are overwritten for every trial.
    permissive = replace(
        base,
        rsi_2m_min=0.0,
        rsi_5m_min=0.0,
        rsi_15m_min=0.0,
        volume_2m_min=0.0,
        volume_5m_min=0.0,
        volume_15m_min=0.0,
        macd_hist_2m_min=-float("inf"),
        macd_hist_5m_min=-float("inf"),
        macd_hist_15m_min=-float("inf"),
        required_confirmations=1,
        require_rising_rsi=False,
        require_rising_volume=False,
        require_rising_macd=False,
    )
    return add_multi_timeframe_confirmation(df, permissive)


def signals_for_config(cache: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Apply one candidate's thresholds to the cached indicators."""
    result = cache.copy(deep=False).copy()

    macd_cross_up = (
        (result["macd"] > result["macd_signal"])
        & (result["macd"].shift(1) <= result["macd_signal"].shift(1))
    )
    macd_cross_down = (
        (result["macd"] < result["macd_signal"])
        & (result["macd"].shift(1) >= result["macd_signal"].shift(1))
    )

    confirmations: list[pd.Series] = []
    thresholds = {
        2: (config.rsi_2m_min, config.volume_2m_min, config.macd_hist_2m_min),
        5: (config.rsi_5m_min, config.volume_5m_min, config.macd_hist_5m_min),
        15: (config.rsi_15m_min, config.volume_15m_min, config.macd_hist_15m_min),
    }

    for minutes in (2, 5, 15):
        suffix = f"_{minutes}m"
        rsi_min, volume_min, hist_min = thresholds[minutes]
        condition = (
            (result[f"rsi{suffix}"] >= rsi_min)
            & (result[f"relative_volume{suffix}"] >= volume_min)
            & (result[f"macd_histogram{suffix}"] >= hist_min)
        )
        if config.require_rising_rsi:
            condition &= result[f"rsi_rising{suffix}"]
        if config.require_rising_volume:
            condition &= result[f"volume_rising{suffix}"]
        if config.require_rising_macd:
            condition &= result[f"macd_rising{suffix}"]
        confirmations.append(condition.fillna(False))

    confirmation_count = sum(series.astype(np.int8) for series in confirmations)
    multi_tf = confirmation_count >= config.required_confirmations
    base_bull = (
        macd_cross_up
        & (result["rsi"] <= config.rsi_buy_level)
        & (result["relative_volume"] >= config.volume_spike_multiple)
    ).fillna(False)

    result["bull_confirmation_count"] = confirmation_count
    result["multi_timeframe_bullish"] = multi_tf
    result["base_bull_setup"] = base_bull
    result["buy_signal"] = (base_bull & multi_tf).fillna(False)
    result["sell_signal"] = (
        macd_cross_down | (result["rsi"] >= config.rsi_sell_level)
    ).fillna(False)
    return result


def score_stats(stats: dict[str, float], minimum_trades: int) -> float:
    """Risk-aware score. Invalid/too-small samples receive a severe penalty."""
    trades = int(stats["number_of_trades"])
    if trades < minimum_trades:
        return -1_000_000.0 + trades

    return_pct = stats["total_return_pct"]
    drawdown = abs(stats["max_drawdown_pct"])
    profit_factor = min(stats["profit_factor"], 5.0)
    win_rate = stats["win_rate_pct"]

    # Return matters, but drawdown and robustness matter too.
    return (
        return_pct
        - 1.25 * drawdown
        + 1.5 * profit_factor
        + 0.02 * win_rate
        + min(trades, 100) * 0.01
    )


def config_parameters(config: StrategyConfig) -> dict[str, Any]:
    keys = [
        "rsi_buy_level", "rsi_sell_level", "volume_spike_multiple",
        "rsi_2m_min", "rsi_5m_min", "rsi_15m_min",
        "volume_2m_min", "volume_5m_min", "volume_15m_min",
        "required_confirmations", "require_rising_rsi",
        "require_rising_volume", "require_rising_macd",
        "stop_loss_pct", "take_profit_pct",
    ]
    data = asdict(config)
    return {key: data[key] for key in keys}


def split_by_sessions(
    df: pd.DataFrame, validation_fraction: float, timezone: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    local_dates = pd.Index(df.index.tz_convert(timezone).date)
    sessions = sorted(pd.unique(local_dates))
    if len(sessions) < 10:
        raise ValueError("Use at least 10 trading sessions for optimization.")

    split_index = max(1, min(len(sessions) - 1, int(len(sessions) * (1 - validation_fraction))))
    validation_start = sessions[split_index]
    train = df.loc[local_dates < validation_start].copy()
    validation = df.loc[local_dates >= validation_start].copy()
    return train, validation, str(validation_start)


def run_trial(
    trial_number: int,
    cache_train: pd.DataFrame,
    cache_validation: pd.DataFrame,
    config: StrategyConfig,
    minimum_trades: int,
) -> dict[str, Any]:
    train_signals = signals_for_config(cache_train, config)
    _, _, train_stats = backtest(train_signals, config)
    train_score = score_stats(train_stats, minimum_trades)

    validation_signals = signals_for_config(cache_validation, config)
    _, _, validation_stats = backtest(validation_signals, config)
    validation_score = score_stats(validation_stats, minimum_trades)

    # Prefer configurations that work in both periods and penalize a large
    # train-to-validation collapse.
    degradation = max(
        0.0,
        train_stats["total_return_pct"] - validation_stats["total_return_pct"],
    )
    robust_score = validation_score + 0.25 * train_score - 0.20 * degradation

    row: dict[str, Any] = {
        "trial": trial_number,
        "robust_score": robust_score,
        "train_score": train_score,
        "validation_score": validation_score,
    }
    row.update({f"param_{k}": v for k, v in config_parameters(config).items()})
    row.update({f"train_{k}": v for k, v in train_stats.items()})
    row.update({f"validation_{k}": v for k, v in validation_stats.items()})
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize one stock strategy with 1,000+ random trials.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", required=True, help="Optimization start YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Optimization end YYYY-MM-DD")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--minimum-trades", type=int, default=5)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output-dir", default="optimization_output")
    parser.add_argument("--cache-csv", default=None, help="Optional existing raw Massive bars CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be at least 1")
    if not 0.10 <= args.validation_fraction <= 0.50:
        raise SystemExit("--validation-fraction must be between 0.10 and 0.50")

    base = StrategyConfig(
        ticker=args.ticker.upper(),
        start_date=args.start,
        end_date=args.end,
        multiplier=1,
        timespan="minute",
        initial_cash=args.initial_cash,
        commission_per_order=args.commission,
        slippage_bps=args.slippage_bps,
    )
    validate_config(base)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cache_path = Path(args.cache_csv) if args.cache_csv else output_dir / f"{base.ticker}_raw_bars.csv"

    if raw_cache_path.exists():
        print(f"Loading cached bars: {raw_cache_path}")
        raw = pd.read_csv(raw_cache_path, parse_dates=["timestamp"], index_col="timestamp")
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("UTC")
        else:
            raw.index = raw.index.tz_convert("UTC")
    else:
        api_key = os.getenv("MASSIVE_API_KEY")
        if not api_key:
            raise SystemExit("Set MASSIVE_API_KEY or provide --cache-csv.")
        warmup_start = (
            date.fromisoformat(args.start) - timedelta(days=75)
        ).isoformat()
        raw = download_massive_bars(
            api_key=api_key,
            ticker=base.ticker,
            start_date=warmup_start,
            end_date=args.end,
            multiplier=1,
            timespan="minute",
        )
        raw.to_csv(raw_cache_path, index_label="timestamp")
        print(f"Saved reusable raw-data cache: {raw_cache_path}")

    print("Precomputing RSI, MACD, volume baselines, and 2m/5m/15m indicators...")
    cache = build_indicator_cache(raw, base)

    requested_start = pd.Timestamp(args.start, tz=base.timezone)
    requested_end = pd.Timestamp(args.end, tz=base.timezone) + pd.Timedelta(days=1)
    local_index = cache.index.tz_convert(base.timezone)
    cache = cache.loc[(local_index >= requested_start) & (local_index < requested_end)].copy()
    cache = cache.dropna(subset=[
        "rsi", "relative_volume", "rsi_2m", "rsi_5m", "rsi_15m",
        "relative_volume_2m", "relative_volume_5m", "relative_volume_15m",
    ])
    if cache.empty:
        raise SystemExit("No usable rows after indicator warmup.")

    train, validation, validation_start = split_by_sessions(
        cache, args.validation_fraction, base.timezone
    )
    print(f"Train rows: {len(train):,}; validation rows: {len(validation):,}; validation starts {validation_start}")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for trial in range(1, args.trials + 1):
        candidate = sample_config(base, rng)
        signature = json.dumps(config_parameters(candidate), sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        row = run_trial(
            trial, train, validation, candidate, args.minimum_trades
        )
        rows.append(row)

        if trial == 1 or trial % 50 == 0 or trial == args.trials:
            best = max(rows, key=lambda item: item["robust_score"])
            print(
                f"Trial {trial:,}/{args.trials:,} | "
                f"best robust score={best['robust_score']:.2f} | "
                f"validation return={best['validation_total_return_pct']:.2f}% | "
                f"validation trades={int(best['validation_number_of_trades'])}"
            )

    results = pd.DataFrame(rows).sort_values(
        ["robust_score", "validation_total_return_pct"], ascending=False
    )
    results_path = output_dir / f"{base.ticker}_{args.trials}_optimization_results.csv"
    results.to_csv(results_path, index=False)

    top = results.head(25).copy()
    top_path = output_dir / f"{base.ticker}_top_25_configurations.csv"
    top.to_csv(top_path, index=False)

    best_row = results.iloc[0]
    best_parameters = {
        column.removeprefix("param_"): best_row[column]
        for column in results.columns
        if column.startswith("param_")
    }
    best_path = output_dir / f"{base.ticker}_best_configuration.json"
    best_path.write_text(
        json.dumps(
            {
                "ticker": base.ticker,
                "optimization_start": args.start,
                "optimization_end": args.end,
                "validation_start": validation_start,
                "trials": len(results),
                "parameters": best_parameters,
                "train_metrics": {
                    k.removeprefix("train_"): best_row[k]
                    for k in results.columns if k.startswith("train_")
                },
                "validation_metrics": {
                    k.removeprefix("validation_"): best_row[k]
                    for k in results.columns if k.startswith("validation_")
                },
                "robust_score": best_row["robust_score"],
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\nBEST CONFIGURATION")
    print("-" * 70)
    for key, value in best_parameters.items():
        print(f"{key:30s}: {value}")
    print(f"Validation return:             {best_row['validation_total_return_pct']:.2f}%")
    print(f"Validation max drawdown:       {best_row['validation_max_drawdown_pct']:.2f}%")
    print(f"Validation trades:             {int(best_row['validation_number_of_trades'])}")
    print(f"Validation win rate:           {best_row['validation_win_rate_pct']:.2f}%")
    print(f"Validation profit factor:      {best_row['validation_profit_factor']:.2f}")
    print("\nSaved:")
    print(f"  All trials: {results_path}")
    print(f"  Top 25:     {top_path}")
    print(f"  Best JSON:  {best_path}")


if __name__ == "__main__":
    main()
