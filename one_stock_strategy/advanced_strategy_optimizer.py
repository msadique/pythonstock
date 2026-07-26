#!/usr/bin/env python3
"""Advanced 5-minute strategy optimizer.

Implements:
1. Cooldown and minimum-hold controls.
2. Separate pullback and trend exits (trend trailing stop/EMA/MACD persistence).
3. Optuna optimization (TPE sampler and pruning-ready objective).
4. Expanding walk-forward validation.
5. Out-of-sample ML probability filter (LightGBM when available, sklearn fallback).
6. Market-regime filters using EMA trend and ATR volatility.

The ML model does not invent trades. Rule-based pullback/breakout logic creates
candidate entries; the model only estimates whether a candidate is likely to
reach a future target before a future stop.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from one_stock_rsi_macd_volume_backtest import download_massive_bars

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install Optuna first: pip install optuna") from exc

try:
    from lightgbm import LGBMClassifier
    MODEL_NAME = "LightGBM"
except ImportError:  # portable fallback
    from sklearn.ensemble import HistGradientBoostingClassifier
    LGBMClassifier = None
    MODEL_NAME = "sklearn HistGradientBoosting"

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Params:
    # Pullback entry
    rsi_buy_min: float
    rsi_buy_max: float
    pullback_volume_min: float
    require_pullback_macd_rising: bool

    # Trend entry
    trend_rsi_min: float
    trend_rsi_max: float
    trend_volume_min: float
    breakout_lookback: int

    # ML gate
    ml_probability_min: float

    # Shared risk
    stop_loss_pct: float
    pullback_take_profit_pct: float
    trend_trailing_stop_pct: float
    cooldown_bars: int
    minimum_hold_bars: int
    max_hold_bars: int

    # Entry-specific exits
    pullback_rsi_sell: float
    trend_macd_down_bars: int

    # Regime
    require_positive_long_trend: bool
    block_high_volatility: bool
    atr_percentile_limit: float


@dataclass(frozen=True)
class Engine:
    initial_cash: float
    commission: float
    slippage_bps: float
    allocation_pct: float = 1.0


def filter_trading_session(
    df: pd.DataFrame,
    timezone: str = "America/New_York",
    session_start: str = "08:00",
    session_end: str = "16:00",
) -> pd.DataFrame:
    """Keep only bars whose timestamps are inside [session_start, session_end).

    Filtering happens before RSI, MACD, volume, ATR, ML labels, signals, and
    walk-forward calculations, so no pre-8:00 AM or post-4:00 PM bars can
    influence the strategy. The 4:00 PM timestamp is excluded because a
    5-minute bar stamped 16:00 represents trading after the requested window.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")

    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    local = out.index.tz_convert(timezone)
    start_time = pd.Timestamp(session_start).time()
    end_time = pd.Timestamp(session_end).time()
    keep = (local.time >= start_time) & (local.time < end_time)
    return out.loc[keep].sort_index()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(avg_loss.ne(0.0), 100.0)


def _add_session_indicators(session: pd.DataFrame) -> pd.DataFrame:
    """Calculate all price indicators from one trading day only."""
    out = session.copy()
    out["rsi"] = _rsi(out["close"], 14)
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
    fast = out["close"].ewm(span=12, adjust=False).mean()
    slow = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = fast - slow
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]
    out["macd_cross_up"] = (out["macd"] > out["macd_signal"]) & (out["macd"].shift(1) <= out["macd_signal"].shift(1))
    out["macd_cross_down"] = (out["macd"] < out["macd_signal"]) & (out["macd"].shift(1) >= out["macd_signal"].shift(1))
    out["macd_rising"] = out["macd_histogram"] > out["macd_histogram"].shift(1)

    for lookback in (6, 12, 18, 24):
        out[f"breakout_{lookback}"] = out["close"] > out["high"].shift(1).rolling(lookback, min_periods=lookback).max()

    prev_close = out["close"].shift(1)
    true_range = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = true_range.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / out["close"]
    out["atr_percentile"] = out["atr_pct"].rolling(48, min_periods=14).rank(pct=True)

    out["ema_50_slope"] = out["ema_50"].pct_change(6)
    out["ema_20_distance"] = out["close"] / out["ema_20"] - 1.0
    out["ema_50_distance"] = out["close"] / out["ema_50"] - 1.0
    out["return_1"] = out["close"].pct_change(1)
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)
    out["return_12"] = out["close"].pct_change(12)
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["macd_slope"] = out["macd_histogram"].diff()
    out["rsi_slope"] = out["rsi"].diff()
    out["volume_log"] = np.log1p(out["volume"])
    return out


def _add_15m_session_indicators(session: pd.DataFrame) -> pd.DataFrame:
    """Build 15-minute indicators independently for one day and align to 5-minute bars."""
    bars15 = session[["open", "high", "low", "close", "volume"]].resample(
        "15min", label="right", closed="right"
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    bars15["rsi_15m"] = _rsi(bars15["close"], 14)
    fast = bars15["close"].ewm(span=12, adjust=False).mean()
    slow = bars15["close"].ewm(span=26, adjust=False).mean()
    macd = fast - slow
    bars15["macd_histogram_15m"] = macd - macd.ewm(span=9, adjust=False).mean()
    aligned = bars15[["rsi_15m", "macd_histogram_15m"]].reindex(session.index, method="ffill")
    return aligned


def build_intraday_indicator_cache(raw: pd.DataFrame, timezone: str, volume_lookback_sessions: int = 20) -> pd.DataFrame:
    """Build indicators with a hard reset at the start of every New York trading day.

    Relative volume is time-of-day aware: each bar is compared with the same
    local clock slot over prior sessions only, preventing look-ahead bias.
    """
    local_dates = pd.Series(raw.index.tz_convert(timezone).date, index=raw.index)
    pieces: list[pd.DataFrame] = []
    for _, session in raw.groupby(local_dates, sort=True):
        day = _add_session_indicators(session)
        aligned15 = _add_15m_session_indicators(session)
        day = day.join(aligned15)
        pieces.append(day)
    out = pd.concat(pieces).sort_index()

    local = out.index.tz_convert(timezone)
    out["session_date"] = pd.Series(local.date, index=out.index)
    out["time_slot"] = local.strftime("%H:%M")
    baseline = out.groupby("time_slot")["volume"].transform(
        lambda x: x.shift(1).rolling(volume_lookback_sessions, min_periods=5).mean()
    )
    out["relative_volume"] = out["volume"] / baseline.replace(0.0, np.nan)

    # 15-minute relative volume uses each completed 15-minute clock bucket.
    out["slot_15m"] = (local.hour * 60 + local.minute) // 15
    vol15 = out.groupby(["session_date", "slot_15m"])["volume"].transform("sum")
    first_in_bucket = ~pd.MultiIndex.from_arrays([out["session_date"], out["slot_15m"]]).duplicated()
    bucket_frame = pd.DataFrame({
        "session_date": out.loc[first_in_bucket, "session_date"],
        "slot_15m": out.loc[first_in_bucket, "slot_15m"],
        "volume_15m": vol15[first_in_bucket],
    }, index=out.index[first_in_bucket])
    bucket_frame["volume_15m_baseline"] = bucket_frame.groupby("slot_15m")["volume_15m"].transform(
        lambda x: x.shift(1).rolling(volume_lookback_sessions, min_periods=5).mean()
    )
    bucket_frame["relative_volume_15m"] = bucket_frame["volume_15m"] / bucket_frame["volume_15m_baseline"].replace(0.0, np.nan)
    out["relative_volume_15m"] = bucket_frame["relative_volume_15m"].reindex(out.index).ffill()
    same_day = out["session_date"].eq(out["session_date"].shift(1))
    out.loc[~same_day, "relative_volume_15m"] = bucket_frame["relative_volume_15m"].reindex(out.index)
    out["relative_volume_15m"] = out.groupby("session_date")["relative_volume_15m"].ffill()

    minutes = local.hour * 60 + local.minute
    out["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    out["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    out["regime_uptrend"] = (
        (out["close"] > out["ema_20"]) & (out["ema_20"] > out["ema_50"]) & (out["ema_50_slope"] > 0)
    )
    out["regime_long_positive"] = (out["ema_20"] > out["ema_50"]) & (out["ema_50_slope"] >= 0)
    out["regime_high_vol"] = out["atr_percentile"] >= 0.85
    return out


FEATURES = [
    "rsi", "relative_volume", "macd_histogram", "macd_slope", "rsi_slope",
    "rsi_15m", "relative_volume_15m", "macd_histogram_15m",
    "ema_20_distance", "ema_50_distance", "ema_50_slope",
    "atr_pct", "atr_percentile", "return_1", "return_3", "return_6",
    "return_12", "range_pct", "volume_log", "time_sin", "time_cos",
]


def make_path_label(df: pd.DataFrame, horizon: int, target_pct: float, stop_pct: float, timezone: str) -> pd.Series:
    """Label paths only within the same trading session; never cross overnight."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    dates = np.asarray(df.index.tz_convert(timezone).date)
    y = np.full(len(df), np.nan)
    for i in range(len(df) - 1):
        entry = close[i]
        target = entry * (1 + target_pct)
        stop = entry * (1 - stop_pct)
        label = 0.0
        last = min(len(df), i + horizon + 1)
        saw_future_bar = False
        for j in range(i + 1, last):
            if dates[j] != dates[i]:
                break
            saw_future_bar = True
            if low[j] <= stop:
                label = 0.0
                break
            if high[j] >= target:
                label = 1.0
                break
        if saw_future_bar:
            y[i] = label
    return pd.Series(y, index=df.index, name="ml_label")


def fit_predict_probabilities(
    df: pd.DataFrame, train_start: int, train_end: int, val_start: int, val_end: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    train = df.iloc[train_start:train_end]
    val = df.iloc[val_start:val_end]
    train_mask = train[FEATURES].notna().all(axis=1) & train["ml_label"].notna()
    val_mask = val[FEATURES].notna().all(axis=1)
    x_train = train.loc[train_mask, FEATURES]
    y_train = train.loc[train_mask, "ml_label"].astype(int)
    result = np.full(len(val), 0.5, dtype=float)
    if len(x_train) < 300 or y_train.nunique() < 2:
        return result, float("nan")

    if LGBMClassifier is not None:
        model = LGBMClassifier(
            n_estimators=250, learning_rate=0.04, num_leaves=31,
            max_depth=-1, subsample=0.85, colsample_bytree=0.85,
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        )
    model.fit(x_train, y_train)
    if val_mask.any():
        result[val_mask.to_numpy()] = model.predict_proba(val.loc[val_mask, FEATURES])[:, 1]

    auc = float("nan")
    known = val_mask & val["ml_label"].notna()
    if known.sum() > 20 and val.loc[known, "ml_label"].nunique() == 2:
        auc = roc_auc_score(val.loc[known, "ml_label"], result[known.to_numpy()])
    return result, float(auc)


def candidate_signals(df: pd.DataFrame, p: Params) -> tuple[np.ndarray, np.ndarray]:
    pullback = (
        df["macd_cross_up"]
        & df["rsi"].between(p.rsi_buy_min, p.rsi_buy_max)
        & (df["relative_volume"] >= p.pullback_volume_min)
        & (df["macd_histogram"] >= 0)
    )
    if p.require_pullback_macd_rising:
        pullback &= df["macd_rising"]

    breakout = df[f"breakout_{p.breakout_lookback}"]
    trend = (
        df["rsi"].between(p.trend_rsi_min, p.trend_rsi_max)
        & (df["relative_volume"] >= p.trend_volume_min)
        & (df["macd_histogram"] > 0)
        & df["macd_rising"]
        & df["regime_uptrend"]
        & breakout
    )
    if p.require_positive_long_trend:
        pullback &= df["regime_long_positive"]
    if p.block_high_volatility:
        vol_ok = df["atr_percentile"] <= p.atr_percentile_limit
        pullback &= vol_ok
        trend &= vol_ok
    return pullback.fillna(False).to_numpy(bool), trend.fillna(False).to_numpy(bool)


def backtest(
    df: pd.DataFrame, start: int, end: int, p: Params, engine: Engine,
    ml_probability: np.ndarray, collect: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    pullback, trend = candidate_signals(df, p)
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    rsi = df["rsi"].to_numpy(float); ema20 = df["ema_20"].to_numpy(float)
    macd = df["macd_histogram"].to_numpy(float)
    ts = df.index
    session_dates = np.asarray(ts.tz_convert("America/New_York").date)
    same_session_as_previous = np.r_[False, session_dates[1:] == session_dates[:-1]]
    is_session_last_bar = np.r_[session_dates[:-1] != session_dates[1:], True]

    cash = engine.initial_cash
    qty = 0; entry = 0.0; entry_cost = 0.0; entry_i = -1
    entry_type = ""; highest = 0.0; cooldown_until = start
    peak = cash; max_dd = 0.0; wins = 0; gross_profit = 0.0; gross_loss = 0.0
    trade_returns: list[float] = []; records: list[dict[str, Any]] = []
    consecutive_macd_down = 0
    slip = engine.slippage_bps / 10000.0

    def close_trade(i: int, raw_price: float, reason: str) -> None:
        nonlocal cash, qty, entry, entry_cost, entry_i, entry_type, highest
        nonlocal wins, gross_profit, gross_loss, cooldown_until, consecutive_macd_down
        exit_price = raw_price * (1 - slip)
        proceeds = qty * exit_price - engine.commission
        pnl = proceeds - entry_cost
        cash += proceeds
        ret = (exit_price / entry - 1) * 100
        trade_returns.append(ret)
        if pnl > 0: wins += 1; gross_profit += pnl
        elif pnl < 0: gross_loss += -pnl
        if collect:
            records.append({
                "entry_time": str(ts[entry_i].tz_convert("America/New_York")), "exit_time": str(ts[i].tz_convert("America/New_York")),
                "entry_type": entry_type, "entry_price": entry, "exit_price": exit_price,
                "quantity": qty, "profit_loss": pnl, "profit_loss_pct": ret,
                "holding_bars": i - entry_i,
                "holding_minutes": (i - entry_i) * 5.0,
                "exit_reason": reason, "ml_probability": float(ml_probability[entry_i]),
            })
        cooldown_until = i + p.cooldown_bars
        qty = 0; entry_i = -1; entry_type = ""; highest = 0.0; consecutive_macd_down = 0

    for i in range(max(start + 1, 1), end):
        prev = i - 1
        # Every trading day begins with a clean cooldown and strategy state.
        if not same_session_as_previous[i]:
            cooldown_until = i
            consecutive_macd_down = 0
        if qty > 0:
            highest = max(highest, h[i])
            bars_held = i - entry_i
            if is_session_last_bar[i]:
                close_trade(i, c[i], "Session close")
            stop = entry * (1 - p.stop_loss_pct)
            if qty > 0 and l[i] <= stop:
                close_trade(i, stop, "Stop loss")
            elif qty > 0 and entry_type == "Pullback":
                target = entry * (1 + p.pullback_take_profit_pct)
                if h[i] >= target:
                    close_trade(i, target, "Pullback take profit")
                elif bars_held >= p.minimum_hold_bars and (df["macd_cross_down"].iat[prev] or rsi[prev] >= p.pullback_rsi_sell):
                    close_trade(i, o[i], "Pullback indicator exit")
            elif qty > 0:  # Trend-specific exits
                consecutive_macd_down = consecutive_macd_down + 1 if macd[prev] < macd[prev - 1] else 0
                trailing = highest * (1 - p.trend_trailing_stop_pct)
                if l[i] <= trailing:
                    close_trade(i, trailing, "Trend trailing stop")
                elif bars_held >= p.minimum_hold_bars and c[prev] < ema20[prev]:
                    close_trade(i, o[i], "Trend close below EMA20")
                elif bars_held >= p.minimum_hold_bars and consecutive_macd_down >= p.trend_macd_down_bars:
                    close_trade(i, o[i], "Trend MACD weakening")
                elif bars_held >= p.max_hold_bars:
                    close_trade(i, o[i], "Maximum holding bars")

        if (
            qty == 0
            and not is_session_last_bar[i]
            and same_session_as_previous[i]
            and i >= cooldown_until
            and ml_probability[prev] >= p.ml_probability_min
        ):
            signal_type = "Pullback" if pullback[prev] else ("Trend breakout" if trend[prev] else "")
            if signal_type:
                entry = o[i] * (1 + slip)
                usable = max(cash * engine.allocation_pct - engine.commission, 0)
                qty = int(usable // entry)
                if qty > 0:
                    entry_cost = qty * entry + engine.commission
                    cash -= entry_cost; entry_i = i; entry_type = signal_type; highest = h[i]

        equity = cash + qty * c[i]
        peak = max(peak, equity)
        if peak > 0: max_dd = min(max_dd, equity / peak - 1)

    if qty > 0:
        close_trade(end - 1, c[end - 1], "End of fold")

    trades = len(trade_returns)
    stats = {
        "total_return_pct": (cash / engine.initial_cash - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "number_of_trades": trades,
        "win_rate_pct": wins / trades * 100 if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (5.0 if gross_profit else 0.0),
        "average_trade_return_pct": float(np.mean(trade_returns)) if trades else 0.0,
    }
    return stats, pd.DataFrame(records)


def session_boundaries(df: pd.DataFrame, timezone: str) -> tuple[np.ndarray, list[date]]:
    dates = np.asarray(df.index.tz_convert(timezone).date)
    sessions = list(np.unique(dates))
    starts = np.array([np.searchsorted(dates, d, side="left") for d in sessions] + [len(df)], dtype=int)
    return starts, sessions


def make_walk_forward_folds(df: pd.DataFrame, timezone: str, folds: int, min_train_fraction: float = 0.45):
    starts, sessions = session_boundaries(df, timezone)
    n = len(sessions)
    min_train = max(30, int(n * min_train_fraction))
    remaining = n - min_train
    if remaining < folds * 5:
        raise ValueError("Not enough sessions for requested walk-forward folds.")
    fold_size = remaining // folds
    result = []
    for k in range(folds):
        train_end_s = min_train + k * fold_size
        val_end_s = n if k == folds - 1 else train_end_s + fold_size
        result.append((0, starts[train_end_s], starts[train_end_s], starts[val_end_s], str(sessions[train_end_s])))
    return result


def suggest_params(trial: optuna.Trial) -> Params:
    rsi_min = trial.suggest_float("rsi_buy_min", 25, 48)
    rsi_max = trial.suggest_float("rsi_buy_max", max(40, rsi_min + 3), 65)
    trend_min = trial.suggest_float("trend_rsi_min", 48, 62)
    trend_max = trial.suggest_float("trend_rsi_max", max(66, trend_min + 8), 82)
    return Params(
        rsi_buy_min=rsi_min, rsi_buy_max=rsi_max,
        pullback_volume_min=trial.suggest_float("pullback_volume_min", 0.85, 1.8),
        require_pullback_macd_rising=trial.suggest_categorical("require_pullback_macd_rising", [True, False]),
        trend_rsi_min=trend_min, trend_rsi_max=trend_max,
        trend_volume_min=trial.suggest_float("trend_volume_min", 0.85, 1.5),
        breakout_lookback=trial.suggest_categorical("breakout_lookback", [6, 12, 18, 24]),
        ml_probability_min=trial.suggest_float("ml_probability_min", 0.50, 0.75),
        stop_loss_pct=trial.suggest_float("stop_loss_pct", 0.005, 0.02),
        pullback_take_profit_pct=trial.suggest_float("pullback_take_profit_pct", 0.008, 0.04),
        trend_trailing_stop_pct=trial.suggest_float("trend_trailing_stop_pct", 0.005, 0.025),
        cooldown_bars=trial.suggest_int("cooldown_bars", 2, 12),
        minimum_hold_bars=trial.suggest_int("minimum_hold_bars", 2, 8),
        max_hold_bars=trial.suggest_int("max_hold_bars", 12, 156),
        pullback_rsi_sell=trial.suggest_float("pullback_rsi_sell", 58, 80),
        trend_macd_down_bars=trial.suggest_int("trend_macd_down_bars", 2, 5),
        require_positive_long_trend=trial.suggest_categorical("require_positive_long_trend", [True, False]),
        block_high_volatility=trial.suggest_categorical("block_high_volatility", [True, False]),
        atr_percentile_limit=trial.suggest_float("atr_percentile_limit", 0.75, 0.98),
    )


def fold_score(s: dict[str, float], minimum_trades: int) -> float:
    if s["number_of_trades"] < minimum_trades:
        return -100 - 5 * (minimum_trades - s["number_of_trades"])
    # Penalize overtrading and drawdown while rewarding return and PF.
    overtrade = max(0.0, s["number_of_trades"] - 80) * 0.03
    return (
        s["total_return_pct"]
        - 1.5 * abs(s["max_drawdown_pct"])
        + 1.5 * min(s["profit_factor"], 4.0)
        + 0.015 * s["win_rate_pct"]
        - overtrade
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced walk-forward ML-filtered 5-minute optimizer")
    p.add_argument("--ticker", required=True); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--trials", type=int, default=300); p.add_argument("--folds", type=int, default=4)
    p.add_argument("--minimum-trades-per-fold", type=int, default=5); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--initial-cash", type=float, default=100000); p.add_argument("--commission", type=float, default=0)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--save-root", default=r"F:\pythonStock\SaveData")
    p.add_argument("--output-dir", default=None, help="Optional per-stock output directory override")
    p.add_argument("--cache-csv", default=None)
    p.add_argument("--session-start", default="08:00", help="New York session start, inclusive")
    p.add_argument("--session-end", default="16:00", help="New York session end, exclusive")
    p.add_argument("--force-fresh", action="store_true", help="Delete this ticker's cached data and outputs first")
    p.add_argument("--overwrite-config", action="store_true", help="Re-run optimization while keeping cached bars")
    p.add_argument("--ml-horizon-bars", type=int, default=24)
    p.add_argument("--ml-target-pct", type=float, default=0.012); p.add_argument("--ml-stop-pct", type=float, default=0.008)
    return p.parse_args()


def main() -> None:
    args = parse_args(); ticker = args.ticker.upper(); timezone = "America/New_York"
    save_root = Path(args.save_root)
    output = Path(args.output_dir) if args.output_dir else save_root / "stocks" / ticker
    cache = Path(args.cache_csv) if args.cache_csv else save_root / "data" / ticker / "5minute_raw_bars.csv"

    if args.force_fresh:
        if output.exists():
            shutil.rmtree(output)
        if cache.exists():
            cache.unlink()
    elif output.exists() and (output / "best_config.json").exists() and not args.overwrite_config:
        print(f"Configuration already exists for {ticker}: {output / 'best_config.json'}")
        print("Use --overwrite-config to re-optimize with cached bars or --force-fresh to rebuild everything.")
        return

    output.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print(f"Loading cached bars: {cache}")
        raw = pd.read_csv(cache, parse_dates=["timestamp"], index_col="timestamp")
        raw.index = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index.tz_convert("UTC")
    else:
        key = os.getenv("MASSIVE_API_KEY")
        if not key: raise SystemExit("Set MASSIVE_API_KEY or pass --cache-csv")
        warmup = (date.fromisoformat(args.start) - timedelta(days=90)).isoformat()
        raw = download_massive_bars(key, ticker, warmup, args.end, 5, "minute")
        cache.parent.mkdir(parents=True, exist_ok=True); raw.to_csv(cache, index_label="timestamp")

    raw_rows = len(raw)
    raw = filter_trading_session(
        raw, timezone=timezone, session_start=args.session_start, session_end=args.session_end
    )
    print(
        f"Session filter {args.session_start}-{args.session_end} ET: "
        f"kept {len(raw):,} of {raw_rows:,} raw bars"
    )
    if raw.empty:
        raise SystemExit("No bars remain after applying the trading-session filter.")

    print("Building day-reset indicators, time-of-day volume baselines, regimes, and session-bounded ML labels...")
    df = build_intraday_indicator_cache(raw, timezone)
    local = df.index.tz_convert(timezone)
    df = df.loc[(local >= pd.Timestamp(args.start, tz=timezone)) & (local < pd.Timestamp(args.end, tz=timezone) + pd.Timedelta(days=1))].copy()
    df["ml_label"] = make_path_label(df, args.ml_horizon_bars, args.ml_target_pct, args.ml_stop_pct, timezone)
    df = df.dropna(subset=["rsi", "relative_volume", "rsi_15m", "atr_pct", "ema_50_slope"])
    folds = make_walk_forward_folds(df, timezone, args.folds)
    print(f"Rows: {len(df):,}; walk-forward folds: {len(folds)}; ML model: {MODEL_NAME}")

    fold_probabilities = []
    for k, (tr_s, tr_e, va_s, va_e, va_date) in enumerate(folds, 1):
        probs, auc = fit_predict_probabilities(df, tr_s, tr_e, va_s, va_e, args.seed + k)
        full = np.full(len(df), 0.5); full[va_s:va_e] = probs; fold_probabilities.append(full)
        print(f"Fold {k}: train rows {tr_e-tr_s:,}, validation rows {va_e-va_s:,}, starts {va_date}, ML AUC={auc:.3f}")

    engine = Engine(args.initial_cash, args.commission, args.slippage_bps)
    history: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial); scores = []; metrics = []
        for k, (_, _, va_s, va_e, _) in enumerate(folds):
            stat, _ = backtest(df, va_s, va_e, params, engine, fold_probabilities[k])
            sc = fold_score(stat, args.minimum_trades_per_fold); scores.append(sc); metrics.append(stat)
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune(): raise optuna.TrialPruned()
        stability_penalty = float(np.std([m["total_return_pct"] for m in metrics]))
        value = float(np.mean(scores) - 0.5 * stability_penalty)
        history.append({"trial": trial.number, "score": value, **asdict(params), **{f"fold_{i+1}_return": m["total_return_pct"] for i,m in enumerate(metrics)}})
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed), pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    best = Params(**study.best_params)

    # Replay all walk-forward validation regions and save detailed trades.
    all_trades = []
    fold_summary = []
    for k, (_, _, va_s, va_e, va_date) in enumerate(folds):
        stat, trades = backtest(df, va_s, va_e, best, engine, fold_probabilities[k], collect=True)
        trades["fold"] = k + 1; all_trades.append(trades)
        fold_summary.append({"fold": k + 1, "validation_start": va_date, **stat})

    pd.DataFrame(history).to_csv(output / f"{ticker}_advanced_trials.csv", index=False)
    pd.DataFrame(fold_summary).to_csv(output / f"{ticker}_walk_forward_summary.csv", index=False)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades_df.to_csv(output / f"{ticker}_advanced_validation_trades.csv", index=False)
    (output / f"{ticker}_advanced_best.json").write_text(json.dumps({
        "ticker": ticker, "best_score": study.best_value, "parameters": asdict(best),
        "model": MODEL_NAME,
        "timezone": timezone,
        "session_start": args.session_start,
        "session_end": args.session_end,
        "session_end_exclusive": True,
        "indicator_reset_each_session": True,
        "relative_volume_method": "same New York clock slot over prior 20 sessions",
        "overnight_positions_allowed": False,
        "cooldown_resets_each_session": True,
        "holding_minutes_excludes_overnight": True,
        "folds": fold_summary,
    }, indent=2, default=float), encoding="utf-8")

    print("\nBEST ADVANCED CONFIGURATION")
    print("-" * 80)
    for key, value in asdict(best).items(): print(f"{key:32s}: {value}")
    print(f"Walk-forward score:              {study.best_value:.3f}")
    print(pd.DataFrame(fold_summary).to_string(index=False))
    print("\nSaved:")
    print(output / f"{ticker}_advanced_trials.csv")
    print(output / f"{ticker}_walk_forward_summary.csv")
    print(output / f"{ticker}_advanced_validation_trades.csv")
    print(output / f"{ticker}_advanced_best.json")


if __name__ == "__main__":
    main()
