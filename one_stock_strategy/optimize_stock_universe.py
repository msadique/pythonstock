#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def load_tickers(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ticker = line.strip().upper()
        if ticker and not ticker.startswith("#"):
            tickers.append(ticker)
    return list(dict.fromkeys(tickers))


def main() -> None:
    p = argparse.ArgumentParser(description="Optimize one independent configuration/model per ticker")
    p.add_argument("--tickers-file", default="tickers.txt")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--trials-per-stock", type=int, default=250)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--minimum-trades-per-fold", type=int, default=5)
    p.add_argument("--save-root", default=r"F:\pythonStock\SaveData")
    p.add_argument("--session-start", default="08:00")
    p.add_argument("--session-end", default="16:00")
    p.add_argument("--force-fresh", action="store_true")
    p.add_argument("--overwrite-config", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    tickers = load_tickers(Path(args.tickers_file))
    if not tickers:
        raise SystemExit("No tickers found.")

    results = []
    for number, ticker in enumerate(tickers, 1):
        print("\n" + "=" * 88)
        print(f"[{number}/{len(tickers)}] Optimizing {ticker}")
        print("=" * 88)
        cmd = [
            sys.executable,
            "advanced_strategy_optimizer.py",
            "--ticker", ticker,
            "--start", args.start,
            "--end", args.end,
            "--trials", str(args.trials_per_stock),
            "--folds", str(args.folds),
            "--minimum-trades-per-fold", str(args.minimum_trades_per_fold),
            "--save-root", args.save_root,
            "--session-start", args.session_start,
            "--session-end", args.session_end,
        ]
        if args.force_fresh:
            cmd.append("--force-fresh")
        elif args.overwrite_config:
            cmd.append("--overwrite-config")

        completed = subprocess.run(cmd, check=False)
        success = completed.returncode == 0
        results.append({"ticker": ticker, "success": success, "return_code": completed.returncode})
        if not success and not args.continue_on_error:
            break

    summary_path = Path(args.save_root) / "universe_run_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ticker", "success", "return_code"])
        writer.writeheader(); writer.writerows(results)

    print(f"\nUniverse run summary: {summary_path}")
    failed = [row["ticker"] for row in results if not row["success"]]
    if failed:
        print("Failed tickers: " + ", ".join(failed))


if __name__ == "__main__":
    main()
