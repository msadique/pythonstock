#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def load_stock_artifacts(ticker: str, output_dir: str = "advanced_optimization_output", require_approved: bool = True) -> tuple[dict[str, Any], Any, list[str]]:
    stock_dir = Path(output_dir) / ticker.upper()
    config_path = stock_dir / "best_config.json"
    model_path = stock_dir / "model.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration for {ticker}: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if require_approved and not config.get("approved", False):
        raise ValueError(f"{ticker} configuration is not approved.")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model for {ticker}: {model_path}")
    payload = joblib.load(model_path)
    return config, payload["model"], payload["features"]
