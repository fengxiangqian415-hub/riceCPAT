#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute regression metrics: R²(%), RMSE, MAE, rRMSE(%), MRE(%)

Usage:
    python3 calc_metrics.py runs/best_decoder/predictions.txt
    python3 calc_metrics.py predictions.csv --precision 4

Input file format (e.g., predictions.txt):
    Header CSV: true_value,predicted_value,absolute_error
    (only the true_value and predicted_value columns are used)
    Also accepts headerless two-column files (true, pred).

Metric definitions (matching the paper):
    R²    = 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²
    RMSE  = sqrt( mean((yᵢ-ŷᵢ)²) )
    MAE   = mean( |yᵢ-ŷᵢ| )
    rRMSE = RMSE / mean(y) × 100%          # relative RMSE
    MRE   = mean( |yᵢ-ŷᵢ| / yᵢ ) × 100%    # mean relative error
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple


def load_pairs(path: Path) -> Tuple[List[float], List[float]]:
    """Load (true, pred) pairs from a CSV, auto-detecting headers and column names."""
    y_true: List[float] = []
    y_pred: List[float] = []

    with path.open("r", newline="", encoding="utf-8") as handle:
        # Read the first line to detect the header
        first_line = handle.readline().strip()
        handle.seek(0)

        has_header = False
        col_true, col_pred = 0, 1

        if first_line:
            parts = [p.strip() for p in first_line.split(",")]
            low = [p.lower() for p in parts]
            # Header detection: rows whose first line contains true/pred/actual/predicted keywords
            keywords = ("true", "actual", "observed", "y_true",
                        "pred", "predict", "y_pred", "forecast", "estimate")
            if any(any(k in w for k in keywords) for w in low):
                has_header = True
                col_true = next(i for i, w in enumerate(low) if "true" in w or "actual" in w or "observed" in w)
                col_pred = next(i for i, w in enumerate(low) if "pred" in w or "forecast" in w)

        reader = csv.reader(handle)
        for idx, row in enumerate(reader, start=2):
            if not row:
                continue
            if has_header and idx == 2:  # skip the header row
                continue
            try:
                y_true.append(float(row[col_true]))
                y_pred.append(float(row[col_pred]))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"line {idx} is not numeric: {row!r} ({exc})") from exc

    if not y_true:
        raise ValueError("no data rows found.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch between true and pred: {len(y_true)} vs {len(y_pred)}")

    return y_true, y_pred


def compute_metrics(y_true: List[float], y_pred: List[float]) -> dict:
    """Compute all 5 metrics."""
    n = len(y_true)
    residuals = [t - p for t, p in zip(y_true, y_pred)]

    mse = sum(r * r for r in residuals) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(r) for r in residuals) / n

    mean_true = sum(y_true) / n
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum(r * r for r in residuals)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)

    # rRMSE: RMSE relative to the mean of the true values
    rrmse = rmse / mean_true * 100.0 if mean_true != 0 else float("nan")

    # MRE: mean of per-sample relative errors (samples with true == 0 are skipped)
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != 0]
    mre = sum(abs(t - p) / abs(t) for t, p in valid) / len(valid) * 100.0 if valid else float("nan")

    return {
        "n": n,
        "r2": r2,           # in [0, 1]
        "rmse": rmse,
        "mae": mae,
        "rrmse": rrmse,     # in %
        "mre": mre,         # in %
        "mean_true": mean_true,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute R²(%), RMSE, MAE, rRMSE(%), MRE(%)")
    parser.add_argument("file", type=Path, help="path to the predictions CSV")
    parser.add_argument("--precision", type=int, default=4, help="decimal places (default 4)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    y_true, y_pred = load_pairs(args.file)
    m = compute_metrics(y_true, y_pred)

    p = args.precision
    print(f"Samples n       : {m['n']}")
    print(f"R²    (%)       : {m['r2'] * 100:.{p}f}")
    print(f"RMSE            : {m['rmse']:.{p}f}")
    print(f"MAE             : {m['mae']:.{p}f}")
    print(f"rRMSE (%)       : {m['rrmse']:.{p}f}")
    print(f"MRE   (%)       : {m['mre']:.{p}f}")


if __name__ == "__main__":
    main()
