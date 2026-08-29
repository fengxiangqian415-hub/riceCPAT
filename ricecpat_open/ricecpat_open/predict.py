#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run inference on the test split with a trained riceCPAT weight,
save predictions and print metrics.

Usage:
    python3 predict.py --features_path dataset/features.csv \
                       --targets_path dataset/targets.csv \
                       --weight runs/train/my_run/best.pth \
                       --seed 42

    If a config.json was saved during training, use --config to
    automatically load the model hyperparameters:
    python3 predict.py ... --weight runs/train/my_run/best.pth \
                       --config runs/train/my_run/config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from train import RiceYieldDataset, build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="riceCPAT inference script")
    p.add_argument("--features_path", required=True)
    p.add_argument("--targets_path", required=True)
    p.add_argument("--weight", required=True, help="checkpoint file (best.pth / last.pth)")
    p.add_argument("--config", default=None, help="optional: training config.json (model hyperparameters)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--normalize", default="max", choices=["max", "mean", "none"])
    p.add_argument("--output", default="predictions.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Model hyperparameters: read from config.json if available, otherwise use paper defaults
    if args.config and Path(args.config).exists():
        cfg = json.load(open(args.config))
        m = cfg.get("model", {})
        model_args = argparse.Namespace(
            model_type="staged_encoder",
            d_model=m.get("d_model", 48),
            q=m.get("q", 12), v=m.get("v", 12), h=m.get("h", 8), N=m.get("N", 2),
            dropout=m.get("dropout", 0.2), pooling=m.get("pooling", "mean"),
            num_stages=m.get("num_stages", 3),
            stage_embed_dim=m.get("stage_embed_dim"),
            stage_pooling=m.get("stage_pooling", "mean"),
        )
        d_input = m.get("d_input", 13)
    else:
        model_args = argparse.Namespace(
            model_type="staged_encoder",
            d_model=48, q=12, v=12, h=8, N=2, dropout=0.2, pooling="mean",
            num_stages=3, stage_embed_dim=None, stage_pooling="mean",
        )
        d_input = 13

    # Data loading and split (identical to training)
    ds = RiceYieldDataset(args.features_path, args.targets_path, normalize=args.normalize)
    total = len(ds)
    tr, te = random_split(
        ds,
        [int(args.train_ratio * total), total - int(args.train_ratio * total)],
        torch.Generator().manual_seed(args.seed),
    )
    print(f"Samples: {len(ds)}  Test split: {len(te)}")

    # Build the model and load weights
    net = build_model(model_args, ds._x.shape[2])
    net.load_state_dict(torch.load(args.weight, map_location="cpu"))
    net.eval()

    # Inference on the test split
    preds, trues = [], []
    with torch.no_grad():
        for x, y, mask in DataLoader(te, batch_size=64):
            preds.append(net(x, mask).cpu())
            trues.append(y)
    t = ds.rescale(torch.cat(trues).numpy())[:, 0]
    p = ds.rescale(torch.cat(preds).numpy())[:, 0]

    # Save predictions
    np.savetxt(args.output, np.column_stack([t, p]), delimiter=",",
               header="true_value,predicted_value", comments="")
    print(f"Predictions saved to: {args.output}")

    # Print metrics
    rmse = np.sqrt(np.mean((t - p) ** 2))
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    print(f"Test: R² = {r2 * 100:.4f}%   RMSE = {rmse:.4f}")


if __name__ == "__main__":
    main()
