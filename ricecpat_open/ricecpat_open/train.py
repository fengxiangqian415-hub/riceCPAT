"""
Single-target time-series regression training script.
=====================================================
Goal: train the riceCPAT model to regress rice yield (Yield) from
multi-temporal remote-sensing time-series features.

Data:
  - Feature file: CSV with (Exp_ID, Date, <features>) columns, e.g.
    dataset/df_24.csv with 13 vegetation-index columns.
  - Target file: CSV with (Exp_ID, Yield), e.g. dataset/test_target.csv.

Features can be selected by category via --feature_categories:
  1 - elevation: DSM
  2 - base spectral bands: Red, Green, NIR, RedEdge
  3 - vegetation indices: LCI, NDRE, NDVI ... (13 columns)
  4 - texture features: Green_* / NIR_* / Red_* / RedEdge_* (32 columns)
  5 - meteorology: avetageT, sumT, sum_Solar ... (10 columns)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Add the project root to sys.path so that the cpat / tools packages import
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cpat import RiceCPAT


# ═══════════════════════════════════════════════════════════════════════════
# 1. Feature category definitions
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_CATEGORIES: dict[int, list[str]] = {
    1: ["DSM"],
    2: ["Red", "Green", "NIR", "RedEdge"],
    3: [
        "LCI", "NDRE", "NDVI", "OSAVI", "GNDVI",
        "CI_green", "CI_red_edge", "WDRVI", "EVI22",
        "DVI", "RVI", "EVI2", "MCARI",
    ],
    4: [
        "Green_asm", "Green_contrast", "Green_correlation", "Green_dissimilarity",
        "Green_entropy", "Green_homogeneity", "Green_mean", "Green_std",
        "NIR_asm", "NIR_contrast", "NIR_correlation", "NIR_dissimilarity",
        "NIR_entropy", "NIR_homogeneity", "NIR_mean", "NIR_std",
        "Red_asm", "Red_contrast", "Red_correlation", "Red_dissimilarity",
        "Red_entropy", "Red_homogeneity", "Red_mean", "Red_std",
        "RedEdge_asm", "RedEdge_contrast", "RedEdge_correlation", "RedEdge_dissimilarity",
        "RedEdge_entropy", "RedEdge_homogeneity", "RedEdge_mean", "RedEdge_std",
    ],
    5: [
        "avetageT", "avetageTMAX", "avetageTMIN", "sumT",
        "sum_Solar", "AVE_Solar", "sum_RAIN",
        "Ta_Avg", "Ta_Max", "Ta_Min",
    ],
}


def resolve_feature_columns(
    all_columns: list[str],
    categories: Optional[list[int]],
) -> list[str]:
    """Select feature columns from the CSV columns by category ids.

    categories=None means use all columns.
    """
    if categories is None:
        return all_columns
    selected: list[str] = []
    col_set = set(all_columns)
    for cid in categories:
        if cid not in FEATURE_CATEGORIES:
            raise ValueError(f"Unsupported feature category id {cid}; only 1~5 are supported.")
        for col in FEATURE_CATEGORIES[cid]:
            if col in col_set and col not in selected:
                selected.append(col)
    if not selected:
        raise ValueError("No columns matched the given categories; check the CSV columns and category settings.")
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ═══════════════════════════════════════════════════════════════════════════

class RiceYieldDataset(Dataset):
    """Rice yield time-series dataset (single target: Yield).

    Each sample is the full time-series feature sequence of one Exp_ID.
    Different Exp_IDs may have different sequence lengths; sequences are
    zero-padded to the same length. __getitem__ also returns a padding mask
    (real time steps = 1, padded steps = 0).

    stage_split="equal" keeps the original near-equal segmentation behavior;
    stage_split="phenology" reads only the PI and FL nodes from the node table
    and generates three stage labels: Date < PI, PI <= Date < FL, Date >= FL.
    """

    def __init__(
        self,
        features_path: str,
        targets_path: str,
        feature_columns: Optional[list[str]] = None,
        normalize: str = "max",
        stage_split: str = "equal",
        phenology_path: Optional[str] = None,
    ):
        super().__init__()
        self._normalize = normalize
        if stage_split not in ("equal", "phenology"):
            raise ValueError("stage_split only supports 'equal' or 'phenology'.")
        if stage_split == "phenology" and not phenology_path:
            raise ValueError("phenology mode requires phenology_path.")
        self.stage_split = stage_split
        self.phenology_path = phenology_path
        self._load(features_path, targets_path, feature_columns)

    @staticmethod
    def _load_phenology_nodes(phenology_path: str) -> pd.DataFrame:
        """Read only the PI / FL nodes used for the three-stage split."""
        try:
            nodes = pd.read_csv(
                phenology_path,
                encoding="utf-8-sig",
                usecols=["Exp_ID", "PI", "FL"],
                dtype={"Exp_ID": str},
            )
        except ValueError as exc:
            raise ValueError(
                f"The node table {phenology_path} must contain Exp_ID, PI, FL columns."
            ) from exc

        if nodes["Exp_ID"].duplicated().any():
            duplicated = nodes.loc[nodes["Exp_ID"].duplicated(), "Exp_ID"].tolist()
            raise ValueError(f"Duplicated Exp_ID in the node table: {duplicated[:10]}")

        for column in ("PI", "FL"):
            raw = nodes[column].copy()
            nodes[column] = pd.to_datetime(raw, errors="coerce")
            if nodes[column].isna().any():
                bad_ids = nodes.loc[nodes[column].isna(), "Exp_ID"].tolist()
                raise ValueError(
                    f"Column {column} of the node table contains missing/unparsable dates: {bad_ids[:10]}"
                )

        invalid_order = nodes["PI"] >= nodes["FL"]
        if invalid_order.any():
            bad_ids = nodes.loc[invalid_order, "Exp_ID"].tolist()
            raise ValueError(f"Node dates must satisfy PI < FL: {bad_ids[:10]}")

        return nodes.set_index("Exp_ID")

    def _load(self, features_path, targets_path, feature_columns):
        df_feat = pd.read_csv(features_path)
        df_tgt  = pd.read_csv(targets_path)

        # Keep only the columns that actually exist in the CSV
        all_cols = [c for c in df_feat.columns if c not in ("Exp_ID", "Date")]
        if feature_columns is None:
            feature_columns = all_cols
        feature_columns = [c for c in feature_columns if c in set(all_cols)]
        if not feature_columns:
            raise ValueError("No feature columns matched; check the feature_columns argument.")
        self.feature_names: list[str] = feature_columns

        phenology_nodes = None
        if self.stage_split == "phenology":
            # The parsed dates are only used for phenology segmentation. The
            # equal mode keeps the original sorting behavior so that existing
            # commands and the default inference of legacy weights are unchanged.
            parsed_dates = pd.to_datetime(df_feat["Date"], errors="coerce")
            if parsed_dates.isna().any():
                bad_ids = df_feat.loc[parsed_dates.isna(), "Exp_ID"].astype(str).tolist()
                raise ValueError(f"Unparsable dates in the feature table: {bad_ids[:10]}")
            df_feat = df_feat.copy()
            df_feat["_parsed_date"] = parsed_dates
            df_feat["Exp_ID"] = df_feat["Exp_ID"].astype(str)
            df_tgt = df_tgt.copy()
            df_tgt["Exp_ID"] = df_tgt["Exp_ID"].astype(str)
            phenology_nodes = self._load_phenology_nodes(self.phenology_path)

        exp_ids = df_feat["Exp_ID"].unique()
        sequences, targets, seq_lengths = [], [], []
        included_exp_ids, stage_id_sequences = [], []

        for eid in exp_ids:
            rows = df_feat[df_feat["Exp_ID"] == eid]
            if self.stage_split == "phenology":
                rows = rows.sort_values("_parsed_date")
            else:
                rows = rows.sort_values("Date")
            tgt_row = df_tgt[df_tgt["Exp_ID"] == eid]
            if tgt_row.empty:
                continue  # skip Exp_IDs without a target

            if self.stage_split == "phenology":
                eid_key = str(eid)
                if eid_key not in phenology_nodes.index:
                    raise ValueError(f"Missing PI/FL nodes for sample {eid_key}.")
                pi_date = phenology_nodes.at[eid_key, "PI"]
                fl_date = phenology_nodes.at[eid_key, "FL"]
                dates = rows["_parsed_date"]
                # Three stages: TP–PI, PI–FL, FL–MS. Only PI/FL are split nodes.
                stage_ids = np.select(
                    [dates < pi_date, dates < fl_date],
                    [0, 1],
                    default=2,
                ).astype(np.int64)
                stage_id_sequences.append(stage_ids)

            sequences.append(rows[feature_columns].values.astype(np.float32))
            targets.append(float(tgt_row["Yield"].values[0]))
            seq_lengths.append(len(rows))
            included_exp_ids.append(eid)

        # Zero-pad to the longest sequence
        max_len = max(len(s) for s in sequences)
        n, f    = len(sequences), len(feature_columns)
        padded  = np.zeros((n, max_len, f), dtype=np.float32)
        for i, seq in enumerate(sequences):
            padded[i, : len(seq)] = seq

        self._x            = padded
        self._y            = np.array(targets, dtype=np.float32).reshape(-1, 1)
        self.exp_ids       = np.asarray(included_exp_ids)
        self.seq_lengths   = seq_lengths

        self._stage_ids = None
        if self.stage_split == "phenology":
            padded_stage_ids = np.full((n, max_len), -1, dtype=np.int64)
            for i, stage_ids in enumerate(stage_id_sequences):
                padded_stage_ids[i, : len(stage_ids)] = stage_ids
            self._stage_ids = torch.from_numpy(padded_stage_ids)
            self.stage_counts = {
                stage: int((padded_stage_ids == stage).sum())
                for stage in range(3)
            }

        # Normalization
        if self._normalize == "max":
            x_min, x_max = self._x.min(axis=(0, 1)), self._x.max(axis=(0, 1))
            self._x      = (self._x - x_min) / (x_max - x_min + np.finfo(float).eps)
            self._y_min, self._y_max = float(self._y.min()), float(self._y.max())
            self._y = (self._y - self._y_min) / (self._y_max - self._y_min + np.finfo(float).eps)
        elif self._normalize == "mean":
            x_mean, x_std = self._x.mean(axis=(0, 1)), self._x.std(axis=(0, 1))
            self._x       = (self._x - x_mean) / (x_std + np.finfo(float).eps)
            self._y_mean, self._y_std = float(self._y.mean()), float(self._y.std())
            self._y = (self._y - self._y_mean) / (self._y_std + np.finfo(float).eps)

        # The float64 epsilon in the normalization can upcast NumPy arrays to
        # float64; explicitly restore float32 to match the PyTorch weights.
        self._x = torch.from_numpy(self._x.astype(np.float32, copy=False))
        self._y = torch.from_numpy(self._y.astype(np.float32, copy=False))

    def rescale(self, y: np.ndarray) -> np.ndarray:
        """Rescale normalized predictions back to the original yield scale."""
        if self._normalize == "max":
            return y * (self._y_max - self._y_min + np.finfo(float).eps) + self._y_min
        if self._normalize == "mean":
            return y * (self._y_std + np.finfo(float).eps) + self._y_mean
        return y

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, idx):
        x    = self._x[idx]           # (T, F)
        y    = self._y[idx]           # (1,)
        mask = torch.zeros(x.shape[0])
        mask[: self.seq_lengths[idx]] = 1.0   # mark real time steps as 1
        if self._stage_ids is not None:
            return x, y, mask, self._stage_ids[idx]
        return x, y, mask


# ═══════════════════════════════════════════════════════════════════════════
# 3. Model factory
# ═══════════════════════════════════════════════════════════════════════════

def build_model(args, d_input: int) -> nn.Module:
    """Build the model from the command-line arguments.

    The open-source version only supports the riceCPAT model
    (--model_type staged_encoder).
    """
    if args.model_type != "staged_encoder":
        raise ValueError("The open-source riceCPAT release only supports staged_encoder.")
    return RiceCPAT(
        d_input=d_input, d_model=args.d_model, d_output=1,
        q=args.q, v=args.v, h=args.h, N=args.N,
        attention_size=None, dropout=args.dropout,
        chunk_mode=None, pe="original", pooling=args.pooling,
        num_stages=args.num_stages,
        stage_embed_dim=args.stage_embed_dim,
        stage_pooling=args.stage_pooling,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Training / evaluation utilities
# ═══════════════════════════════════════════════════════════════════════════

def prepare_batch(batch, device):
    """Handle both the (x, y, mask) and the phenology (x, y, mask, stage_ids) formats."""
    if len(batch) == 3:
        x, y, mask = batch
        stage_ids = None
    elif len(batch) == 4:
        x, y, mask, stage_ids = batch
    else:
        raise ValueError(f"Unsupported batch structure with {len(batch)} elements.")

    x, y, mask = x.to(device), y.to(device), mask.to(device)
    if stage_ids is not None:
        stage_ids = stage_ids.to(device)
    return x, y, mask, stage_ids


def model_forward(model, x, mask, stage_ids=None):
    """Pass the stage labels to the model only in the phenology mode."""
    if stage_ids is None:
        return model(x, mask)
    return model(x, mask, stage_ids=stage_ids)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total = 0.0
    for batch in loader:
        x, y, mask, stage_ids = prepare_batch(batch, device)
        optimizer.zero_grad()
        loss = criterion(model_forward(model, x, mask, stage_ids), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate the model; returns (avg_loss, preds_tensor, trues_tensor).

    preds / trues are torch.Tensors in the normalized scale.
    """
    model.eval()
    total, preds_list, trues_list = 0.0, [], []
    for batch in loader:
        x, y, mask, stage_ids = prepare_batch(batch, device)
        pred = model_forward(model, x, mask, stage_ids)
        total += criterion(pred, y).item()
        preds_list.append(pred.cpu())
        trues_list.append(y.cpu())
    return total / len(loader), torch.cat(preds_list), torch.cat(trues_list)


def compute_metrics(trues: np.ndarray, preds: np.ndarray) -> dict:
    """Compute MSE / RMSE / MAE / R² (inputs are numpy arrays in the original scale)."""
    mse = float(mean_squared_error(trues, preds))
    return {
        "mse":  mse,
        "rmse": float(np.sqrt(mse)),
        "mae":  float(np.mean(np.abs(trues - preds))),
        "r2":   float(r2_score(trues, preds)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Saving results
# ═══════════════════════════════════════════════════════════════════════════

def save_curves(out_dir: Path, train_losses, test_losses, train_r2s, test_r2s):
    """Plot and save the training curves (Loss and R²)."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, tr, te, ylabel, title in [
        (ax1, train_losses, test_losses, "Loss (MSE)", "Loss Curve"),
        (ax2, train_r2s,   test_r2s,   "R²",         "R² Curve"),
    ]:
        ax.plot(epochs, tr, label="Train")
        ax.plot(epochs, te, label="Test")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def save_predictions(out_dir: Path, dataset, train_sub, test_sub, model, device):
    """Save per-sample predictions for the train and test splits to predictions.csv.

    Columns: exp_id, split, true_yield, pred_yield, abs_error
    """
    rows = []
    model.eval()
    for split_name, subset in [("train", train_sub), ("test", test_sub)]:
        loader  = DataLoader(subset, batch_size=256, shuffle=False)
        indices = subset.indices if hasattr(subset, "indices") else list(range(len(subset)))
        ps, ts  = [], []
        with torch.no_grad():
            for batch in loader:
                x, y, mask, stage_ids = prepare_batch(batch, device)
                ps.append(model_forward(model, x, mask, stage_ids).cpu())
                ts.append(y.cpu())
        preds = dataset.rescale(torch.cat(ps).numpy())
        trues = dataset.rescale(torch.cat(ts).numpy())
        for i, orig_idx in enumerate(indices):
            eid = dataset.exp_ids[orig_idx] if orig_idx < len(dataset.exp_ids) else orig_idx
            rows.append(dict(
                exp_id=eid, split=split_name,
                true_yield=round(float(trues[i, 0]), 4),
                pred_yield=round(float(preds[i, 0]), 4),
                abs_error =round(abs(float(trues[i, 0]) - float(preds[i, 0])), 4),
            ))
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)


def save_config(
    out_dir: Path, args, dataset, d_input: int,
    best_r2: float, best_epoch: int,
    final_train: dict, final_test: dict,
):
    """Save all hyperparameters and results of this run to config.json."""
    cfg = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "features_path":     args.features_path,
            "targets_path":      args.targets_path,
            "normalize":         args.normalize,
            "stage_split":       getattr(args, "stage_split", "equal"),
            "phenology_path":    (
                getattr(args, "phenology_path", None)
                if getattr(args, "stage_split", "equal") == "phenology"
                else None
            ),
            "phenology_nodes":   (
                ["PI", "FL"]
                if getattr(args, "stage_split", "equal") == "phenology"
                else None
            ),
            "feature_categories": args.feature_categories,
            "feature_count":     d_input,
            "feature_names":     dataset.feature_names,
            "num_samples":       len(dataset),
        },
        "model": {
            "type":    args.model_type,
            "d_model": args.d_model, "d_input": d_input, "d_output": 1,
            "q": args.q, "v": args.v, "h": args.h, "N": args.N,
            "dropout": args.dropout, "pooling": args.pooling,
            "num_stages":     getattr(args, "num_stages", None),
            "stage_embed_dim": getattr(args, "stage_embed_dim", None),
            "stage_pooling":   getattr(args, "stage_pooling", None),
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "optimizer": "Adam", "loss": "MSE",
            "train_ratio": args.train_ratio, "seed": args.seed,
        },
        "results": {
            "best_test_r2": best_r2,
            "best_epoch":   best_epoch,
            "final_train":  final_train,
            "final_test":   final_test,
        },
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    """Full training pipeline: load data -> build model -> train -> save."""

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stage_split = getattr(args, "stage_split", "equal")
    phenology_path = getattr(args, "phenology_path", "dataset/phenology_nodes.csv")
    if stage_split == "phenology":
        if args.model_type != "staged_encoder":
            raise ValueError("phenology segmentation currently only supports --model_type staged_encoder.")
        if args.num_stages != 3:
            raise ValueError("The PI/FL nodes fix 3 phenological stages; please set --num_stages 3.")

    # ── Output directory ──────────────────────────────────────────────────────
    run_name = args.name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Append a tag of the feature categories to the run name for clarity
    if args.feature_categories:
        cat_tag = "feat" + "".join(str(c) for c in sorted(args.feature_categories))
        run_name = f"{run_name}_{cat_tag}"
    if stage_split == "phenology":
        run_name = f"{run_name}_phenology"
    out_dir = Path("runs/train") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Read one row to determine all feature columns, then filter by category
    all_cols = [
        c for c in pd.read_csv(args.features_path, nrows=0).columns
        if c not in ("Exp_ID", "Date")
    ]
    feat_cols = resolve_feature_columns(all_cols, args.feature_categories)
    print(f"Feature count: {len(feat_cols)}  categories: {args.feature_categories}")
    print(f"Feature columns: {feat_cols}")

    dataset = RiceYieldDataset(
        features_path=args.features_path,
        targets_path=args.targets_path,
        feature_columns=feat_cols,
        normalize=args.normalize,
        stage_split=stage_split,
        phenology_path=phenology_path if stage_split == "phenology" else None,
    )
    print(f"Total samples: {len(dataset)}  sequence shape: {dataset._x.shape}")
    if stage_split == "phenology":
        print(
            "Phenology segmentation: PI / FL nodes -> "
            "TP-PI, PI-FL, FL-MS stages  "
            f"steps: {dataset.stage_counts}"
        )

    # Split into train / test with a fixed seed for reproducibility
    train_size = int(len(dataset) * args.train_ratio)
    test_size  = len(dataset) - train_size
    train_sub, test_sub = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    dl_train = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True)
    dl_test  = DataLoader(test_sub,  batch_size=args.batch_size, shuffle=False)
    print(f"Train: {train_size}  Test: {test_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    d_input = len(feat_cols)
    model   = build_model(args, d_input).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type}  trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    # Record the loss and R² of every epoch (original scale)
    train_losses, test_losses = [], []
    train_r2s,    test_r2s    = [], []

    # result.csv records per-epoch metrics for easy inspection
    result_csv = out_dir / "result.csv"
    with result_csv.open("w") as f:
        f.write("epoch,train_loss,test_loss,train_r2,test_r2,"
                "train_rmse,train_mae,test_rmse,test_mae\n")

    best_r2, best_epoch = -float("inf"), 0

    with tqdm(total=args.epochs, desc="Training") as pbar:
        for epoch in range(1, args.epochs + 1):

            # Training
            tr_loss = train_one_epoch(model, dl_train, optimizer, criterion, device)

            # Evaluation (original-scale metrics)
            _, tr_preds, tr_trues = evaluate(model, dl_train, criterion, device)
            te_loss, te_preds, te_trues = evaluate(model, dl_test,  criterion, device)

            tr_met = compute_metrics(
                dataset.rescale(tr_trues.numpy()),
                dataset.rescale(tr_preds.numpy()),
            )
            te_met = compute_metrics(
                dataset.rescale(te_trues.numpy()),
                dataset.rescale(te_preds.numpy()),
            )

            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            train_r2s.append(tr_met["r2"])
            test_r2s.append(te_met["r2"])

            # Write to result.csv
            with result_csv.open("a") as f:
                f.write(
                    f"{epoch},{tr_loss:.6f},{te_loss:.6f},"
                    f"{tr_met['r2']:.4f},{te_met['r2']:.4f},"
                    f"{tr_met['rmse']:.4f},{tr_met['mae']:.4f},"
                    f"{te_met['rmse']:.4f},{te_met['mae']:.4f}\n"
                )

            # Save the best model (by test R²)
            if te_met["r2"] > best_r2:
                best_r2, best_epoch = te_met["r2"], epoch
                torch.save(model.state_dict(), out_dir / "best.pth")

            pbar.set_postfix(
                tr_loss=f"{tr_loss:.4f}",
                te_r2=f"{te_met['r2']:.4f}",
                best=f"{best_r2:.4f}@{best_epoch}",
            )
            pbar.update()

    # Save the weights of the last epoch
    torch.save(model.state_dict(), out_dir / "last.pth")

    # ── Save results ──────────────────────────────────────────────────────────
    save_curves(out_dir, train_losses, test_losses, train_r2s, test_r2s)
    save_predictions(out_dir, dataset, train_sub, test_sub, model, device)
    save_config(
        out_dir, args, dataset, d_input,
        best_r2, best_epoch,
        final_train=tr_met, final_test=te_met,
    )

    print(f"\nTraining complete! Best test R²: {best_r2:.4f} (epoch {best_epoch})")
    print(f"All results saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. Command-line arguments
# ═══════════════════════════════════════════════════════════════════════════

def parse_feature_categories(value: str) -> Optional[List[int]]:
    """Parse a '1,3,5' string into a list of ints; 'all'/'none' return None."""
    v = value.strip().lower()
    if v in ("all", "none", "", "0"):
        return None
    parts = [p for p in v.replace(" ", "").split(",") if p]
    try:
        cats = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--feature_categories must be comma-separated numbers, e.g. '1,3'"
        )
    invalid = [c for c in cats if c not in range(1, 6)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid categories: {invalid}; only 1~5 are supported.")
    # Deduplicate while preserving order
    seen, ordered = set(), []
    for c in cats:
        if c not in seen:
            seen.add(c); ordered.append(c)
    return ordered


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rice yield single-target Transformer training script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--features_path", default="df_all_Two.csv",
                   help="path to the feature CSV")
    p.add_argument("--targets_path",  default="dataset/target_Two.csv",
                   help="path to the target CSV (must contain a Yield column)")
    p.add_argument("--normalize",     default="max", choices=["max", "mean", "none"],
                   help="feature/target normalization")
    p.add_argument("--feature_categories", type=parse_feature_categories, default=None,
                   help="feature categories to use, e.g. '3' / '1,3,5' / 'all'")
    p.add_argument(
        "--stage_split", default="equal", choices=["equal", "phenology"],
        help=(
            "stage_split for staged_encoder: 'equal' keeps the original "
            "near-equal segmentation; 'phenology' uses the PI/FL nodes from "
            "the node table to define the three phenological stages"
        ),
    )
    p.add_argument(
        "--phenology_path", default="dataset/phenology_nodes.csv",
        help="phenology node CSV (phenology mode reads only Exp_ID, PI, FL)",
    )
    p.add_argument("--train_ratio",   type=float, default=0.8,
                   help="train ratio")
    p.add_argument("--seed",          type=int,   default=42,
                   help="random seed")

    # Experiment
    p.add_argument("--name",       default=None, help="experiment name (default: timestamp)")
    p.add_argument("--epochs",     type=int,   default=500)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-3)

    # Model
    p.add_argument("--model_type", default="staged_encoder",
                   choices=["staged_encoder"])
    p.add_argument("--d_model",  type=int,   default=48)
    p.add_argument("--q",        type=int,   default=12)
    p.add_argument("--v",        type=int,   default=12)
    p.add_argument("--h",        type=int,   default=8)
    p.add_argument("--N",        type=int,   default=2)
    p.add_argument("--dropout",  type=float, default=0.2)
    p.add_argument("--pooling",  default="mean",
                   choices=["mean", "max", "last", "cls"])

    # staged_encoder specific
    p.add_argument("--num_stages",     type=int,   default=3)
    p.add_argument("--stage_embed_dim",type=int,   default=None)
    p.add_argument("--stage_pooling",  default="mean",
                   choices=["mean", "max", "last", "first"])

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
