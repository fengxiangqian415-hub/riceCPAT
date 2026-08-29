# riceCPAT: Cross-Period Attention Transformer for Rice Yield Prediction

This repository contains the **model architecture and training pipeline** of the
**riceCPAT** model proposed in our manuscript (under review). The model combines
**phenology-guided adaptive segmentation** with **cross-period multi-head
attention** to predict rice yield from multi-temporal UAV-based vegetation-index
time series.

---

## 1. Overview

riceCPAT (`cpat/cpat.py`, class `RiceCPAT`) processes a variable-length
multi-temporal sequence of vegetation indices as follows:

1. **Input**: per-plot time series of 13 vegetation indices over T time steps
   (LCI, NDRE, NDVI, OSAVI, GNDVI, CI_green, CI_red_edge, WDRVI, EVI2, DVI,
   RVI, EVI, MCARI).
2. **Phenology-guided segmentation**: the sequence is split into K = 3
   phenological stages.
3. **Stage encoding**: each stage is encoded independently and compressed via
   masked mean pooling into a fixed-dimensional stage embedding.
4. **Cross-period attention**: N stacked cross-period multi-head attention
   layers model the dependencies between the phenological stages.
5. **Prediction**: the stage embeddings are globally pooled and mapped by a
   linear head to the predicted yield.

---

## 2. Repository Structure

```
ricecpat_open/
├── train.py                  # Full training pipeline (data loading, split, train, evaluate, save)
├── predict.py                # Inference with a trained weight → predictions.csv
├── calc_metrics.py           # Regression metrics: R², RMSE, MAE, rRMSE, MRE
├── requirements.txt
├── cpat/                     # riceCPAT model
│   ├── __init__.py
│   └── cpat.py               # class RiceCPAT (72,561 parameters)
└── tools/                    # Low-level Transformer building blocks
    ├── __init__.py
    ├── encoder.py                  # Transformer encoder layer
    ├── multi_head_attention.py     # Multi-head attention (full/chunk/window)
    ├── positionwise_feed_forward.py
    └── positional_encoding.py      # Sinusoidal positional encodings & masks
```

---

## 3. Installation

```bash
pip install -r requirements.txt
# torch>=2.0, numpy, pandas, scikit-learn, matplotlib, tqdm
```

Python 3.9+ recommended.

---

## 4. Input Data Format

Two CSV files are required as input (same format as used in the paper):

**Feature file** (e.g., `features.csv`):

```
Exp_ID,Date,LCI,NDRE,NDVI,OSAVI,GNDVI,CI_green,CI_red_edge,WDRVI,EVI2,DVI,RVI,EVI,MCARI
L1,2024/7/2,0.115241,0.096902,0.511925,0.245880,0.431566,1.604300,0.259938,-0.521723,0.127061,0.058834,3.166945,0.126175,0.110762
...
```

- Each row is one plot (`Exp_ID`) at one date (`Date`); a plot has multiple
  rows. Different plots may have different sequence lengths (handled by
  per-batch zero-padding with a padding mask).

**Target file** (e.g., `targets.csv`):

```
Exp_ID,Yield
L1,460.346337
...
```

---

## 5. Training

```bash
python3 train.py \
  --features_path dataset/features.csv \
  --targets_path dataset/targets.csv \
  --name my_run \
  --epochs 500 --batch_size 64 --lr 0.001 \
  --train_ratio 0.8 --seed 42
```

Key hyperparameters (defaults match the paper):

| Parameter | Default | Meaning |
|---|---|---|
| `--d_model` | 48 | model dimension |
| `--h` | 8 | attention heads |
| `--N` | 2 | cross-period attention layers |
| `--num_stages` | 3 | phenological stages (K) |
| `--dropout` | 0.2 | dropout |
| `--epochs` | 500 | training epochs |
| `--batch_size` | 64 | batch size |
| `--lr` | 0.001 | learning rate (Adam) |
| `--train_ratio` | 0.8 | train/test split ratio |
| `--seed` | 42 | random seed for data split & training |
| `--normalize` | `max` | feature normalization (min-max to [0,1]) |

Outputs are written to `runs/train/<name>/`: `best.pth` (highest test R²),
`last.pth`, `config.json`, `result.csv` (per-epoch metrics), `predictions.csv`,
`training_curves.png`.

> **Note on overfitting**: with small datasets (a few hundred plots), the best
> test R² is typically reached before the final epoch; use `best.pth` for
> evaluation/reporting. Reducing `--lr` to 5e-4–7e-4 usually improves the peak
> test R².

---

## 6. Prediction / Metrics

Run inference on the test split with a trained weight, then compute metrics:

```bash
python3 predict.py   --features_path dataset/features.csv   --targets_path dataset/targets.csv   --weight runs/train/my_run/best.pth   --config runs/train/my_run/config.json

python3 calc_metrics.py predictions.csv
# R² (%), RMSE, MAE, rRMSE (%), MRE (%)
```

- `--weight`: trained checkpoint (`best.pth` recommended).
- `--config`: optional; if provided, model hyperparameters are read from the
  training config; otherwise paper defaults are used.
- Output: `predictions.csv` (true_value, predicted_value), plus test-set
  R²/RMSE printed to the console.

## 7. License

Released for reproducibility; please cite the manuscript if you use this code.

---

## 8. Citation

```
[to be updated after acceptance]
```
