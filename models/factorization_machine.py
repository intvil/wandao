"""
Factorization machine for lineup-based win rate prediction.

Features are built by multiplying each hero's lineup sign (-1/0/1) by its
position probability vector (Position1..Position5), yielding NUM_HEROS * 5 features.
"""

import itertools
import multiprocessing as mp
import os
from typing import List, Sequence, Tuple, Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from wandao.config import (
    DATA_PATH,
    FM_MODEL_PATH,
    POSITION_PROBS_PATH,
    POSITION_PROB_SPARSE_THRESHOLD,
    device,
    FM_LATENT_DIM,
    FM_BATCH_SIZE,
    FM_EPOCHS,
    FM_LR,
    FM_L2_LAMBDA0,
    FM_EARLY_STOP_PATIENCE,
    FM_EARLY_STOP_MIN_DELTA,
    FM_GRID_LATENT_DIMS,
    FM_GRID_LRS,
    FM_GRID_L2_LAMBDAS,
    FM_GRID_NUM_WORKERS,
)

ROLE_COLUMNS = ["Position1", "Position2", "Position3", "Position4", "Position5"]
DEFAULT_POSITION_PROBS = POSITION_PROBS_PATH
DEFAULT_LATENT_DIM = FM_LATENT_DIM


def _normalize_hero_columns(hero_cols: Sequence) -> List[int]:
    hero_ids = []
    for col in hero_cols:
        if isinstance(col, (int, np.integer)):
            hero_ids.append(int(col))
        elif isinstance(col, str) and col.isdigit():
            hero_ids.append(int(col))
        else:
            raise ValueError(f"Unexpected hero column: {col!r}")
    if len(set(hero_ids)) != len(hero_ids):
        raise ValueError("Duplicate hero IDs in lineup columns.")
    return hero_ids


def load_position_probs(path: str = DEFAULT_POSITION_PROBS) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hero position probs not found at {path}")
    df = pd.read_csv(path)
    if "Hero_ID" not in df.columns:
        raise ValueError("hero_position_probs.csv missing Hero_ID column")
    missing = [c for c in ROLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"hero_position_probs.csv missing columns: {missing}")
    if df["Hero_ID"].duplicated().any():
        raise ValueError("Duplicate Hero_ID values in hero_position_probs.csv")
    df = df[["Hero_ID"] + ROLE_COLUMNS].copy()
    df["Hero_ID"] = df["Hero_ID"].astype(int)
    threshold = float(POSITION_PROB_SPARSE_THRESHOLD)
    if threshold > 0:
        role_vals = df[ROLE_COLUMNS].to_numpy(dtype=np.float32)
        mask = role_vals >= threshold
        sparse_vals = np.where(mask, role_vals, 0.0)
        row_sums = sparse_vals.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze() == 0.0
        if np.any(zero_rows):
            sparse_vals[zero_rows] = role_vals[zero_rows]
            row_sums = sparse_vals.sum(axis=1, keepdims=True)
        sparse_vals = sparse_vals / row_sums
        df[ROLE_COLUMNS] = sparse_vals
    return df.set_index("Hero_ID")


def build_role_matrix(
    hero_ids: Sequence[int], pos_df: pd.DataFrame
) -> Tuple[np.ndarray, List[int]]:
    missing = [hid for hid in hero_ids if hid not in pos_df.index]
    if missing:
        preview = ", ".join(str(h) for h in missing[:10])
        print(
            "Warning: dropping heroes missing position probabilities "
            f"({len(missing)}): {preview}"
        )
    filtered_ids = [hid for hid in hero_ids if hid in pos_df.index]
    role_probs = pos_df.loc[filtered_ids, ROLE_COLUMNS].to_numpy(dtype=np.float32)
    return role_probs, filtered_ids


def load_fm_data(
    fea_path: str = DATA_PATH, position_probs_path: str = DEFAULT_POSITION_PROBS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    if not os.path.exists(fea_path):
        raise FileNotFoundError(f"Encoded match data not found at {fea_path}")
    df = pd.read_feather(fea_path)
    if "radiant_win" not in df.columns:
        raise ValueError("Dataset must contain 'radiant_win' column")
    hero_cols = [c for c in df.columns if c != "radiant_win"]
    hero_ids = _normalize_hero_columns(hero_cols)
    pos_df = load_position_probs(position_probs_path)
    role_probs, filtered_ids = build_role_matrix(hero_ids, pos_df)
    keep_cols = []
    for col, hid in zip(hero_cols, hero_ids):
        if hid in set(filtered_ids):
            keep_cols.append(col)
    lineups = df[keep_cols].to_numpy(dtype=np.float32)
    targets = df["radiant_win"].to_numpy(dtype=np.float32)
    return lineups, targets, role_probs, filtered_ids


def build_fm_features(lineups: np.ndarray, role_probs: np.ndarray) -> np.ndarray:
    lineups = np.asarray(lineups, dtype=np.float32)
    role_probs = np.asarray(role_probs, dtype=np.float32)
    if lineups.ndim != 2:
        raise ValueError("lineups must be a 2D array (n_samples, n_heroes)")
    if role_probs.ndim != 2 or role_probs.shape[1] != len(ROLE_COLUMNS):
        raise ValueError("role_probs must be (n_heroes, 5)")
    if lineups.shape[1] != role_probs.shape[0]:
        raise ValueError("lineups and role_probs hero dimensions do not match")
    return (lineups[:, :, None] * role_probs[None, :, :]).reshape(lineups.shape[0], -1)


class LineupFMDataset(Dataset):
    def __init__(self, lineups: np.ndarray, targets: np.ndarray):
        if lineups.shape[0] != targets.shape[0]:
            raise ValueError("lineups and targets must have the same length")
        self.lineups = torch.tensor(lineups, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        return self.lineups[idx], self.targets[idx]


def make_fm_collate(role_probs: np.ndarray):
    role_probs_t = torch.tensor(role_probs, dtype=torch.float32)

    def collate(batch):
        lineups, targets = zip(*batch)
        lineup_batch = torch.stack(lineups, dim=0)
        target_batch = torch.stack(targets, dim=0)
        feats = (lineup_batch.unsqueeze(-1) * role_probs_t.unsqueeze(0)).reshape(
            lineup_batch.size(0), -1
        )
        return feats, target_batch

    return collate


class FactorizationMachine(nn.Module):
    def __init__(self, n_features: int, latent_dim: int = DEFAULT_LATENT_DIM):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.linear = nn.Parameter(torch.zeros(n_features))
        self.factors = nn.Parameter(torch.randn(n_features, latent_dim) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.bias + torch.matmul(x, self.linear)
        xv = torch.matmul(x, self.factors)
        x2v2 = torch.matmul(x * x, self.factors * self.factors)
        interactions = 0.5 * torch.sum(xv * xv - x2v2, dim=1)
        return linear + interactions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def _evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for feats, targets in loader:
            feats = feats.to(device)
            targets = targets.to(device)
            logits = model(feats)
            loss = loss_fn(logits, targets)
            total_loss += loss.item() * feats.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == targets).sum().item()
            total += feats.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train_fm(
    fea_path: str = DATA_PATH,
    position_probs_path: str = DEFAULT_POSITION_PROBS,
    latent_dim: int = DEFAULT_LATENT_DIM,
    batch_size: int = FM_BATCH_SIZE,
    epochs: int = FM_EPOCHS,
    lr: float = FM_LR,
    l2_lambda0: float = FM_L2_LAMBDA0,
    early_stop_patience: int = FM_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = FM_EARLY_STOP_MIN_DELTA,
    model_path: Optional[str] = FM_MODEL_PATH,
    return_metrics: bool = False,
) -> Union[nn.Module, Tuple[nn.Module, float, float]]:
    lineups, targets, role_probs, hero_ids = load_fm_data(
        fea_path=fea_path, position_probs_path=position_probs_path
    )
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        lineups, targets, test_size=0.2, random_state=42, stratify=targets
    )
    train_ds = LineupFMDataset(X_train, y_train)
    val_ds = LineupFMDataset(X_val, y_val)

    collate = make_fm_collate(role_probs)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate
    )

    n_features = role_probs.shape[0] * role_probs.shape[1]
    hero_counts = np.sum(np.abs(lineups), axis=0)
    n_max = float(np.max(hero_counts)) if hero_counts.size else 0.0
    if l2_lambda0 < 0:
        raise ValueError("l2_lambda0 must be non-negative")
    if n_max <= 0.0:
        lambda_h = np.zeros_like(hero_counts, dtype=np.float32)
    else:
        lambda_h = l2_lambda0 * np.sqrt(n_max / (hero_counts + 1.0))
    lambda_feat = np.repeat(lambda_h, role_probs.shape[1]).astype(np.float32)
    lambda_feat_t = torch.tensor(lambda_feat, dtype=torch.float32, device=device)
    model = FactorizationMachine(n_features, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(
        [
            {"params": [model.bias], "weight_decay": 0.0},
            {"params": [model.linear], "weight_decay": 0.0},
            {"params": [model.factors], "weight_decay": 0.0},
        ],
        lr=lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for feats, targets in train_loader:
            feats = feats.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            logits = model(feats)
            base_loss = loss_fn(logits, targets)
            linear_l2 = (lambda_feat_t * (model.linear ** 2)).sum()
            factor_l2 = (lambda_feat_t[:, None] * (model.factors ** 2)).sum()
            loss = base_loss + linear_l2 + factor_l2
            loss.backward()
            opt.step()
            total_loss += loss.item() * feats.size(0)
            total += feats.size(0)
        train_loss = total_loss / max(total, 1)
        val_loss, val_acc = _evaluate(model, val_loader)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
        )
        if val_loss < (best_val - early_stop_min_delta):
            best_val = val_loss
            best_val_acc = val_acc
            best_state = {
                "state_dict": model.state_dict(),
                "latent_dim": latent_dim,
                "n_features": n_features,
                "role_columns": ROLE_COLUMNS,
                "hero_ids": hero_ids,
                "model_path": model_path,
            }
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch {best_epoch}, val_loss={best_val:.6f})"
                )
                break

    if best_state is None:
        raise RuntimeError("FM training failed to produce a valid model")
    model.load_state_dict(best_state["state_dict"])
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_state, model_path)
        print(f"Saved FM model to {model_path}")
    if return_metrics:
        return model, best_val, best_val_acc
    return model


def _fm_grid_worker(item: Tuple) -> Dict[str, Any]:
    (
        fea_path,
        position_probs_path,
        latent_dim,
        lr,
        l2_lambda0,
        batch_size,
        epochs,
        early_stop_patience,
        early_stop_min_delta,
        idx,
        total,
    ) = item
    print(
        "\n=== FM grid run "
        f"{idx}/{total} "
        f"(latent_dim={latent_dim}, lr={lr}, l2_lambda0={l2_lambda0}) ===",
        flush=True,
    )
    model, val_loss, val_acc = train_fm(
        fea_path=fea_path,
        position_probs_path=position_probs_path,
        latent_dim=latent_dim,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        l2_lambda0=l2_lambda0,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        model_path=None,
        return_metrics=True,
    )
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "latent_dim": latent_dim,
        "lr": lr,
        "l2_lambda0": l2_lambda0,
        "state_dict": state_dict,
        "n_features": model.linear.shape[0],
    }


def train_fm_grid(
    fea_path: str = DATA_PATH,
    position_probs_path: str = DEFAULT_POSITION_PROBS,
    latent_dims: Sequence[int] = FM_GRID_LATENT_DIMS,
    lrs: Sequence[float] = FM_GRID_LRS,
    l2_lambdas: Sequence[float] = FM_GRID_L2_LAMBDAS,
    batch_size: int = FM_BATCH_SIZE,
    epochs: int = FM_EPOCHS,
    early_stop_patience: int = FM_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = FM_EARLY_STOP_MIN_DELTA,
    num_workers: int = FM_GRID_NUM_WORKERS,
    model_path: Optional[str] = FM_MODEL_PATH,
) -> Tuple[nn.Module, Dict[str, Any], float]:
    _, _, _, hero_ids = load_fm_data(
        fea_path=fea_path, position_probs_path=position_probs_path
    )
    best_state = None
    best_cfg = None
    best_val = float("inf")
    best_acc = 0.0

    combos = list(
        itertools.product(latent_dims, lrs, l2_lambdas)
    )
    total = len(combos)
    if num_workers <= 0:
        num_workers = mp.cpu_count()

    args = []
    for idx, (latent_dim, lr, l2_lambda0) in enumerate(combos, start=1):
        args.append(
            (
                fea_path,
                position_probs_path,
                latent_dim,
                lr,
                l2_lambda0,
                batch_size,
                epochs,
                early_stop_patience,
                early_stop_min_delta,
                idx,
                total,
            )
        )

    def _update_best(result):
        nonlocal best_val, best_acc, best_state, best_cfg
        if result["val_loss"] < best_val:
            best_val = result["val_loss"]
            best_acc = result["val_acc"]
            best_state = {
                "state_dict": result["state_dict"],
                "latent_dim": result["latent_dim"],
                "n_features": result["n_features"],
                "role_columns": ROLE_COLUMNS,
                "hero_ids": hero_ids,
                "model_path": model_path,
            }
            best_cfg = {
                "latent_dim": result["latent_dim"],
                "lr": result["lr"],
                "l2_lambda0": result["l2_lambda0"],
                "val_loss": result["val_loss"],
                "val_acc": result["val_acc"],
            }
            print(
                f"New best val_loss={result['val_loss']:.6f} "
                f"val_acc={result['val_acc']:.4f}"
            )

    if num_workers == 1:
        for result in map(_fm_grid_worker, args):
            _update_best(result)
    else:
        with mp.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_fm_grid_worker, args):
                _update_best(result)

    if best_state is None:
        raise RuntimeError("FM grid search failed to produce a valid model")
    model = FactorizationMachine(
        best_state["n_features"], latent_dim=best_state["latent_dim"]
    ).to(device)
    model.load_state_dict(best_state["state_dict"])
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_state, model_path)
        print(f"Saved best FM model to {model_path}")
    print(
        "Best FM config: "
        f"latent_dim={best_cfg['latent_dim']}, "
        f"lr={best_cfg['lr']}, "
        f"l2_lambda0={best_cfg['l2_lambda0']}, "
        f"val_loss={best_cfg['val_loss']:.6f}, "
        f"val_acc={best_cfg['val_acc']:.4f}"
    )
    return model, best_cfg, best_val
