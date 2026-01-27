"""
Factorization machine for lineup-based win rate prediction.

Features are built by multiplying each hero's lineup sign (-1/0/1) by its
position probability vector (Position1..Position5), yielding NUM_HEROS * 5 features.
"""

import os
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from wandao.config import (
    DATA_PATH,
    FM_MODEL_PATH,
    POSITION_PROBS_PATH,
    device,
    FM_LATENT_DIM,
    FM_BATCH_SIZE,
    FM_EPOCHS,
    FM_LR,
    FM_WEIGHT_DECAY,
    FM_EARLY_STOP_PATIENCE,
    FM_EARLY_STOP_MIN_DELTA,
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
    weight_decay: float = FM_WEIGHT_DECAY,
    early_stop_patience: int = FM_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = FM_EARLY_STOP_MIN_DELTA,
    model_path: Optional[str] = FM_MODEL_PATH,
):
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
    model = FactorizationMachine(n_features, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
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
            loss = loss_fn(logits, targets)
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
    return model
