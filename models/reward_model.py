"""
Reward model for win prediction using factorization machine (FM).
"""

import os
from typing import Optional

import numpy as np
import torch

from wandao.config import DATA_PATH, FM_MODEL_PATH
from wandao.models.factorization_machine import (
    FactorizationMachine,
    build_fm_features,
    build_role_matrix,
    load_fm_data,
    load_position_probs,
    train_fm,
)


class RewardModel:
    """Reward model backed by FM only."""

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.model = None
        self.role_probs = None
        self.fm_hero_ids = None
        self.fm_feature_indices = None

    def load(self):
        return self._load_fm()

    def _load_fm(self):
        if not os.path.exists(FM_MODEL_PATH):
            return False
        checkpoint = torch.load(FM_MODEL_PATH, map_location="cpu")
        n_features = checkpoint["n_features"]
        latent_dim = checkpoint["latent_dim"]
        model = FactorizationMachine(n_features, latent_dim=latent_dim)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model
        self.role_probs = None
        self.fm_hero_ids = checkpoint.get("hero_ids")
        self.fm_feature_indices = None
        print(f"Loaded FM model from {FM_MODEL_PATH}")
        return True

    def _prepare_fm_mappings(self):
        if self.fm_hero_ids is None:
            _, _, role_probs, hero_ids = load_fm_data()
            self.fm_hero_ids = hero_ids
            self.role_probs = role_probs
        if self.role_probs is None:
            pos_df = load_position_probs()
            role_probs, _ = build_role_matrix(self.fm_hero_ids, pos_df)
            self.role_probs = role_probs
        feature_index = {str(hid): i for i, hid in enumerate(self.feature_names)}
        missing = [hid for hid in self.fm_hero_ids if str(hid) not in feature_index]
        if missing:
            preview = ", ".join(str(h) for h in missing[:10])
            raise ValueError(
                "FM hero ids missing from feature_names: "
                f"{len(missing)} (e.g., {preview})"
            )
        self.fm_feature_indices = np.array(
            [feature_index[str(hid)] for hid in self.fm_hero_ids], dtype=np.int64
        )

    def predict_proba(self, vec: np.ndarray) -> float:
        if self.model is None:
            raise ValueError("Reward model not loaded or trained.")
        if self.role_probs is None or self.fm_feature_indices is None:
            self._prepare_fm_mappings()
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] == len(self.feature_names):
            vec = vec[self.fm_feature_indices]
        elif vec.shape[0] != len(self.fm_hero_ids):
            raise ValueError(
                "FM expects lineup vector sized to feature_names or FM hero ids."
            )
        x = build_fm_features(vec.reshape(1, -1), self.role_probs)
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x).item()
        proba = 1 / (1 + np.exp(-logits))
        return float(np.clip(proba, 0.0, 1.0))

    def train(
        self,
        fea_path: Optional[str] = None,
        fetch_data: bool = False,
        n_batches: int = 1000,
        api_key: Optional[str] = None,
        **_: object,
    ):
        if fetch_data:
            raise ValueError("Use the pipeline data fetch step before training FM.")
        train_fm(fea_path=fea_path or DATA_PATH)
        self._load_fm()
