#!/usr/bin/env python3
"""
UMAP + HDBSCAN clustering of per-side lineups.

Provides fit_cluster_model() for pipeline use and a CLI for ad-hoc analysis.
Projects lineups with UMAP (cosine) and clusters with HDBSCAN (no fixed k).
"""

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import joblib


def _resolve_hero_columns(df: pd.DataFrame, feature_names: List[str]) -> List:
    feat_str = [str(f) for f in feature_names]
    str_cols = [c for c in feat_str if c in df.columns]
    if str_cols:
        print(f"Detected string hero columns ({len(str_cols)}).")
        return str_cols

    num_cols = [int(c) for c in feat_str if c.isdigit() and int(c) in df.columns]
    if num_cols:
        print(f"Detected numeric hero columns ({len(num_cols)}).")
        return num_cols

    raise ValueError("No hero columns found in lineup data.")


def _format_heroes(indices: List, feature_names: List[str], id_to_name: dict) -> str:
    names = []
    for col in indices:
        hid = None
        if isinstance(col, str) and col.isdigit():
            hid = int(col)
        elif isinstance(col, (int, np.integer)):
            hid = int(col)
        if hid is not None and hid in id_to_name:
            names.append(id_to_name[hid])
        else:
            names.append(str(col))
    return ", ".join(names)


class ClusterModel:
    """Encapsulates UMAP reducer and HDBSCAN cluster centers for reward shaping."""

    def __init__(
        self,
        reducer,
        centers: np.ndarray,
        hero_cols: Sequence,
        feature_names: List[str],
        cluster_params: dict,
    ):
        self.reducer = reducer
        self.centers = centers
        self.hero_cols = list(hero_cols)
        self.hero_to_pos = {
            int(h): i
            for i, h in enumerate(hero_cols)
            if isinstance(h, (int, np.integer))
        }
        self.hero_to_pos.update(
            {
                int(h): i
                for i, h in enumerate(hero_cols)
                if isinstance(h, str) and h.isdigit()
            }
        )
        self.feature_names = feature_names
        self.cluster_params = cluster_params

    def distance_from_team(self, team_indices: Sequence[int]) -> float:
        if self.centers.size == 0:
            return 0.0
        vec = np.zeros(len(self.hero_cols), dtype=np.float32)
        for idx in team_indices:
            try:
                hid = int(self.feature_names[idx])
            except Exception:
                continue
            pos = self.hero_to_pos.get(hid)
            if pos is not None:
                vec[pos] = 1.0
        if vec.sum() == 0:
            return float(np.linalg.norm(self.centers, axis=1).min())
        v = normalize(vec.reshape(1, -1), norm="l2", copy=False)
        emb = self.reducer.transform(v)
        dists = np.linalg.norm(emb - self.centers, axis=1)
        return float(dists.min())


def fit_cluster_model(
    lineup_df: pd.DataFrame,
    feature_names: List[str],
    components: int = 5,
    n_neighbors: int = 30,
    min_dist: float = 0.05,
    min_cluster_size: int = 30,
    min_samples: Optional[int] = None,
    sample_frac: float = 1.0,
    random_state: Optional[int] = None,
) -> ClusterModel:
    hero_cols = _resolve_hero_columns(lineup_df, feature_names)
    X = lineup_df[hero_cols].astype(np.float32)

    if not (0 < sample_frac <= 1.0):
        raise ValueError("sample_frac must be in (0,1].")
    if sample_frac < 1.0:
        X = X.sample(frac=sample_frac, random_state=random_state)

    X_norm = normalize(X.values, norm="l2", copy=False)

    try:
        import umap
    except ImportError as e:
        raise ImportError("Please install umap-learn (pip install umap-learn).") from e

    try:
        import hdbscan
    except ImportError as e:
        raise ImportError("Please install hdbscan (pip install hdbscan).") from e

    reducer = umap.UMAP(
        n_components=components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X_norm)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embedding)

    centers = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        if mask.any():
            centers.append(embedding[mask].mean(axis=0))
    centers = np.array(centers, dtype=np.float32)
    if centers.size == 0:
        print("Warning: HDBSCAN found no clusters (all noise).")
    cluster_params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": "euclidean",
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
    }
    return ClusterModel(reducer, centers, hero_cols, feature_names, cluster_params)


def save_cluster_model(model: ClusterModel, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved cluster model to {path}")


def load_cluster_model(path: str) -> ClusterModel:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cluster model not found at {path}")
    model = joblib.load(path)
    return model
