"""
Expectation-Maximization (EM) model for estimating hero position probabilities.
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from wandao.config import (
    LINEUP_PATH,
    POSITION_PROBS_PATH,
    EM_MAX_LINEUPS,
    EM_MAX_ITERATIONS,
    EM_CONVERGENCE_THRESHOLD,
    EM_EARLY_STOP_PATIENCE,
    EM_EARLY_STOP_MIN_DELTA,
    EM_NUM_WORKERS,
    EM_VALID_FRACTION,
    EM_SEED,
)


ROLE_COLUMNS = ["Position1", "Position2", "Position3", "Position4", "Position5"]

_MP_SHM = None
_MP_PROBS = None
_MP_PERMS = None
_MP_HARD = False


def _hard_counts_for_lineup(lineup, hero_position_probs, position_permutations):
    best_log_prob = -np.inf
    best_perm = None
    for perm in position_permutations:
        log_prob = 0.0
        for i, hero in enumerate(lineup):
            position = perm[i]
            prob = hero_position_probs[hero, position]
            if prob > 0:
                log_prob += np.log(prob)
            else:
                log_prob += -1e10
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_perm = perm

    position_counts = np.zeros((5, hero_position_probs.shape[1]))
    for i, hero_idx in enumerate(lineup):
        position_counts[i, best_perm[i]] = 1.0

    return position_counts, -best_log_prob


def _chunk_lineups(lineups, num_workers):
    if num_workers <= 1:
        return [lineups]
    chunk_size = max(1, len(lineups) // (num_workers * 4))
    return [lineups[i : i + chunk_size] for i in range(0, len(lineups), chunk_size)]


def _mp_init(shm_name, shape, dtype, position_permutations, hard_em):
    global _MP_SHM, _MP_PROBS, _MP_PERMS, _MP_HARD
    _MP_SHM = shared_memory.SharedMemory(name=shm_name)
    _MP_PROBS = np.ndarray(shape, dtype=np.dtype(dtype), buffer=_MP_SHM.buf)
    _MP_PERMS = position_permutations
    _MP_HARD = hard_em


def _mp_process_chunk(lineups):
    hero_position_counts = np.zeros((_MP_PROBS.shape[0], _MP_PROBS.shape[1]))
    total_loss = 0.0
    for lineup in lineups:
        position_counts, loss = _hard_counts_for_lineup(lineup, _MP_PROBS, _MP_PERMS)
        total_loss += loss
        for i, hero in enumerate(lineup):
            hero_position_counts[hero] += position_counts[i]

    return hero_position_counts, total_loss, len(lineups)


class Dota2PositionLearner:
    def __init__(self, num_heroes=127, num_positions=5, seed=42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.num_heroes = num_heroes
        self.num_positions = num_positions
        self.hero_position_probs = self._initialize_priors()
        self.convergence_history = []
        self.loss_history = []
        self._position_permutations = list(
            itertools.permutations(range(self.num_positions))
        )

    def _initialize_priors(self):
        return self.rng.dirichlet([1] * self.num_positions, size=self.num_heroes)

    def _kmeans(self, data, k, n_init=10, max_iter=100):
        best_inertia = np.inf
        best_labels = None
        best_centers = None
        n_samples = data.shape[0]

        for _ in range(n_init):
            init_idx = self.rng.choice(n_samples, size=k, replace=False)
            centers = data[init_idx].copy()

            for _ in range(max_iter):
                dists = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = np.argmin(dists, axis=1)
                new_centers = np.zeros_like(centers)

                for c in range(k):
                    mask = labels == c
                    if not np.any(mask):
                        new_centers[c] = data[self.rng.integers(0, n_samples)]
                    else:
                        new_centers[c] = data[mask].mean(axis=0)

                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

            inertia = np.sum(((data - centers[labels]) ** 2).sum(axis=1))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centers = centers

        return best_labels, best_centers

    def _initialize_from_lineups(self, lineups, n_init=10, max_iter=100):
        if not lineups:
            raise ValueError("Lineups are empty; cannot initialize from data.")

        counts = np.zeros(self.num_heroes, dtype=float)
        co_matrix = np.zeros((self.num_heroes, self.num_heroes), dtype=float)

        for lineup in lineups:
            for hero in lineup:
                counts[hero] += 1
            for hero_i in lineup:
                for hero_j in lineup:
                    if hero_i != hero_j:
                        co_matrix[hero_i, hero_j] += 1

        denom = np.sqrt(np.outer(counts, counts))
        denom[denom == 0] = 1.0
        sim = co_matrix / denom
        np.fill_diagonal(sim, 1.0)

        _, vecs = np.linalg.eigh(sim)
        topk = vecs[:, -self.num_positions :]
        norms = np.linalg.norm(topk, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = topk / norms

        labels, centers = self._kmeans(
            embeddings, self.num_positions, n_init=n_init, max_iter=max_iter
        )
        if labels is None or centers is None:
            raise RuntimeError("KMeans initialization failed.")

        dists = ((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        scale = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
        scale = max(scale, 1e-6)
        scores = np.exp(-dists / scale)
        return scores / scores.sum(axis=1, keepdims=True)

    def _evaluate_loss(self, lineups, hard_em=True, num_workers=0):
        num_workers = num_workers or mp.cpu_count()
        if not lineups:
            return 0.0
        total_loss = 0.0
        if num_workers <= 1:
            for lineup in lineups:
                if hard_em:
                    _, loss = _hard_counts_for_lineup(
                        lineup, self.hero_position_probs, self._position_permutations
                    )
                    total_loss += loss
            return total_loss / max(len(lineups), 1)

        shm = shared_memory.SharedMemory(
            create=True, size=self.hero_position_probs.nbytes
        )
        shm_array = np.ndarray(
            self.hero_position_probs.shape,
            dtype=self.hero_position_probs.dtype,
            buffer=shm.buf,
        )
        shm_array[:] = self.hero_position_probs
        chunks = _chunk_lineups(lineups, num_workers)
        with mp.Pool(
            processes=num_workers,
            initializer=_mp_init,
            initargs=(
                shm.name,
                self.hero_position_probs.shape,
                self.hero_position_probs.dtype,
                self._position_permutations,
                hard_em,
            ),
        ) as pool:
            results = list(pool.imap(_mp_process_chunk, chunks))
        shm.close()
        shm.unlink()
        total_loss = sum(loss for _, loss, _ in results)
        total_count = sum(count for _, _, count in results)
        return total_loss / max(total_count, 1)

    def train_em(
        self,
        lineups,
        max_iterations=20,
        convergence_threshold=1e-4,
        init_from_lineups=True,
        hard_em=True,
        early_stop_patience=0,
        early_stop_min_delta=1e-4,
        num_workers=0,
        valid_lineups=None,
    ):
        mode_label = "hard" if hard_em else "soft"
        print(f"Starting {mode_label} EM training...")
        if init_from_lineups:
            try:
                self.hero_position_probs = self._initialize_from_lineups(lineups)
            except Exception as exc:
                print(f"Warning: data-driven init failed ({exc}); using random priors.")
                self.hero_position_probs = self._initialize_priors()
        else:
            self.hero_position_probs = self._initialize_priors()

        num_workers = num_workers or mp.cpu_count()
        valid_lineups = valid_lineups or []
        best_loss = None
        bad_epochs = 0
        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")
            total_loss = 0.0
            if num_workers <= 1:
                all_position_counts = []
                for lineup in lineups:
                    position_counts, loss = _hard_counts_for_lineup(
                        lineup, self.hero_position_probs, self._position_permutations
                    )
                    total_loss += loss
                    all_position_counts.append(position_counts)
                hero_position_counts = np.zeros((self.num_heroes, self.num_positions))
                for lineup, position_counts in zip(lineups, all_position_counts):
                    for i, hero in enumerate(lineup):
                        hero_position_counts[hero] += position_counts[i]
                hero_position_counts += 1e-3
                new_hero_position_probs = (
                    hero_position_counts
                    / hero_position_counts.sum(axis=1, keepdims=True)
                )
            else:
                shm = shared_memory.SharedMemory(
                    create=True, size=self.hero_position_probs.nbytes
                )
                shm_array = np.ndarray(
                    self.hero_position_probs.shape,
                    dtype=self.hero_position_probs.dtype,
                    buffer=shm.buf,
                )
                shm_array[:] = self.hero_position_probs
                chunks = _chunk_lineups(lineups, num_workers)
                with mp.Pool(
                    processes=num_workers,
                    initializer=_mp_init,
                    initargs=(
                        shm.name,
                        self.hero_position_probs.shape,
                        self.hero_position_probs.dtype,
                        self._position_permutations,
                        True,
                    ),
                ) as pool:
                    results = list(pool.imap(_mp_process_chunk, chunks))
                shm.close()
                shm.unlink()

                hero_position_counts = np.zeros_like(self.hero_position_probs)
                total_lineups = 0
                for counts, loss, count in results:
                    hero_position_counts += counts
                    total_loss += loss
                    total_lineups += count

                hero_position_counts += 1e-3
                new_hero_position_probs = (
                    hero_position_counts
                    / hero_position_counts.sum(axis=1, keepdims=True)
                )

            avg_loss = total_loss / max(len(lineups), 1)
            self.loss_history.append(avg_loss)
            print(f"Train assignment loss: {avg_loss:.6f}")
            change = np.abs(new_hero_position_probs - self.hero_position_probs).mean()
            self.convergence_history.append(change)
            print(f"Average parameter change: {change:.6f}")
            self.hero_position_probs = new_hero_position_probs

            if valid_lineups:
                valid_loss = self._evaluate_loss(
                    valid_lineups, hard_em=True, num_workers=num_workers
                )
                print(f"Validation assignment loss: {valid_loss:.6f}")
                if early_stop_patience > 0:
                    if (
                        best_loss is None
                        or (best_loss - valid_loss) > early_stop_min_delta
                    ):
                        best_loss = valid_loss
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= early_stop_patience:
                            print("Early stopping: validation loss stopped improving.")
                            break
            elif early_stop_patience > 0:
                if best_loss is None or (best_loss - avg_loss) > early_stop_min_delta:
                    best_loss = avg_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        print("Early stopping: loss stopped improving.")
                        break
            if change < convergence_threshold:
                print(f"Converged at iteration {iteration + 1}")
                break

        return self.hero_position_probs


def _resolve_paths(fea_path: str, output_csv: str) -> Tuple[Path, Path]:
    fea = Path(fea_path) if fea_path else Path(LINEUP_PATH)
    out = Path(output_csv) if output_csv else Path(POSITION_PROBS_PATH)
    return fea, out


def load_lineups(fea_path: Path, max_lineups: int, seed: int):
    df = pd.read_feather(fea_path)
    hero_cols = [c for c in df.columns if c not in ("win", "side", "match_id")]
    if not hero_cols:
        raise ValueError("No hero columns found in lineup data.")

    try:
        hero_cols = sorted(hero_cols, key=lambda c: int(c))
    except ValueError:
        hero_cols = sorted(hero_cols)

    hero_ids = [int(c) if str(c).isdigit() else c for c in hero_cols]
    hero_matrix = df[hero_cols].values
    row_sums = hero_matrix.sum(axis=1)
    valid_mask = row_sums == 5
    if not np.all(valid_mask):
        print(f"Filtered out {np.size(row_sums) - valid_mask.sum()} invalid lineups.")
        hero_matrix = hero_matrix[valid_mask]

    if max_lineups and max_lineups > 0 and hero_matrix.shape[0] > max_lineups:
        rng = np.random.default_rng(seed)
        indices = rng.choice(hero_matrix.shape[0], size=max_lineups, replace=False)
        hero_matrix = hero_matrix[indices]

    lineups = [np.flatnonzero(row).tolist() for row in hero_matrix]
    print(f"Loaded {len(lineups)} lineups with {len(hero_ids)} heroes.")
    return lineups, hero_ids


def split_lineups(lineups, valid_frac, seed):
    if valid_frac <= 0:
        return lineups, []
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(lineups))
    split_idx = int(len(lineups) * (1 - valid_frac))
    train_idx = indices[:split_idx]
    valid_idx = indices[split_idx:]
    train = [lineups[i] for i in train_idx]
    valid = [lineups[i] for i in valid_idx]
    print(f"Train lineups: {len(train)}, Validation lineups: {len(valid)}")
    return train, valid


def save_results(hero_ids, hero_position_probs, output_path: Path):
    df = pd.DataFrame(hero_position_probs, columns=ROLE_COLUMNS)
    df.insert(0, "Hero_ID", hero_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    return df


def train_em(
    fea_path: str | None = None,
    output_csv: str | None = None,
    max_lineups: int = EM_MAX_LINEUPS,
    max_iterations: int = EM_MAX_ITERATIONS,
    convergence_threshold: float = EM_CONVERGENCE_THRESHOLD,
    early_stop_patience: int = EM_EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = EM_EARLY_STOP_MIN_DELTA,
    num_workers: int = EM_NUM_WORKERS,
    valid_fraction: float = EM_VALID_FRACTION,
    seed: int = EM_SEED,
):
    fea_path, output_path = _resolve_paths(fea_path, output_csv)
    if not fea_path.exists():
        raise FileNotFoundError(f"Lineup data not found: {fea_path}")

    lineups, hero_ids = load_lineups(fea_path, max_lineups, seed)
    train_lineups, valid_lineups = split_lineups(lineups, valid_fraction, seed)
    learner = Dota2PositionLearner(num_heroes=len(hero_ids), num_positions=5, seed=seed)

    learner.train_em(
        train_lineups,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        num_workers=num_workers,
        valid_lineups=valid_lineups,
    )

    save_results(hero_ids, learner.hero_position_probs, output_path)
    return learner


def main():
    parser = argparse.ArgumentParser(
        description="Train Dota2PositionLearner on real lineup data."
    )
    parser.add_argument("--fea-path", default=None, help="Path to pub_lineups.fea")
    parser.add_argument(
        "--output-csv", default=None, help="Path to write hero position probabilities"
    )
    args = parser.parse_args()

    train_em(fea_path=args.fea_path, output_csv=args.output_csv)


if __name__ == "__main__":
    main()
