"""
Reward models for win prediction: default XGBoost with optional MLP backend.
"""

import json
import os
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import joblib
except ImportError:
    joblib = None

from draft.config import (
    BASE_DIR,
    DATA_PATH,
    REWARD_MODEL_JOBLIB,
    REWARD_MODEL_TXT,
    REWARD_MODEL_MLP,
    REWARD_MODEL_META,
)
from draft.data.data_fetch import construct_dataset, get_encoded_data
from draft.data.encoding import encode_df


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RewardModel:
    """Reward model supporting XGBoost (default) and MLP backends."""

    def __init__(self, feature_names, backend="xgb"):
        self.feature_names = feature_names
        self.backend = backend
        self.model = None
        self.temperature = 1.0
        self.best_iteration = None  # for xgboost

    # ------------------------------------------------------------------ summary helper
    @staticmethod
    def _print_summary(train_logloss, train_acc, val_logloss, val_acc):
        loss_ratio = val_logloss / train_logloss if train_logloss != 0 else float("inf")
        acc_ratio = val_acc / train_acc if train_acc != 0 else float("inf")
        print()
        print(f"Training logloss: {train_logloss:.6f}")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation logloss: {val_logloss:.6f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Loss ratio (val/train): {loss_ratio:.4f}")
        print(f"Accuracy ratio (val/train): {acc_ratio:.4f}")
        return loss_ratio, acc_ratio

    # ------------------------------------------------------------------ Loading
    def load(self):
        if self.backend == "mlp":
            return self._load_mlp()
        elif self.backend == "xgb":
            return self._load_xgb()
        return False

    def _load_mlp(self):
        if not os.path.exists(REWARD_MODEL_MLP):
            return False
        checkpoint = torch.load(REWARD_MODEL_MLP, map_location="cpu")
        input_dim = checkpoint["input_dim"]
        hidden = checkpoint["hidden"]
        dropout = checkpoint["dropout"]
        self.temperature = checkpoint.get("temperature", 1.0)
        model = SimpleMLP(input_dim, hidden=hidden, dropout=dropout)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.model = model
        print(
            f"Loaded MLP reward model from {REWARD_MODEL_MLP} (temp={self.temperature})"
        )
        return True

    def _load_xgb(self):
        # Try joblib first
        if joblib and os.path.exists(REWARD_MODEL_JOBLIB):
            try:
                self.model = joblib.load(REWARD_MODEL_JOBLIB)
                if hasattr(self.model, "attr"):
                    bi = self.model.attr("best_iteration")
                    if bi is not None:
                        try:
                            self.best_iteration = int(bi)
                        except ValueError:
                            self.best_iteration = None
                print("Loaded XGBoost model via joblib")
                return True
            except Exception as e:
                print("joblib load failed:", e)

        # Try XGBoost text/binary format
        if xgb and os.path.exists(REWARD_MODEL_TXT):
            try:
                booster = xgb.Booster()
                booster.load_model(REWARD_MODEL_TXT)
                self.model = booster
                bi = booster.attr("best_iteration")
                if bi is not None:
                    try:
                        self.best_iteration = int(bi)
                    except ValueError:
                        self.best_iteration = None
                print("Loaded XGBoost Booster from file")
                return True
            except Exception as e:
                print("XGBoost load failed:", e)
        return False

    # ------------------------------------------------------------------ Predict
    def predict_proba(self, vec: np.ndarray) -> float:
        if self.model is None:
            raise ValueError("Reward model not loaded or trained.")

        if self.backend == "mlp":
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(x) / self.temperature
                proba = torch.sigmoid(logits).item()
            return float(np.clip(proba, 0.0, 1.0))

        # XGB
        dm = xgb.DMatrix(vec.reshape(1, -1))
        iter_range = (0, self.best_iteration) if self.best_iteration else None
        proba = self.model.predict(
            dm, iteration_range=iter_range if iter_range else None
        )
        if isinstance(proba, np.ndarray):
            proba = proba[0]
        return float(np.clip(proba, 0.0, 1.0))

    # ------------------------------------------------------------------ Train entry
    def train(
        self,
        fea_path: Optional[str] = None,
        fetch_data: bool = False,
        n_batches: int = 1000,
        api_key: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """Train reward model.

        Args:
            fea_path: Path to encoded feather file (optional).
            fetch_data: If True, fetch data from OpenDota before training.
            n_batches: Batches to fetch when fetch_data=True.
            api_key: Optional OpenDota API key.
            backend: Override backend ("mlp" or "xgb"); default uses self.backend.
        """
        backend = backend or self.backend
        self.backend = backend
        df = self._load_data(
            fea_path, fetch_data, n_batches, api_key, force_refresh=True
        )
        y = df["radiant_win"].values
        X = df.drop(columns=["radiant_win"]).values.astype(np.float32)

        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if backend == "xgb":
            self._train_xgb(X_train, y_train, X_val, y_val, **kwargs)
        else:
            self._train_mlp(X_train, y_train, X_val, y_val, **kwargs)

    # ------------------------------------------------------------------ Data helper
    def _load_data(self, fea_path, fetch_data, n_batches, api_key, force_refresh=False):
        if fetch_data:
            print(
                f"Fetching up to {n_batches} batches of match data from OpenDota API..."
            )
            raw_df = construct_dataset(
                n=n_batches, feature_names=self.feature_names, api_key=api_key
            )
            if raw_df.empty:
                raise ValueError("No data fetched from API")
            print("Encoding dataset...")
            df = encode_df(raw_df, self.feature_names)
            print(f"Saving encoded dataset to {DATA_PATH}")
            df.to_feather(DATA_PATH)
        elif fea_path and os.path.exists(fea_path):
            df = pd.read_feather(fea_path)
        else:
            df = get_encoded_data(self.feature_names, force_refresh=force_refresh)
            if df.empty:
                raise FileNotFoundError(
                    "No training data available. Set fetch_data=True to download data."
                )

        if "radiant_win" not in df.columns:
            raise ValueError("Dataset must contain 'radiant_win' column")
        return df

    # ------------------------------------------------------------------ MLP training
    def _train_mlp(
        self, X_train, y_train, X_val, y_val, num_epochs=15, batch_size=256, **_: object
    ):
        input_dim = X_train.shape[1]
        grid = {
            "hidden": [128, 256],
            "dropout": [0.1, 0.2],
            "lr": [3e-4, 1e-3],
            "weight_decay": [0.0, 1e-4],
        }
        combos = [
            {k: v for (k, _), v in zip(grid.items(), values)}
            for values in product(*grid.values())
        ]
        if not combos:
            combos = [{"hidden": 256, "dropout": 0.1, "lr": 1e-3, "weight_decay": 0.0}]

        from sklearn.metrics import log_loss, accuracy_score

        best_loss = float("inf")
        best_state = None
        best_combo = None
        best_temp = 1.0

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_np = y_val

        for i, combo in enumerate(combos, 1):
            print(f"[{i}/{len(combos)}] Training MLP with params: {combo}")
            model = SimpleMLP(
                input_dim, hidden=combo["hidden"], dropout=combo["dropout"]
            )
            opt = torch.optim.Adam(
                model.parameters(), lr=combo["lr"], weight_decay=combo["weight_decay"]
            )
            model.train()
            for _ in range(num_epochs):
                idx = torch.randperm(X_train_t.size(0))
                for start in range(0, len(idx), batch_size):
                    batch_idx = idx[start : start + batch_size]
                    xb = X_train_t[batch_idx]
                    yb = y_train_t[batch_idx]
                    opt.zero_grad()
                    logits = model(xb)
                    loss = F.binary_cross_entropy_with_logits(logits, yb)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).numpy()

            # Temperature tuning (small grid)
            temps = [0.7, 1.0, 1.3, 1.6, 2.0]
            best_temp_local = 1.0
            best_val_loss_local = float("inf")
            for temp in temps:
                probs = 1 / (1 + np.exp(-val_logits / temp))
                ll = log_loss(y_val_np, probs, labels=[0, 1])
                if ll < best_val_loss_local:
                    best_val_loss_local = ll
                    best_temp_local = temp

            if best_val_loss_local < best_loss:
                best_loss = best_val_loss_local
                best_state = {
                    "state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden": combo["hidden"],
                    "dropout": combo["dropout"],
                    "temperature": best_temp_local,
                }
                best_combo = combo
                best_temp = best_temp_local

            val_acc = accuracy_score(
                y_val_np, (1 / (1 + np.exp(-val_logits / best_temp_local)) >= 0.5)
            )
            print(
                f"Validation logloss: {best_val_loss_local:.6f}, acc: {val_acc:.4f}, temp: {best_temp_local}"
            )

        if best_state is None:
            raise RuntimeError("Failed to train MLP reward model")

        # Save best model
        os.makedirs(os.path.dirname(REWARD_MODEL_MLP), exist_ok=True)
        torch.save(best_state, REWARD_MODEL_MLP)
        self.model = SimpleMLP(
            input_dim, hidden=best_state["hidden"], dropout=best_state["dropout"]
        )
        self.model.load_state_dict(best_state["state_dict"])
        self.model.eval()
        self.temperature = best_temp

        # Training/validation metrics summary
        train_logits = self.model(X_train_t).detach().numpy()
        train_probs = 1 / (1 + np.exp(-train_logits / best_temp))
        train_logloss = log_loss(y_train, train_probs, labels=[0, 1])
        train_acc = accuracy_score(y_train, (train_probs >= 0.5))

        val_logits = (
            self.model(torch.tensor(X_val, dtype=torch.float32)).detach().numpy()
        )
        val_probs = 1 / (1 + np.exp(-val_logits / best_temp))
        val_logloss = log_loss(y_val, val_probs, labels=[0, 1])
        val_acc = accuracy_score(y_val, (val_probs >= 0.5))

        loss_ratio, acc_ratio = self._print_summary(
            train_logloss, train_acc, val_logloss, val_acc
        )

        # Persist metadata
        self._save_metadata(
            {
                "backend": "mlp",
                "params": best_combo,
                "temperature": best_temp,
                "val_logloss": val_logloss,
                "train_logloss": train_logloss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "loss_ratio": loss_ratio,
                "acc_ratio": acc_ratio,
            }
        )
        print(f"Saved MLP reward model to {REWARD_MODEL_MLP} (temp={best_temp})")

    # ------------------------------------------------------------------ XGBoost training
    def _train_xgb(self, X_train, y_train, X_val, y_val, num_boost_round=400):
        if xgb is None:
            raise RuntimeError("xgboost is not available in this environment")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        base_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.06,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "alpha": 0.0,
        }
        grid_space = {
            "eta": [0.03, 0.06],
            "max_depth": [4, 6],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        allowed_meta_keys = set(base_params.keys()) | set(grid_space.keys())
        if os.path.exists(REWARD_MODEL_META):
            try:
                with open(REWARD_MODEL_META, "r") as mf:
                    meta = json.load(mf)
                for k, v in meta.get("params", {}).items():
                    if k == "objective":
                        continue
                    if k in allowed_meta_keys:
                        base_params[k] = v
            except Exception:
                pass
        base_params["objective"] = "binary:logistic"

        combos = [
            {k: v for (k, _), v in zip(grid_space.items(), values)}
            for values in product(*grid_space.values())
        ]
        if not combos:
            combos = [{}]

        print(f"Running XGBoost grid search over {len(combos)} hyperparameter sets...")
        from sklearn.metrics import log_loss, accuracy_score

        best_logloss = float("inf")
        best_model = None
        best_params = None
        best_iter = None

        for idx, combo in enumerate(combos, 1):
            params = dict(base_params)
            params.update(combo)
            print(f"[{idx}/{len(combos)}] Training with params: {combo}")

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "validation")],
                early_stopping_rounds=50,
                verbose_eval=False,
            )

            use_iter = (
                booster.best_iteration + 1
                if booster.best_iteration is not None
                else None
            )
            y_pred = booster.predict(
                dval, iteration_range=(0, use_iter) if use_iter else None
            )
            val_logloss = log_loss(y_val, y_pred)
            print(f"Validation logloss: {val_logloss:.6f}")

            if val_logloss < best_logloss:
                best_logloss = val_logloss
                best_model = booster
                best_params = params
                best_iter = use_iter

        if best_model is None:
            raise RuntimeError("Failed to train any XGBoost models during grid search")

        y_pred_best = best_model.predict(
            dval, iteration_range=(0, best_iter) if best_iter else None
        )
        val_acc = accuracy_score(y_val, (y_pred_best >= 0.5).astype(int))

        # In-sample metrics
        y_pred_train = best_model.predict(
            dtrain, iteration_range=(0, best_iter) if best_iter else None
        )
        from sklearn.metrics import log_loss as sk_log_loss, accuracy_score as sk_acc

        train_logloss = sk_log_loss(y_train, y_pred_train)
        train_acc = sk_acc(y_train, (y_pred_train >= 0.5).astype(int))
        loss_ratio, acc_ratio = self._print_summary(
            train_logloss, train_acc, best_logloss, val_acc
        )

        # Save model
        models_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)
        if best_iter is not None:
            best_model.set_attr(best_iteration=str(best_iter))
        try:
            best_model.save_model(REWARD_MODEL_TXT)
            print(f"Saved model to {REWARD_MODEL_TXT}")
        except Exception as e:
            print(f"Failed to save text model: {e}")

        if joblib is not None:
            try:
                joblib.dump(best_model, REWARD_MODEL_JOBLIB)
                print(f"Saved model to {REWARD_MODEL_JOBLIB}")
            except Exception as e:
                print(f"Failed to save joblib model: {e}")

        self.model = best_model
        self.best_iteration = best_iter
        self.temperature = 1.0

        self._save_metadata(
            {
                "backend": "xgb",
                "params": best_params,
                "val_logloss": best_logloss,
                "best_iteration": best_iter,
                "train_logloss": train_logloss,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "loss_ratio": loss_ratio,
                "acc_ratio": acc_ratio,
            }
        )

        print("XGBoost model training complete")

    # ------------------------------------------------------------------ metadata
    def _save_metadata(self, payload):
        try:
            with open(REWARD_MODEL_META, "w") as mf:
                json.dump(payload, mf, indent=2)
        except Exception as e:
            print(f"Failed to save metadata: {e}")
