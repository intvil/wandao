#!/usr/bin/env python3
"""Minimal draft UI backend."""

from __future__ import annotations

import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from flask import Flask, jsonify, request, send_from_directory
import numpy as np

try:
    from wandao.config import (
        DATA_CACHE_DIR,
        FM_MODEL_PATH,
        TOTAL_STEPS,
        DRAFT_BATCH_EVAL,
        DRAFT_EVAL_CACHE_SIZE,
        DRAFT_MP_EVAL,
        DRAFT_MP_WORKERS,
        DRAFT_TOP_K,
    )
    from wandao.models.policy import is_ban_step, is_pick_step, side_to_move
    from wandao.models.reward_model import RewardModel
    from wandao.search.expectimax import _EvalContext, _choose_random_action_with_trace
    from wandao.utils.utils import load_feature_names
except ImportError:
    # Allow running as a script from repo root without installing the package
    PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROJECT_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
    for path in (PROJECT_ROOT,):
        if path not in sys.path:
            sys.path.insert(0, path)
    from wandao.config import (
        DATA_CACHE_DIR,
        FM_MODEL_PATH,
        TOTAL_STEPS,
        DRAFT_BATCH_EVAL,
        DRAFT_EVAL_CACHE_SIZE,
        DRAFT_MP_EVAL,
        DRAFT_MP_WORKERS,
        DRAFT_TOP_K,
    )
    from wandao.models.policy import is_ban_step, is_pick_step, side_to_move
    from wandao.models.reward_model import RewardModel
    from wandao.search.expectimax import _EvalContext, _choose_random_action_with_trace
    from wandao.utils.utils import load_feature_names


app = Flask(__name__)


@dataclass
class DraftState:
    step: int
    available: List[bool]
    team_radiant: List[int]
    team_dire: List[int]
    bans: List[int]
    user_side: int


_state: Optional[DraftState] = None
_hero_names: List[str] = []
_name_to_idx: Dict[str, int] = {}
_reward_model: Optional[RewardModel] = None
_draft_ctx: Optional[_EvalContext] = None


def _load_hero_names() -> List[str]:
    cache_path = os.path.join(DATA_CACHE_DIR, "hero_names.json")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Hero names cache not found at {cache_path}; run pipeline first."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Hero names cache is invalid")
    # feature_names are hero IDs (strings) in model order
    feature_names = load_feature_names()
    id_to_name = {int(k): v for k, v in raw.items() if v}
    return [id_to_name.get(int(hid), str(hid)) for hid in feature_names]


def _load_model() -> RewardModel:
    if not os.path.exists(FM_MODEL_PATH):
        raise FileNotFoundError(
            f"FM model not found at {FM_MODEL_PATH}; train the model first."
        )
    feature_names = load_feature_names()
    model = RewardModel(feature_names)
    if not model.load():
        raise RuntimeError("Failed to load FM model")
    return model


def _ensure_loaded():
    global _hero_names, _name_to_idx, _reward_model, _draft_ctx
    if not _hero_names:
        _hero_names = _load_hero_names()
        _name_to_idx = {name.lower(): i for i, name in enumerate(_hero_names)}
    if _reward_model is None:
        _reward_model = _load_model()
        _draft_ctx = _EvalContext(
            reward_model=_reward_model,
            n_champs=len(_hero_names),
            use_batch=DRAFT_BATCH_EVAL,
            use_mp=DRAFT_MP_EVAL,
            mp_workers=max(1, DRAFT_MP_WORKERS),
            cache_size=DRAFT_EVAL_CACHE_SIZE,
            cache=OrderedDict(),
        )


def _new_state(user_side: int) -> DraftState:
    _ensure_loaded()
    return DraftState(
        step=0,
        available=[True] * len(_hero_names),
        team_radiant=[],
        team_dire=[],
        bans=[],
        user_side=user_side,
    )


def _current_win_rate(state: DraftState) -> float:
    if _reward_model is None:
        raise RuntimeError("Model not loaded")
    teamA = state.team_radiant
    teamB = state.team_dire
    if state.user_side == 0:
        return float(_reward_model.predict_proba(_build_vec(teamA, teamB)))
    return 1.0 - float(_reward_model.predict_proba(_build_vec(teamA, teamB)))


def _build_vec(teamA: List[int], teamB: List[int]) -> np.ndarray:
    vec = np.zeros(len(_hero_names), dtype=np.float32)
    if teamA:
        vec[teamA] = 1.0
    if teamB:
        vec[teamB] = -1.0
    return vec


def _apply_action(state: DraftState, hero_idx: int):
    if not state.available[hero_idx]:
        raise ValueError("Hero already used")
    if state.step >= TOTAL_STEPS:
        raise ValueError("Draft is complete")

    side = side_to_move(state.step)
    if is_pick_step(state.step):
        if side == 0:
            state.team_radiant.append(hero_idx)
        else:
            state.team_dire.append(hero_idx)
    else:
        state.bans.append(hero_idx)

    state.available[hero_idx] = False
    state.step += 1


def _model_choose(state: DraftState) -> int:
    if _draft_ctx is None:
        raise RuntimeError("Model context not loaded")
    action, _ = _choose_random_action_with_trace(
        state.available,
        state.team_radiant,
        state.team_dire,
        state.step,
        _draft_ctx,
        DRAFT_TOP_K,
    )
    return int(action)


def _turn_label(state: DraftState) -> Dict[str, Optional[str]]:
    if state.step >= TOTAL_STEPS:
        return {"turn_side": None, "turn_type": None}
    return {
        "turn_side": side_to_move(state.step),
        "turn_type": "ban" if is_ban_step(state.step) else "pick",
    }


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/hero_names")
def hero_names():
    _ensure_loaded()
    return jsonify(_hero_names)


@app.route("/api/sequence")
def sequence():
    seq = [
        {
            "step": step,
            "side": side_to_move(step),
            "type": "ban" if is_ban_step(step) else "pick",
        }
        for step in range(TOTAL_STEPS)
    ]
    return jsonify(seq)


@app.route("/api/start", methods=["POST"])
def start():
    data = request.get_json(force=True)
    user_side = int(data.get("user_side", 0))
    if user_side not in (0, 1):
        return jsonify({"error": "user_side must be 0 (radiant) or 1 (dire)"}), 400
    global _state
    _state = _new_state(user_side)
    while _state.step < TOTAL_STEPS and side_to_move(_state.step) != _state.user_side:
        idx = _model_choose(_state)
        _apply_action(_state, idx)
    return jsonify(_serialize_state(_state))


@app.route("/api/pick", methods=["POST"])
def pick():
    if _state is None:
        return jsonify({"error": "Draft not started"}), 400
    data = request.get_json(force=True)
    hero_name = str(data.get("hero_name", "")).strip()
    hero_idx = _name_to_idx.get(hero_name.lower())
    if hero_idx is None:
        return jsonify({"error": "Unknown hero name"}), 400

    if side_to_move(_state.step) != _state.user_side:
        return jsonify({"error": "Not user turn"}), 400

    _apply_action(_state, hero_idx)
    while _state.step < TOTAL_STEPS and side_to_move(_state.step) != _state.user_side:
        idx = _model_choose(_state)
        _apply_action(_state, idx)
    return jsonify(_serialize_state(_state))


def _serialize_state(state: DraftState) -> Dict:
    win_rate = _current_win_rate(state)
    turn_info = _turn_label(state)
    return {
        **asdict(state),
        **turn_info,
        "win_rate": win_rate,
        "hero_names": _hero_names,
    }


if __name__ == "__main__":
    _ensure_loaded()
    host = "127.0.0.1"
    port = 5000
    print(f"Draft UI running at http://{host}:{port}")
    app.run(debug=True, host=host, port=port)
