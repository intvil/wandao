"""
Draft search utilities (expectimax + FM evaluator).
"""

import multiprocessing as mp
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from wandao.config import (
    TOTAL_STEPS,
    DRAFT_TOP_K,
    DRAFT_BATCH_EVAL,
    DRAFT_MP_EVAL,
    DRAFT_MP_WORKERS,
    DRAFT_EVAL_CACHE_SIZE,
)
from wandao.models.policy import side_to_move, is_ban_step, is_pick_step
from wandao.data.encoding import build_feature_vector


@dataclass
class _EvalContext:
    reward_model: object
    n_champs: int
    use_batch: bool
    use_mp: bool
    mp_workers: int
    cache_size: int
    cache: OrderedDict


_MP_CONTEXT = None
_RNG = np.random.default_rng()


def _mp_init(reward_model, n_champs, use_batch, cache_size):
    global _MP_CONTEXT
    _MP_CONTEXT = _EvalContext(
        reward_model=reward_model,
        n_champs=n_champs,
        use_batch=use_batch,
        use_mp=False,
        mp_workers=0,
        cache_size=cache_size,
        cache=OrderedDict(),
    )


def _eval_key(teamA, teamB, n_champs):
    key = np.zeros(n_champs, dtype=np.int8)
    if teamA:
        key[teamA] = 1
    if teamB:
        key[teamB] = -1
    return key.tobytes()


def _cache_get(cache, key):
    value = cache.get(key)
    if value is None:
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache, key, value, cache_size):
    if cache_size <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > cache_size:
        cache.popitem(last=False)


def _evaluate_state(teamA, teamB, ctx: _EvalContext):
    key = _eval_key(teamA, teamB, ctx.n_champs)
    cached = _cache_get(ctx.cache, key)
    if cached is not None:
        p = cached
    else:
        vec = build_feature_vector(teamA, teamB, ctx.n_champs)
        p = float(ctx.reward_model.predict_proba(vec))
        _cache_put(ctx.cache, key, p, ctx.cache_size)
    return p


def _score_pick_candidates_batch(
    hero_indices, teamA, teamB, ctx: _EvalContext, pick_side
):
    vecs = np.zeros((len(hero_indices), ctx.n_champs), dtype=np.float32)
    for row, hero_idx in enumerate(hero_indices):
        if teamA:
            vecs[row, teamA] = 1.0
        if teamB:
            vecs[row, teamB] = -1.0
        if pick_side == 0:
            vecs[row, hero_idx] = 1.0
        else:
            vecs[row, hero_idx] = -1.0
    scores = ctx.reward_model.predict_proba_batch(vecs)
    if pick_side == 1:
        scores = 1.0 - scores
    return scores


def _score_pick_candidate_single(hero_idx, teamA, teamB, ctx: _EvalContext, pick_side):
    if pick_side == 0:
        vec = build_feature_vector(teamA + [hero_idx], teamB, ctx.n_champs)
        return float(ctx.reward_model.predict_proba(vec))
    vec = build_feature_vector(teamA, teamB + [hero_idx], ctx.n_champs)
    return 1.0 - float(ctx.reward_model.predict_proba(vec))


def _score_pick_candidates(hero_indices, teamA, teamB, ctx: _EvalContext, pick_side):
    if ctx.use_batch:
        return _score_pick_candidates_batch(hero_indices, teamA, teamB, ctx, pick_side)
    scores = np.empty(len(hero_indices), dtype=np.float32)
    for i, idx in enumerate(hero_indices):
        scores[i] = _score_pick_candidate_single(idx, teamA, teamB, ctx, pick_side)
    return scores


def _mp_score_candidates(args):
    hero_indices, teamA, teamB, pick_side = args
    return _score_pick_candidates(hero_indices, teamA, teamB, _MP_CONTEXT, pick_side)


@torch.no_grad()
def _next_pick_side(t):
    for step in range(t, TOTAL_STEPS):
        if is_pick_step(step):
            return side_to_move(step)
    return side_to_move(TOTAL_STEPS - 1)


def _top_k_candidates(available, teamA, teamB, ctx: _EvalContext, t, k):
    available_indices = [i for i, ok in enumerate(available) if ok]
    if len(available_indices) <= k:
        return available_indices
    pick_side = _next_pick_side(t)
    if ctx.use_mp and len(available_indices) > ctx.mp_workers:
        chunk = max(1, len(available_indices) // ctx.mp_workers)
        batches = [
            available_indices[i : i + chunk]
            for i in range(0, len(available_indices), chunk)
        ]
        with mp.Pool(
            processes=ctx.mp_workers,
            initializer=_mp_init,
            initargs=(
                ctx.reward_model,
                ctx.n_champs,
                ctx.use_batch,
                ctx.cache_size,
            ),
        ) as pool:
            results = pool.map(
                _mp_score_candidates,
                [(b, teamA, teamB, pick_side) for b in batches],
            )
        scores = np.concatenate(results, axis=0)
        scored = list(zip(scores.tolist(), available_indices))
    else:
        scores = _score_pick_candidates(available_indices, teamA, teamB, ctx, pick_side)
        scored = list(zip(scores.tolist(), available_indices))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [idx for _, idx in scored[:k]]


def _rank_candidates_with_scores(available, teamA, teamB, ctx: _EvalContext, t, k):
    available_indices = [i for i, ok in enumerate(available) if ok]
    pick_side = _next_pick_side(t)
    if ctx.use_mp and len(available_indices) > ctx.mp_workers:
        chunk = max(1, len(available_indices) // ctx.mp_workers)
        batches = [
            available_indices[i : i + chunk]
            for i in range(0, len(available_indices), chunk)
        ]
        with mp.Pool(
            processes=ctx.mp_workers,
            initializer=_mp_init,
            initargs=(
                ctx.reward_model,
                ctx.n_champs,
                ctx.use_batch,
                ctx.cache_size,
            ),
        ) as pool:
            results = pool.map(
                _mp_score_candidates,
                [(b, teamA, teamB, pick_side) for b in batches],
            )
        scores = np.concatenate(results, axis=0)
    else:
        scores = _score_pick_candidates(available_indices, teamA, teamB, ctx, pick_side)
    scored = list(zip(scores.tolist(), available_indices))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:k]


def _apply_action(available, teamA, teamB, t, action):
    next_available = available.copy()
    next_available[action] = False
    next_teamA = teamA
    next_teamB = teamB
    side = side_to_move(t)
    if is_pick_step(t):
        if side == 0:
            next_teamA = teamA + [action]
        else:
            next_teamB = teamB + [action]
    return next_available, next_teamA, next_teamB


def _expectimax_value(
    available,
    teamA,
    teamB,
    t,
    ctx: _EvalContext,
    top_k,
    last_action_was_ban,
    opponent_side,
    opp_picks_seen,
    root_side,
):
    if t >= TOTAL_STEPS:
        return _evaluate_state(teamA, teamB, ctx)
    if opp_picks_seen >= 1:
        return _evaluate_state(teamA, teamB, ctx)

    candidates = _top_k_candidates(available, teamA, teamB, ctx, t, top_k)
    if not candidates:
        return _evaluate_state(teamA, teamB, ctx)

    values = []
    for action in candidates:
        next_available, next_teamA, next_teamB = _apply_action(
            available, teamA, teamB, t, action
        )
        next_opp_picks = opp_picks_seen
        if is_pick_step(t) and side_to_move(t) == opponent_side:
            next_opp_picks += 1
        values.append(
            _expectimax_value(
                next_available,
                next_teamA,
                next_teamB,
                t + 1,
                ctx,
                top_k,
                is_ban_step(t),
                opponent_side,
                next_opp_picks,
                root_side,
            )
        )
    return sum(values) / len(values)


def _choose_random_action_with_trace(
    available,
    teamA,
    teamB,
    t,
    ctx: _EvalContext,
    top_k,
):
    root_side = side_to_move(t)
    opponent_side = 1 - root_side
    ranked = _rank_candidates_with_scores(available, teamA, teamB, ctx, t, top_k)
    candidates = [idx for _, idx in ranked]
    if not candidates:
        raise ValueError("No available actions to choose from.")
    expected = {}
    for action in candidates:
        next_available, next_teamA, next_teamB = _apply_action(
            available, teamA, teamB, t, action
        )
        opp_picks_seen = (
            1 if is_pick_step(t) and side_to_move(t) == opponent_side else 0
        )
        expected[action] = _expectimax_value(
            next_available,
            next_teamA,
            next_teamB,
            t + 1,
            ctx,
            top_k,
            is_ban_step(t),
            opponent_side,
            opp_picks_seen,
            root_side,
        )
    ranked_with_expected = [(score, expected[idx], idx) for score, idx in ranked]
    return int(_RNG.choice(candidates)), ranked_with_expected


@torch.no_grad()
def greedy_draft(
    reward_model,
    n_champs,
    top_k=DRAFT_TOP_K,
):
    """Execute an expectimax draft using the FM reward model."""
    ctx = _EvalContext(
        reward_model=reward_model,
        n_champs=n_champs,
        use_batch=DRAFT_BATCH_EVAL,
        use_mp=DRAFT_MP_EVAL,
        mp_workers=max(1, DRAFT_MP_WORKERS),
        cache_size=DRAFT_EVAL_CACHE_SIZE,
        cache=OrderedDict(),
    )
    available = [True] * n_champs
    bans = []
    teamA, teamB = [], []
    steps = []

    for t in range(TOTAL_STEPS):
        action, ranked = _choose_random_action_with_trace(
            available, teamA, teamB, t, ctx, top_k
        )
        steps.append(
            {
                "step": t,
                "side": side_to_move(t),
                "action_type": "ban" if is_ban_step(t) else "pick",
                "ranked_candidates": ranked,
                "chosen": action,
                "choice_reason": "random_top_k_expected",
            }
        )
        available[action] = False
        if is_ban_step(t):
            bans.append(action)
        else:
            if side_to_move(t) == 0:
                teamA.append(action)
            else:
                teamB.append(action)

    p = _evaluate_state(teamA, teamB, ctx)
    return bans, teamA, teamB, p, steps
