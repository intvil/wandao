"""
Training utilities for the draft policy using self-play RL.
"""

import os
import math
import torch

from wandao.config import device, TOTAL_STEPS, POLICY_PATH
from wandao.models.policy import (
    side_to_move,
    is_ban_step,
    is_pick_step,
    build_state,
    masked_categorical_sample,
)
from wandao.data.encoding import build_feature_vector


def train_self_play(
    policy,
    reward_model,
    n_champs,
    iters=1200,
    batch_episodes=32,
    cluster_model=None,
    cluster_weight=0.0,
):
    """Train policy via self-play with policy gradient.

    Args:
        policy: DraftPolicy network
        reward_model: RewardModel for computing rewards
        n_champs: Number of champions/heroes
        iters: Number of training iterations
        batch_episodes: Number of episodes per batch
        cluster_model: Optional ClusterModel for human-likeness shaping
        cluster_weight: Weight for cluster proximity reward (negative distance)
    """
    opt = torch.optim.Adam(policy.parameters(), lr=2e-4)
    baseline = 0.0

    for it in range(iters):
        losses = []
        rewards = []

        for _ in range(batch_episodes):
            available = torch.ones(n_champs, device=device, dtype=torch.bool)
            pickA = torch.zeros(n_champs, device=device)
            pickB = torch.zeros(n_champs, device=device)

            teamA, teamB = [], []
            logps = []
            signs = []

            for t in range(TOTAL_STEPS):
                side = side_to_move(t)
                s = build_state(available.float(), pickA, pickB, t, side)
                logits = policy(s)

                mask_detached = available.clone().detach()
                a, logp = masked_categorical_sample(logits, mask_detached)
                available[a] = False

                if is_pick_step(t):
                    if side == 0:
                        teamA.append(a)
                        pickA[a] = 1.0
                    else:
                        teamB.append(a)
                        pickB[a] = 1.0

                logps.append(logp)
                signs.append(+1.0 if side == 0 else -1.0)

            # Reward from XGBoost reward model
            vec = build_feature_vector(teamA, teamB, n_champs)
            p = reward_model.predict_proba(vec)
            p_clamped = min(max(p, 1e-6), 1 - 1e-6)
            total_reward = math.log(p_clamped) - math.log1p(-p_clamped)  # log-odds

            # Optional shaping toward human-like clusters (teamA only)
            if cluster_model is not None and cluster_weight != 0.0:
                dist = cluster_model.distance_from_team(teamA)
                total_reward = total_reward + cluster_weight * (-dist)

            R = torch.tensor(total_reward, device=device, dtype=torch.float32)

            # Policy gradient loss with baseline
            adv = (R - baseline).detach()
            ep_loss = 0.0
            for logp, sign in zip(logps, signs):
                ep_loss = ep_loss + (-logp * adv * sign)

            losses.append(ep_loss)
            rewards.append(float(R.item()))

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        meanR = sum(rewards) / len(rewards)
        baseline = 0.95 * baseline + 0.05 * meanR

        if (it + 1) % 200 == 0:
            print(f"RL iter {it+1}, mean reward={meanR:.3f}, baseline={baseline:.3f}")


@torch.no_grad()
def greedy_draft(policy, reward_model, n_champs):
    """Execute a greedy draft using the trained policy."""
    available = torch.ones(n_champs, device=device, dtype=torch.bool)
    pickA = torch.zeros(n_champs, device=device)
    pickB = torch.zeros(n_champs, device=device)
    bans = []

    teamA, teamB = [], []
    for t in range(TOTAL_STEPS):
        side = side_to_move(t)
        s = build_state(available.float(), pickA, pickB, t, side)
        logits = policy(s)
        masked = logits.clone()
        masked[~available] = -1e9
        a = int(torch.argmax(masked).item())

        available[a] = False
        if is_ban_step(t):
            bans.append(a)
        else:
            if side == 0:
                teamA.append(a)
                pickA[a] = 1.0
            else:
                teamB.append(a)
                pickB[a] = 1.0

    vec = build_feature_vector(teamA, teamB, n_champs)
    p = float(reward_model.predict_proba(vec))
    return bans, teamA, teamB, p


def save_policy(policy, n_champs, path=POLICY_PATH):
    """Save policy checkpoint to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "n_champs": n_champs,
            "total_steps": TOTAL_STEPS,
        },
        path,
    )
    print(f"Saved policy to {path}")


def load_policy(n_champs, path=POLICY_PATH):
    """Load policy checkpoint from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy checkpoint not found at {path}")

    checkpoint = torch.load(path, map_location=device)
    ck_n = checkpoint.get("n_champs")
    ck_steps = checkpoint.get("total_steps")
    if ck_n is not None and ck_n != n_champs:
        raise ValueError(
            f"Checkpoint n_champs {ck_n} does not match current {n_champs}"
        )
    if ck_steps is not None and ck_steps != TOTAL_STEPS:
        raise ValueError(
            f"Checkpoint total_steps {ck_steps} does not match current {TOTAL_STEPS}"
        )

    from wandao.models.policy import (
        DraftPolicy,
    )  # Local import to avoid circular at module load

    policy = DraftPolicy(n_champs, TOTAL_STEPS).to(device)
    policy.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded policy from {path}")
    return policy
