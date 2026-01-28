"""
Main entry point for training and running the draft agent.
"""

import argparse
import itertools
import math
from wandao.config import (
    device,
    TOTAL_STEPS,
    DATA_PATH,
    FETCH_BATCHES,
    POLICY_ITERS,
    POLICY_BATCH_EPISODES,
)
from wandao.utils.utils import (
    ensure_feature_names,
    fetch_hero_names,
    idx_to_hero_name,
    resolve_position_role_mapping,
)
from wandao.models.reward_model import RewardModel
from wandao.models.factorization_machine import (
    load_position_probs,
    train_fm,
    ROLE_COLUMNS,
)
from wandao.models.position_em import train_em
from wandao.models.policy import DraftPolicy
from wandao.training.training import (
    train_self_play,
    greedy_draft,
    save_policy,
    load_policy,
)
from wandao.data.data_fetch import construct_dataset, get_lineup_data
from wandao.data.encoding import encode_df


def _infer_team_positions(team_indices, feature_names, id_to_name, pos_df):
    hero_ids = []
    for idx in team_indices:
        try:
            hero_ids.append(int(feature_names[idx]))
        except Exception as exc:
            raise ValueError(f"Invalid hero index in draft: {idx}") from exc
    missing = [hid for hid in hero_ids if hid not in pos_df.index]
    if missing:
        preview = ", ".join(str(h) for h in missing[:10])
        raise ValueError(
            "Position probabilities missing hero ids: "
            f"{len(missing)} (e.g., {preview})"
        )
    position_to_role = resolve_position_role_mapping(pos_df, id_to_name=id_to_name)
    probs = pos_df.loc[hero_ids, ROLE_COLUMNS].to_numpy(dtype=float)
    best_perm = None
    best_log = -float("inf")
    for perm in itertools.permutations(range(len(ROLE_COLUMNS))):
        logp = 0.0
        for row_idx, pos in enumerate(perm):
            logp += math.log(probs[row_idx, pos] + 1e-12)
        if logp > best_log:
            best_log = logp
            best_perm = perm
    return [
        f"{id_to_name.get(hid, str(hid))} ({position_to_role[ROLE_COLUMNS[pos_idx]]})"
        for hid, pos_idx in zip(hero_ids, best_perm)
    ]


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Dota 2 Draft Agent Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch new data
  python3 -m wandao.cli.run_draft --fetch-data
  
  # Encode lineups (after fetch)
  python3 -m wandao.cli.run_draft --encode-lineups
  
  # Train position EM model (after encoding lineups)
  python3 -m wandao.cli.run_draft --train-em
  
  # Train factorization machine (after EM position probs)
  python3 -m wandao.cli.run_draft --train-fm
  
  # Train policy (requires FM model)
  python3 -m wandao.cli.run_draft --train-policy
  
  # Run a sample draft using saved models
  python3 -m wandao.cli.run_draft --sample-draft --load-policy
        """,
    )

    # Data fetching
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch new match data from OpenDota API (see config.py for batches)",
    )

    # Encoding / lineups
    parser.add_argument(
        "--encode-lineups",
        action="store_true",
        help="Encode cached matches into per-side lineups for EM training",
    )

    # Feature names
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Force refresh hero feature_names.json from OpenDota API",
    )

    # Encoding / EM / FM
    parser.add_argument(
        "--train-em",
        action="store_true",
        help="Train EM position model and save hero_position_probs.csv",
    )
    parser.add_argument(
        "--train-fm",
        action="store_true",
        help="Train factorization machine on lineup+position features",
    )

    # Policy training
    parser.add_argument(
        "--train-policy", action="store_true", help="Train the draft policy via RL"
    )
    parser.add_argument(
        "--load-policy",
        action="store_true",
        help="Load saved draft policy instead of training from scratch",
    )

    # Sampling
    parser.add_argument(
        "--sample-draft",
        action="store_true",
        help="Run a sample greedy draft with the current policy",
    )

    args = parser.parse_args()
    feature_names = ensure_feature_names(force_refresh=args.refresh_features)

    # If no action flags are provided, show help and exit
    if not any(
        [
            args.fetch_data,
            args.train_policy,
            args.sample_draft,
            args.encode_lineups,
            args.refresh_features,
            args.train_fm,
            args.train_em,
        ]
    ):
        parser.print_help()
        return

    print("device:", device)

    # Load feature names (needed for all steps except a no-op help call)
    N_CHAMPS = len(feature_names)
    print(f"Loaded {N_CHAMPS} features (heroes) from feature_names.json")

    reward_model = None
    policy = None
    raw_df = None

    # Fetch data if requested
    if args.fetch_data:
        print(f"\n=== Fetching match data (batches={FETCH_BATCHES}) ===")
        raw_df = construct_dataset(n=FETCH_BATCHES, feature_names=feature_names)
        if raw_df.empty:
            print("No data fetched; aborting fetch step.")
        else:
            df = encode_df(raw_df, feature_names)
            df.to_feather(DATA_PATH)
            print(f"Saved encoded dataset to {DATA_PATH} ({len(df)} rows)")

    # Encode per-side lineups for EM
    if args.encode_lineups:
        lineups_df = get_lineup_data(
            feature_names, force_refresh=args.fetch_data, raw_df=raw_df
        )
        if not lineups_df.empty:
            print(f"Prepared {len(lineups_df)} per-side lineups")

    # Train EM position model if requested
    if args.train_em:
        print("\n=== Training EM position model ===")
        train_em()

    # Train factorization machine if requested
    if args.train_fm:
        print("\n=== Training factorization machine ===")
        train_fm()

    # Policy training if requested
    if args.train_policy:
        if reward_model is None:
            reward_model = RewardModel(feature_names)
            if not reward_model.load():
                raise RuntimeError(
                    "No trained reward model found. Run with --train-fm first."
                )
        if args.load_policy:
            try:
                policy = load_policy(N_CHAMPS)
            except Exception as e:
                print(f"Failed to load saved policy, training from scratch: {e}")
        if policy is None:
            policy = DraftPolicy(N_CHAMPS, TOTAL_STEPS).to(device)
        print(
            f"\n=== Training draft policy via self-play ({POLICY_ITERS} iterations, "
            f"batch_episodes={POLICY_BATCH_EPISODES}) ==="
        )
        train_self_play(
            policy,
            reward_model,
            N_CHAMPS,
            iters=POLICY_ITERS,
            batch_episodes=POLICY_BATCH_EPISODES,
        )
        save_policy(policy, N_CHAMPS)

    # Sample draft if requested
    if args.sample_draft:
        if reward_model is None:
            reward_model = RewardModel(feature_names)
            if not reward_model.load():
                raise RuntimeError(
                    "No trained reward model found. Train or load it before sampling."
                )
        if policy is None:
            try:
                policy = load_policy(N_CHAMPS)
            except Exception as e:
                raise RuntimeError(
                    "No trained policy available; train or load one before sampling."
                ) from e

        # Fetch hero names for display
        id_to_name = fetch_hero_names()
        if id_to_name:
            print(f"Fetched {len(id_to_name)} hero names from OpenDota")
        else:
            print(
                "Warning: failed to fetch hero names from OpenDota; falling back to ids"
            )

        bans, a, b, p = greedy_draft(policy, reward_model, N_CHAMPS)

        # Map indices to hero names for readability
        bans_names = [idx_to_hero_name(i, feature_names, id_to_name) for i in bans]
        a_names = [idx_to_hero_name(i, feature_names, id_to_name) for i in a]
        b_names = [idx_to_hero_name(i, feature_names, id_to_name) for i in b]

        print("\n=== Sample greedy draft ===")
        print("bans:", bans_names)
        print("teamA picks:", a_names)
        print("teamB picks:", b_names)
        print("predicted P(A wins):", round(p, 3))

        pos_df = load_position_probs()
        print(
            "teamA positions:",
            _infer_team_positions(a, feature_names, id_to_name, pos_df),
        )
        print(
            "teamB positions:",
            _infer_team_positions(b, feature_names, id_to_name, pos_df),
        )


if __name__ == "__main__":
    main()
