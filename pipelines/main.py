"""
Main entry point for training and running the draft agent.
"""

import argparse
import os
from wandao.config import device, TOTAL_STEPS, DATA_PATH, CLUSTER_MODEL_PATH
from wandao.utils.utils import ensure_feature_names, fetch_hero_names, idx_to_hero_name
from wandao.models.reward_model import RewardModel
from wandao.models.policy import DraftPolicy
from wandao.training.training import (
    train_self_play,
    greedy_draft,
    save_policy,
    load_policy,
)
from wandao.data.data_fetch import construct_dataset, get_lineup_data
from wandao.data.encoding import encode_df
from wandao.models.cluster_model import (
    fit_cluster_model,
    load_cluster_model,
    save_cluster_model,
)


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
  
  # Fit cluster model (after encoding)
  python3 -m wandao.cli.run_draft --fit-clusters
  
  # Train reward model (uses cached/previously fetched data)
  python3 -m wandao.cli.run_draft --train-reward
  
  # Train policy (requires reward model)
  python3 -m wandao.cli.run_draft --train-policy
  
  # Run a sample draft using saved models
  python3 -m wandao.cli.run_draft --sample-draft --load-policy
        """,
    )

    # Data fetching
    parser.add_argument(
        "--fetch-data",
        action="store_true",
        help="Fetch new match data from OpenDota API",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1000,
        help="Number of batches to fetch (default: 1000, ~100k matches)",
    )

    # Encoding / lineups
    parser.add_argument(
        "--encode-lineups",
        action="store_true",
        help="Encode cached matches into per-side lineups for analysis/clustering",
    )

    # Cluster fit/load and shaping
    parser.add_argument(
        "--fit-clusters",
        action="store_true",
        help="Fit/save cluster model and exit (no training)",
    )
    parser.add_argument(
        "--cluster-model-path",
        default=CLUSTER_MODEL_PATH,
        help="Path to load/save cluster model (default: model_cache/cluster_model.pkl)",
    )
    parser.add_argument(
        "--cluster-rebuild",
        action="store_true",
        help="Force refit of cluster model even if a saved one exists",
    )
    parser.add_argument(
        "--cluster-components",
        type=int,
        default=5,
        help="UMAP components for cluster model",
    )
    parser.add_argument(
        "--use-cluster-reward",
        action="store_true",
        help="Add human-likeness shaping via UMAP+HDBSCAN cluster distance",
    )
    parser.add_argument(
        "--cluster-weight",
        type=float,
        default=0.5,
        help="Weight for negative distance to nearest human cluster center",
    )

    # Feature names
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Force refresh hero feature_names.json from OpenDota API",
    )

    # Reward model training
    parser.add_argument(
        "--reward-backend",
        choices=["mlp", "xgb"],
        default="xgb",
        help="Reward model backend (default: xgb)",
    )
    parser.add_argument(
        "--train-reward", action="store_true", help="Train the reward model"
    )
    parser.add_argument(
        "--train-xgb", action="store_true", dest="train_reward", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--train-lgb", action="store_true", dest="train_reward", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--reward-rounds",
        type=int,
        default=400,
        help="Boosting rounds for XGBoost backend (default: 400)",
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
    parser.add_argument(
        "--policy-iters",
        type=int,
        default=2000,
        help="Policy training iterations (default: 2000)",
    )
    parser.add_argument(
        "--batch-episodes",
        type=int,
        default=32,
        help="Episodes per training batch (default: 32)",
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
            args.train_reward,
            args.train_policy,
            args.sample_draft,
            args.encode_lineups,
            args.fit_clusters,
            args.refresh_features,
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
    cluster_model = None

    # Fetch data if requested
    if args.fetch_data:
        print(f"\n=== Fetching match data (batches={args.n_batches}) ===")
        raw_df = construct_dataset(n=args.n_batches, feature_names=feature_names)
        if raw_df.empty:
            print("No data fetched; aborting fetch step.")
        else:
            df = encode_df(raw_df, feature_names)
            df.to_feather(DATA_PATH)
            print(f"Saved encoded dataset to {DATA_PATH} ({len(df)} rows)")

    # Encode per-side lineups for clustering/analysis
    if args.encode_lineups:
        lineups_df = get_lineup_data(
            feature_names, force_refresh=args.fetch_data, raw_df=raw_df
        )
        if not lineups_df.empty:
            print(f"Prepared {len(lineups_df)} per-side lineups")

    # Fit cluster model without training if requested
    if args.fit_clusters:
        lineups_df = get_lineup_data(
            feature_names, force_refresh=args.fetch_data, raw_df=raw_df
        )
        if lineups_df.empty:
            raise RuntimeError(
                "Cannot build cluster model without lineup data. Run with --encode-lineups and data available."
            )
        if (
            (not args.cluster_rebuild)
            and args.cluster_model_path
            and os.path.exists(args.cluster_model_path)
        ):
            try:
                cluster_model = load_cluster_model(args.cluster_model_path)
                print(
                    f"\nCluster model already exists at {args.cluster_model_path}; use --cluster-rebuild to refit."
                )
            except Exception:
                print("Existing cluster model load failed; refitting...")
        else:
            print("\n=== Fitting UMAP+HDBSCAN cluster model ===")
            cluster_model = fit_cluster_model(
                lineups_df,
                feature_names,
                components=args.cluster_components,
            )
            if args.cluster_model_path:
                save_cluster_model(cluster_model, args.cluster_model_path)
            print("Cluster model fit complete.")
            # If no further actions requested, exit; otherwise continue and reuse this model

    # Prepare cluster model if requested for training
    if args.use_cluster_reward:
        lineups_df = get_lineup_data(
            feature_names, force_refresh=args.fetch_data, raw_df=raw_df
        )
        if lineups_df.empty:
            raise RuntimeError(
                "Cannot build cluster model without lineup data. Run with --encode-lineups and data available."
            )
        if (
            (not args.cluster_rebuild)
            and args.cluster_model_path
            and os.path.exists(args.cluster_model_path)
        ):
            try:
                cluster_model = load_cluster_model(args.cluster_model_path)
                print(f"\nLoaded cluster model from {args.cluster_model_path}")
            except Exception as e:
                print(f"Failed to load cluster model ({e}), refitting...")

        if cluster_model is None:
            print(
                "\n=== Fitting UMAP+HDBSCAN cluster model for human-likeness shaping ==="
            )
            cluster_model = fit_cluster_model(
                lineups_df,
                feature_names,
                components=args.cluster_components,
            )
            if args.cluster_model_path:
                save_cluster_model(cluster_model, args.cluster_model_path)
            if cluster_model.centers.size == 0:
                print(
                    "Warning: cluster model has no clusters (all noise); cluster reward will be skipped."
                )
                cluster_model = None

    # Train reward model if requested
    if args.train_reward:
        reward_model = RewardModel(feature_names, backend=args.reward_backend)
        print(f"\n=== Training reward model ({args.reward_backend}) ===")
        reward_model.train(
            fetch_data=False,
            n_batches=args.n_batches,
            num_boost_round=args.reward_rounds,
            backend=args.reward_backend,
        )
        print()

    # Policy training if requested
    if args.train_policy:
        if reward_model is None:
            reward_model = RewardModel(feature_names, backend=args.reward_backend)
            if not reward_model.load():
                raise RuntimeError(
                    "No trained reward model found. Run with --train-reward first."
                )
        if args.load_policy:
            try:
                policy = load_policy(N_CHAMPS)
            except Exception as e:
                print(f"Failed to load saved policy, training from scratch: {e}")
        if policy is None:
            policy = DraftPolicy(N_CHAMPS, TOTAL_STEPS).to(device)
        print(
            f"\n=== Training draft policy via self-play ({args.policy_iters} iterations) ==="
        )
        train_self_play(
            policy,
            reward_model,
            N_CHAMPS,
            iters=args.policy_iters,
            batch_episodes=args.batch_episodes,
            cluster_model=cluster_model,
            cluster_weight=args.cluster_weight if cluster_model is not None else 0.0,
        )
        save_policy(policy, N_CHAMPS)

    # Sample draft if requested
    if args.sample_draft:
        if reward_model is None:
            reward_model = RewardModel(feature_names, backend=args.reward_backend)
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


if __name__ == "__main__":
    main()
