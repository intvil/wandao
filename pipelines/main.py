"""
Main entry point for training and running the draft agent.
"""

import argparse
import itertools
import math
from wandao.config import device, DATA_PATH, FETCH_BATCHES
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
from wandao.search.expectimax import greedy_draft
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
  
    # Run a sample draft using FM + expectimax
    python3 -m wandao.cli.run_draft --sample-draft
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

    # Sampling
    parser.add_argument(
        "--sample-draft",
        action="store_true",
        help="Run a sample draft using FM + expectimax",
    )

    args = parser.parse_args()
    feature_names = ensure_feature_names(force_refresh=args.refresh_features)

    # If no action flags are provided, show help and exit
    if not any(
        [
            args.fetch_data,
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

    # Sample draft if requested
    if args.sample_draft:
        if reward_model is None:
            reward_model = RewardModel(feature_names)
            if not reward_model.load():
                raise RuntimeError(
                    "No trained reward model found. Train or load it before sampling."
                )

        # Fetch hero names for display
        id_to_name = fetch_hero_names()
        if id_to_name:
            print(f"Fetched {len(id_to_name)} hero names from OpenDota")
        else:
            print(
                "Warning: failed to fetch hero names from OpenDota; falling back to ids"
            )

        bans, a, b, p, steps = greedy_draft(reward_model, N_CHAMPS)

        print("\n=== Draft steps ===")
        for entry in steps:
            step_idx = entry["step"] + 1
            side = "A" if entry["side"] == 0 else "B"
            action_type = entry["action_type"]
            ranked = entry["ranked_candidates"]
            chosen = entry["chosen"]
            print(f"Step {step_idx:02d}: Team {side} {action_type}")
            print("  Top candidates (FM score / expected value):")
            for score, expected, idx in ranked:
                name = idx_to_hero_name(idx, feature_names, id_to_name)
                print(f"    - {name}: {score:.3f} / {expected:.3f}")
            chosen_name = idx_to_hero_name(chosen, feature_names, id_to_name)
            reason = entry.get("choice_reason", "best_value")
            print(f"  Chosen {action_type}: {chosen_name} ({reason})")

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
