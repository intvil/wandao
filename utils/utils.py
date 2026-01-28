"""
Utility functions for hero name mapping and feature loading.
"""

import json
import os
import requests

from wandao.config import (
    DATA_CACHE_DIR,
    FEATURES_PATH,
    OPENDOTA_HEROES_URL,
    OPENDOTA_API_KEY,
    PIVOTAL_HERO_ROLES,
)


def load_feature_names():
    """Load feature names (hero IDs) from JSON file.

    Returns:
        List of feature names as strings
    """
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature names file: {FEATURES_PATH}")

    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    return feature_names


def fetch_feature_names(api_key=None):
    """Fetch hero IDs from OpenDota API.

    Returns:
        List of hero IDs as strings (sorted ascending)
    """
    params = {}
    key = api_key or OPENDOTA_API_KEY
    if key:
        params["api_key"] = key
    resp = requests.get(OPENDOTA_HEROES_URL, params=params, timeout=10)
    resp.raise_for_status()
    heroes = resp.json()
    hero_ids = sorted({int(h["id"]) for h in heroes if "id" in h})
    if not hero_ids:
        raise ValueError("OpenDota heroes response missing ids")
    return [str(hid) for hid in hero_ids]


def ensure_feature_names(force_refresh=False, api_key=None):
    """Ensure feature_names.json exists; optionally refresh from OpenDota."""
    if not force_refresh and os.path.exists(FEATURES_PATH):
        return load_feature_names()
    feature_names = fetch_feature_names(api_key=api_key)
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)
    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)
    print(f"Saved feature names to {FEATURES_PATH} ({len(feature_names)} heroes)")
    return feature_names


def _load_hero_names_cache(cache_path):
    with open(cache_path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Hero names cache is not a dictionary")
    return {int(hid): name for hid, name in raw.items() if name}


def fetch_hero_names(force_refresh=False):
    """Fetch hero metadata from OpenDota API (cached on disk).

    Returns:
        Dictionary mapping hero ID (int) to localized name (str)
    """
    cache_path = os.path.join(DATA_CACHE_DIR, "hero_names.json")
    cached = None
    if not force_refresh and os.path.exists(cache_path):
        try:
            cached = _load_hero_names_cache(cache_path)
            if cached:
                return cached
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"Warning: failed to read hero names cache ({exc}); refetching.")

    params = {}
    try:
        resp = requests.get(OPENDOTA_HEROES_URL, params=params, timeout=10)
        resp.raise_for_status()
        heroes = resp.json()
    except requests.RequestException as exc:
        if cached is not None:
            print(
                f"Warning: hero name fetch failed ({exc}); using cache at {cache_path}."
            )
            return cached
        raise

    id_to_name = {
        int(h["id"]): h.get("localized_name") or h.get("name")
        for h in heroes
        if "id" in h
    }
    if not id_to_name:
        if cached is not None:
            print("Warning: hero name fetch returned no ids; using cached names.")
            return cached
        raise ValueError("OpenDota heroes response missing ids")

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump({str(hid): name for hid, name in id_to_name.items()}, f)
    except OSError as exc:
        print(f"Warning: failed to write hero names cache ({exc}).")

    return id_to_name


def _normalize_hero_name(name: str) -> str:
    return name.strip().lower()


def resolve_position_role_mapping(pos_df, id_to_name=None, pivotal_hero_roles=None):
    """Map position columns to role names based on pivotal heroes."""
    from wandao.models.factorization_machine import ROLE_COLUMNS

    missing = [c for c in ROLE_COLUMNS if c not in pos_df.columns]
    if missing:
        raise ValueError(f"Position probabilities missing columns: {missing}")

    pivotal_hero_roles = pivotal_hero_roles or PIVOTAL_HERO_ROLES
    id_to_name = id_to_name or fetch_hero_names()
    name_to_id = {
        _normalize_hero_name(name): int(hid) for hid, name in id_to_name.items()
    }
    role_to_position = {}
    used_positions = set()
    for hero_name, role in pivotal_hero_roles.items():
        hero_id = name_to_id.get(_normalize_hero_name(hero_name))
        if hero_id is None:
            raise ValueError(f"Pivotal hero not found in hero names: {hero_name}")
        if "Hero_ID" in pos_df.columns:
            hero_rows = pos_df.loc[pos_df["Hero_ID"] == hero_id, ROLE_COLUMNS]
        else:
            hero_rows = (
                pos_df.loc[[hero_id], ROLE_COLUMNS]
                if hero_id in pos_df.index
                else pos_df.iloc[0:0]
            )
        if hero_rows.empty:
            raise ValueError(f"Pivotal hero missing in position probs: {hero_name}")
        position = hero_rows.iloc[0].idxmax()
        if position in used_positions:
            raise ValueError(
                f"Position {position} assigned to multiple pivotal heroes"
            )
        role_to_position[role] = position
        used_positions.add(position)
    remaining_positions = [c for c in ROLE_COLUMNS if c not in used_positions]
    if len(remaining_positions) != 1:
        raise ValueError(
            "Expected exactly one remaining position for softsup, "
            f"got {remaining_positions}"
        )
    role_to_position["softsup"] = remaining_positions[0]
    return {pos: role for role, pos in role_to_position.items()}


def idx_to_hero_name(idx, feature_names, id_to_name):
    """Convert internal feature index to hero name.

    Args:
        idx: Internal feature index
        feature_names: List of feature names (hero IDs as strings)
        id_to_name: Dictionary mapping hero ID to name

    Returns:
        Hero name as string
    """
    try:
        hid = int(feature_names[idx])
    except Exception:
        return f"hero_{idx}"
    return id_to_name.get(hid, str(hid))
