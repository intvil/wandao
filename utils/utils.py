"""
Utility functions for hero name mapping and feature loading.
"""

import json
import os
import requests

from wandao.config import FEATURES_PATH, OPENDOTA_HEROES_URL, OPENDOTA_API_KEY


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


def fetch_hero_names():
    """Fetch hero metadata from OpenDota API.

    Returns:
        Dictionary mapping hero ID (int) to localized name (str)
    """
    try:
        resp = requests.get(OPENDOTA_HEROES_URL, timeout=10)
        resp.raise_for_status()
        heroes = resp.json()
        id_to_name = {
            int(h["id"]): h.get("localized_name") or h.get("name") for h in heroes
        }
        return id_to_name
    except Exception:
        return {}


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
