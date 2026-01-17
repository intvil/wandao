"""
Utility functions for hero name mapping and feature loading.
"""

import json
import os
import requests

from draft.config import FEATURES_PATH, OPENDOTA_HEROES_URL


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
