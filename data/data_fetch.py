"""
Data fetching utilities for downloading match data from OpenDota API with caching support.
"""

import os
import requests
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from wandao.config import (
    OPENDOTA_HEROES_URL,
    OPENDOTA_MATCHES_URL,
    DATA_CACHE_DIR,
    DATA_RAW_CACHE,
    DATA_PATH,
    OPENDOTA_API_KEY,
    LINEUP_PATH,
)
from wandao.data.encoding import encode_df, encode_lineups


def request_data(less_than_match_id=None, min_rank=80, api_key=None):
    """Request a batch of public matches from OpenDota API (up to 100 matches per call).

    Args:
        less_than_match_id: Optional match ID to paginate from
        min_rank: Minimum rank tier for matches (0-80, where 80 is Immortal)
        api_key: Optional OpenDota API key (improves rate limits)

    Returns:
        DataFrame with match data
    """
    params = {"min_rank": min_rank}
    if less_than_match_id:
        params["less_than_match_id"] = less_than_match_id

    # Add API key if provided (either via parameter or config)
    key = api_key or OPENDOTA_API_KEY
    if key:
        params["api_key"] = key

    response = requests.get(OPENDOTA_MATCHES_URL, params=params, timeout=30)
    if response.status_code == 200:
        data = response.json()
    else:
        print("Error:", response.status_code, response.text)
        return pd.DataFrame()

    df = pd.DataFrame(data).query("duration > 0")
    if not df.empty and "start_time" in df.columns:
        df = df.assign(start_time=lambda x: pd.to_datetime(x["start_time"], unit="s"))
    return df


def get_hero_ids():
    """Fetch list of valid hero IDs from OpenDota API.

    Returns:
        List of hero IDs
    """
    try:
        resp = requests.get(OPENDOTA_HEROES_URL, timeout=10)
        resp.raise_for_status()
        heroes = resp.json()
        return [h["id"] for h in heroes]
    except Exception as e:
        print(f"Failed to fetch hero list: {e}")
        return None


def load_cached_data():
    """Load cached RAW match data if it exists.

    Returns:
        DataFrame with cached raw match data, or empty DataFrame if no cache exists
    """
    if os.path.exists(DATA_RAW_CACHE):
        try:
            df = pd.read_feather(DATA_RAW_CACHE)
            print(f"Loaded {len(df)} cached matches from {DATA_RAW_CACHE}")
            return df
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return pd.DataFrame()
    else:
        print("No cached raw data found")
        return pd.DataFrame()


def save_cache(df):
    """Save RAW match data to cache.

    Args:
        df: DataFrame with raw match data (must have match_id column)
    """
    # Ensure cache directory exists
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    try:
        df.to_feather(DATA_RAW_CACHE)
        print(f"Saved {len(df)} matches to cache: {DATA_RAW_CACHE}")
    except Exception as e:
        print(f"Failed to save cache: {e}")


def construct_dataset(n=100, min_rank=80, feature_names=None, api_key=None):
    """Fetch n batches of matches from OpenDota API with caching.

    This function implements smart caching:
    1. Loads existing cached data
    2. Requests new matches from the API
    3. Stops when overlap with cache is detected (match IDs are descending)
    4. Concatenates new and cached data
    5. Updates the cache

    Args:
        n: Number of batches to fetch (max)
        min_rank: Minimum rank tier for matches
        feature_names: Optional list of valid feature names for filtering
        api_key: Optional OpenDota API key (improves rate limits)

    Returns:
        DataFrame with raw match data (combined new + cached)
    """
    # Load cached data
    cached_df = load_cached_data()

    # Get valid hero IDs
    hero_ids = get_hero_ids()
    if hero_ids is None and feature_names:
        hero_ids = list(map(int, feature_names))

    # If we have cached data, get the maximum match ID
    max_cached_match_id = None
    if not cached_df.empty and "match_id" in cached_df.columns:
        max_cached_match_id = cached_df["match_id"].max()
        print(f"Maximum cached match ID: {max_cached_match_id}")

    # Display API key status
    key = api_key or OPENDOTA_API_KEY
    if key:
        print("Using OpenDota API key (improves rate limits)")

    # Fetch new matches
    less_than_match_id = None
    new_matches = []
    iterator = tqdm(range(n), desc="Fetching new matches") if tqdm else range(n)

    for i in iterator:
        df = request_data(less_than_match_id, min_rank=min_rank, api_key=api_key)
        if df.empty:
            print(f"No more matches returned at iteration {i+1}")
            break

        # Check if we've reached the cached data
        if max_cached_match_id is not None and "match_id" in df.columns:
            min_fetched_match_id = df["match_id"].min()

            # If the minimum ID in this batch is <= max cached ID, we've overlapped
            if min_fetched_match_id <= max_cached_match_id:
                print(f"Reached cached data at iteration {i+1}")
                print(f"  Min fetched ID: {min_fetched_match_id}")
                print(f"  Max cached ID: {max_cached_match_id}")

                # Only keep matches that are newer than cached data
                df = df[df["match_id"] > max_cached_match_id]
                if not df.empty:
                    new_matches.append(df)
                    print(f"  Kept {len(df)} new matches from this batch")
                break

        new_matches.append(df)
        less_than_match_id = df["match_id"].min()

    # Combine new and cached data
    if new_matches:
        new_df = pd.concat(new_matches, ignore_index=True)
        print(f"Fetched {len(new_df)} new matches")

        # Filter new matches for valid heroes
        if hero_ids:
            before_filter = len(new_df)
            new_df = new_df[
                new_df["radiant_team"].apply(lambda x: set(x).issubset(set(hero_ids)))
            ]
            print(
                f"Filtered to {len(new_df)} matches with valid heroes (removed {before_filter - len(new_df)})"
            )

        # Combine with cached data
        if not cached_df.empty:
            combined_df = pd.concat([new_df, cached_df], ignore_index=True)
            # Remove duplicates based on match_id
            if "match_id" in combined_df.columns:
                before_dedup = len(combined_df)
                combined_df = combined_df.drop_duplicates(
                    subset=["match_id"], keep="first"
                )
                print(f"Removed {before_dedup - len(combined_df)} duplicate matches")
        else:
            combined_df = new_df

        # Sort by match_id descending (newest first)
        if "match_id" in combined_df.columns:
            combined_df = combined_df.sort_values(
                "match_id", ascending=False
            ).reset_index(drop=True)

        # Save to cache
        save_cache(combined_df)

        print(f"Total: {len(combined_df)} matches in dataset")
        return combined_df
    else:
        print("No new matches fetched, using cached data")
        if not cached_df.empty:
            print(f"Total: {len(cached_df)} matches in dataset")
        return cached_df


def get_encoded_data(feature_names, force_refresh=False):
    """Get encoded feature data from cache or create it from raw cache.

    Args:
        feature_names: List of feature names (hero IDs as strings)
        force_refresh: If True, always re-encode from raw cache

    Returns:
        DataFrame with encoded features and radiant_win column
    """
    # Check if encoded cache exists and is not forced to refresh
    if not force_refresh and os.path.exists(DATA_PATH):
        try:
            df = pd.read_feather(DATA_PATH)
            if "radiant_win" in df.columns:
                print(f"Loaded {len(df)} encoded matches from cache: {DATA_PATH}")
                return df
        except Exception as e:
            print(f"Failed to load encoded cache: {e}")

    # Load raw cache and encode it
    raw_df = load_cached_data()
    if raw_df.empty:
        print("No raw data available to encode")
        return pd.DataFrame()

    print(f"Encoding {len(raw_df)} matches from raw cache...")
    encoded_df = encode_df(raw_df, feature_names)

    # Save encoded cache
    try:
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        encoded_df.to_feather(DATA_PATH)
        print(f"Saved encoded data to: {DATA_PATH}")
    except Exception as e:
        print(f"Failed to save encoded cache: {e}")

    return encoded_df


def save_lineup_cache(df):
    """Save per-side lineup data to cache."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    try:
        df.to_feather(LINEUP_PATH)
        print(f"Saved {len(df)} lineups to cache: {LINEUP_PATH}")
    except Exception as e:
        print(f"Failed to save lineup cache: {e}")


def get_lineup_data(feature_names, force_refresh=False, raw_df=None):
    """Get per-side lineup data (two lineups per match) from cache or create it."""
    if not force_refresh and os.path.exists(LINEUP_PATH):
        try:
            df = pd.read_feather(LINEUP_PATH)
            if "side" in df.columns and "win" in df.columns:
                print(f"Loaded {len(df)} lineups from cache: {LINEUP_PATH}")
                return df
        except Exception as e:
            print(f"Failed to load lineup cache: {e}")

    base_df = raw_df if raw_df is not None else load_cached_data()
    if base_df.empty:
        print("No raw data available to encode lineups")
        return pd.DataFrame()

    print(f"Encoding {len(base_df)} matches into per-side lineups...")
    lineup_df = encode_lineups(base_df, feature_names)
    save_lineup_cache(lineup_df)
    return lineup_df
