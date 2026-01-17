"""
Data encoding utilities for converting match data to feature vectors.
"""

import pandas as pd
import numpy as np


def encode_df(df, feature_names):
    """Encode matches into feature vectors: +1 for radiant heroes, -1 for dire heroes.

    Adds a mirrored sample for each match (swap sides and flip result) to balance side advantage.

    Args:
        df: DataFrame with raw match data containing radiant_team, dire_team, radiant_win columns
        feature_names: List of hero IDs as strings

    Returns:
        DataFrame with encoded features and radiant_win column
    """
    hero_ids = [int(fid) for fid in feature_names]

    lineup_matrix = []
    for _, row in df.iterrows():
        lineup = {hid: 0 for hid in hero_ids}
        for hid in row["radiant_team"]:
            if hid in lineup:
                lineup[hid] = 1
        for hid in row["dire_team"]:
            if hid in lineup:
                lineup[hid] = -1
        lineup_matrix.append(lineup)

    res = pd.DataFrame(lineup_matrix)
    res["radiant_win"] = df["radiant_win"].astype(float).values

    # Mirror data: swap sides (flip signs) and flip outcome
    mirrored = res.copy()
    mirrored[hero_ids] = -mirrored[hero_ids]
    mirrored["radiant_win"] = 1.0 - mirrored["radiant_win"]

    return pd.concat([res, mirrored], ignore_index=True)


def encode_lineups(df, feature_names):
    """Encode each match into two per-side lineup vectors (radiant and dire).

    Each lineup row is 1 for heroes picked on that side, 0 otherwise.
    """
    hero_ids = [int(fid) for fid in feature_names]
    records = []

    for _, row in df.iterrows():
        for team in (row["radiant_team"], row["dire_team"]):
            lineup = {hid: 0 for hid in hero_ids}
            for hid in team:
                if hid in lineup:
                    lineup[hid] = 1
            records.append(lineup)

    return pd.DataFrame.from_records(records)


def build_feature_vector(teamA, teamB, n_champs):
    """Build feature vector for a single draft.

    For each hero feature, value is +1 if in teamA, -1 if in teamB, else 0.

    Args:
        teamA: List of hero indices in team A
        teamB: List of hero indices in team B
        n_champs: Total number of champions/heroes

    Returns:
        numpy array of shape (n_champs,)
    """
    v = np.zeros(n_champs, dtype=np.float32)
    for idx in teamA:
        if 0 <= idx < n_champs:
            v[idx] = 1.0
    for idx in teamB:
        if 0 <= idx < n_champs:
            v[idx] = -1.0
    return v
