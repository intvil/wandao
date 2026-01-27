#!/usr/bin/env python3
"""
Inspect EM position probabilities and FM linear weights.
"""

import os
import sys

import pandas as pd
import torch

try:
    from wandao.config import (
        FM_MODEL_PATH,
        POSITION_PROBS_PATH,
        POSITION_PROBS_NAMED_PATH,
        FM_LINEAR_WEIGHTS_PATH,
    )
    from wandao.models.factorization_machine import load_fm_data, ROLE_COLUMNS
    from wandao.utils.utils import fetch_hero_names
except ImportError:
    # Allow running as a script from repo root without installing the package
    PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    PROJECT_ROOT = os.path.abspath(os.path.join(PKG_ROOT, ".."))
    for path in (PROJECT_ROOT,):
        if path not in sys.path:
            sys.path.insert(0, path)
    from wandao.config import (
        FM_MODEL_PATH,
        POSITION_PROBS_PATH,
        POSITION_PROBS_NAMED_PATH,
        FM_LINEAR_WEIGHTS_PATH,
    )
    from wandao.models.factorization_machine import load_fm_data, ROLE_COLUMNS
    from wandao.utils.utils import fetch_hero_names


def _load_hero_names():
    id_to_name = fetch_hero_names()
    return {int(k): v for k, v in id_to_name.items()}


def export_position_probs():
    pos_path = POSITION_PROBS_PATH
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"Position probabilities not found at {pos_path}")
    df = pd.read_csv(pos_path)
    if "Hero_ID" not in df.columns:
        raise ValueError("Position probabilities missing Hero_ID column")

    id_to_name = _load_hero_names()
    df.insert(
        1,
        "Hero_Name",
        df["Hero_ID"].map(lambda hid: id_to_name.get(int(hid), str(hid))),
    )
    out_path = POSITION_PROBS_NAMED_PATH
    df.to_csv(out_path, index=False)
    print(f"Saved position probabilities to {out_path}")
    return df


def export_fm_linear_weights():
    if not os.path.exists(FM_MODEL_PATH):
        raise FileNotFoundError(f"FM model not found at {FM_MODEL_PATH}")
    ckpt = torch.load(FM_MODEL_PATH, map_location="cpu")
    _, _, _, hero_ids = load_fm_data()
    id_to_name = _load_hero_names()
    hero_labels = [id_to_name.get(hid, str(hid)) for hid in hero_ids]

    linear = ckpt["state_dict"]["linear"].detach().cpu().numpy()
    weights = linear.reshape(len(hero_ids), len(ROLE_COLUMNS))
    df = pd.DataFrame(weights, index=hero_labels, columns=ROLE_COLUMNS)
    out_path = FM_LINEAR_WEIGHTS_PATH
    df.to_csv(out_path, index_label="Hero_Name")
    print(f"Saved FM linear weights to {out_path}")
    return df


def main():
    print("=== Exporting EM position probabilities ===")
    export_position_probs()
    print("=== Exporting FM linear weights ===")
    export_fm_linear_weights()


if __name__ == "__main__":
    main()
