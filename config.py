"""
Configuration settings for the draft module.
"""

import os
import torch

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Load from current directory (project root)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

# Base directory - current directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
FEATURES_PATH = os.path.join(MODEL_CACHE_DIR, "feature_names.json")
REWARD_MODEL_JOBLIB = os.path.join(MODEL_CACHE_DIR, "reward_model.joblib")
REWARD_MODEL_TXT = os.path.join(MODEL_CACHE_DIR, "reward_model.txt")
REWARD_MODEL_MLP = os.path.join(MODEL_CACHE_DIR, "reward_model_mlp.pt")
REWARD_MODEL_META = os.path.join(MODEL_CACHE_DIR, "reward_model_meta.json")
POLICY_PATH = os.path.join(MODEL_CACHE_DIR, "draft_policy.pt")
CLUSTER_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "cluster_model.pkl")

# Data cache directory and files
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
DATA_RAW_CACHE = os.path.join(
    DATA_CACHE_DIR, "pub_matches_raw.fea"
)  # Raw match data with match_id
DATA_PATH = os.path.join(
    DATA_CACHE_DIR, "pub_matches_draft.fea"
)  # Encoded feature data
LINEUP_PATH = os.path.join(DATA_CACHE_DIR, "pub_lineups.fea")  # Per-side lineup data

# Draft step count (derived from actual ban/pick sequence in policy.py)
TOTAL_STEPS = 24

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# API endpoints
OPENDOTA_HEROES_URL = "https://api.opendota.com/api/heroes"
OPENDOTA_MATCHES_URL = "https://api.opendota.com/api/publicMatches"

# API key (loaded from .env file or environment variable)
OPENDOTA_API_KEY = os.environ.get("OPENDOTA_API_KEY", None)
