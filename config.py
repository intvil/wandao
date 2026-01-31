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
FM_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "fm_model.pt")

# Data cache directory and files
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
DATA_RAW_CACHE = os.path.join(
    DATA_CACHE_DIR, "pub_matches_raw.fea"
)  # Raw match data with match_id
DATA_PATH = os.path.join(
    DATA_CACHE_DIR, "pub_matches_draft.fea"
)  # Encoded feature data
LINEUP_PATH = os.path.join(DATA_CACHE_DIR, "pub_lineups.fea")  # Per-side lineup data
POSITION_PROBS_PATH = os.path.join(DATA_CACHE_DIR, "hero_position_probs.csv")
POSITION_PROBS_NAMED_PATH = os.path.join(
    DATA_CACHE_DIR, "hero_position_probs_named.csv"
)
FM_LINEAR_WEIGHTS_PATH = os.path.join(DATA_CACHE_DIR, "fm_linear_weights_named.csv")
FM_FACTORS_PATH = os.path.join(DATA_CACHE_DIR, "fm_factors_named.csv")

# Position probability sparsification (applies to FM training + inference)
POSITION_PROB_SPARSE_THRESHOLD = 0.2

# Draft step count (derived from actual ban/pick sequence in policy.py)
TOTAL_STEPS = 24

# Data fetch defaults
FETCH_BATCHES = 1000

# EM training defaults
EM_MAX_LINEUPS = 0
EM_MAX_ITERATIONS = 200
EM_CONVERGENCE_THRESHOLD = 1e-4
EM_EARLY_STOP_PATIENCE = 5
EM_EARLY_STOP_MIN_DELTA = 1e-4
EM_NUM_WORKERS = 0
EM_VALID_FRACTION = 0.1
EM_SEED = 42

# FM training defaults
FM_LATENT_DIM = 8
FM_BATCH_SIZE = 1024
FM_EPOCHS = 10
FM_LR = 1e-3
FM_L2_LAMBDA0 = 1e-4
FM_EARLY_STOP_PATIENCE = 3
FM_EARLY_STOP_MIN_DELTA = 1e-4
FM_GRID_LATENT_DIMS = [FM_LATENT_DIM, 16]
FM_GRID_LRS = [FM_LR, 5e-4]
FM_GRID_L2_LAMBDAS = [FM_L2_LAMBDA0, 5e-4]
FM_GRID_NUM_WORKERS = 0

# Draft search defaults
DRAFT_TOP_K = 5
DRAFT_BATCH_EVAL = True
DRAFT_MP_EVAL = False
DRAFT_MP_WORKERS = 4
DRAFT_EVAL_CACHE_SIZE = 20000

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# API endpoints
OPENDOTA_HEROES_URL = "https://api.opendota.com/api/heroes"
OPENDOTA_MATCHES_URL = "https://api.opendota.com/api/publicMatches"

# API key (loaded from .env file or environment variable)
OPENDOTA_API_KEY = os.environ.get("OPENDOTA_API_KEY", None)

# Role labels anchored by pivotal heroes (used for inspecting position probs)
PIVOTAL_HERO_ROLES = {
    "anti-mage": "carry",
    "storm spirit": "mid",
    "axe": "offlane",
    "warlock": "hardsup",
}
