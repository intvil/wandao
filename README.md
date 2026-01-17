# Dota 2 Draft Agent

RL-based draft agent using an XGBoost reward model.

## Quick Start

```bash
# Install dependencies
pip install torch xgboost pandas numpy scikit-learn requests tqdm python-dotenv
# Optional (for clustering + human-likeness shaping)
pip install umap-learn hdbscan

# Set up API key (optional, improves speed)
cp .env.example .env
# Edit .env and add your OpenDota API key

# Show help (no action without flags)
python3 -m wandao.cli.run_draft

# Fetch new data
python3 -m wandao.cli.run_draft --fetch-data

# Train the XGBoost reward model (uses cached data)
python3 -m wandao.cli.run_draft --train-reward

# Train the policy
python3 -m wandao.cli.run_draft --train-policy

# Run a sample draft with saved models
python3 -m wandao.cli.run_draft --sample-draft --load-policy
```

Feature names are stored in `model_cache/feature_names.json` and will be fetched from OpenDota on first run (use `--refresh-features` to force refresh).

## Command Line Options

```
--fetch-data              Fetch new match data from OpenDota API
--n-batches N             Number of batches to fetch (default: 1000)
--reward-backend {mlp,xgb} Reward model backend (default: xgb)
--train-reward           Train the reward model
--reward-rounds N        XGBoost boosting rounds (default: 400, only for xgb backend)
--train-policy           Train draft policy via RL
--load-policy            Load saved policy before training/sampling
--policy-iters N         Policy training iterations (default: 2000)
--batch-episodes N       Episodes per batch (default: 32)
--sample-draft           Run a sample greedy draft with saved models
--encode-lineups         Encode cached matches into per-side lineups (saves to data_cache/pub_lineups.fea)
--use-cluster-reward     Add human-likeness shaping (UMAP+HDBSCAN) to policy reward
--cluster-weight F       Weight for negative distance to human cluster centers (default: 0.5)
--cluster-components N   UMAP output dimensions for cluster model (default: 5)
--cluster-model-path P   Path to load/save cluster model (default: models/cluster_model.pkl)
--cluster-rebuild        Force refit of cluster model even if a saved one exists
--refresh-features       Force refresh hero feature_names.json from OpenDota API
--fit-clusters           Fit/save cluster model and exit (no training)
```

## Examples

```bash
# Fetch and train only the XGBoost model
python3 -m wandao.cli.run_draft --fetch-data --train-reward

# Train XGBoost with more data
python3 -m wandao.cli.run_draft --fetch-data --train-reward --n-batches 200

# Train policy with custom iterations
python3 -m wandao.cli.run_draft --train-policy --policy-iters 2000 --batch-episodes 64

# Full pipeline from scratch
python3 -m wandao.cli.run_draft --fetch-data --train-reward --n-batches 200 --train-policy --policy-iters 2000

# Export per-side lineups for clustering
python3 -m wandao.cli.run_draft --encode-lineups

# Train policy with human-likeness shaping (UMAP+HDBSCAN)
python3 -m wandao.cli.run_draft --train-policy --use-cluster-reward --cluster-weight 0.5

# Fit and save cluster model without training (from main entry point)
python3 -m wandao.cli.run_draft --fit-clusters
```

## Project Structure

```
wandao/
├── cli/                 # Entrypoints
│   └── run_draft.py
├── config.py            # Configuration and paths
├── data/                # Data fetching/encoding
│   ├── data_fetch.py
│   └── encoding.py
├── models/              # Model definitions and cluster fitting code
│   ├── policy.py
│   ├── reward_model.py
│   └── cluster_model.py
├── training/            # RL training utilities
│   └── training.py
├── utils/               # Helpers (feature names, hero names)
│   └── utils.py
├── pipelines/           # Orchestration
│   └── main.py
├── model_cache/         # Saved model artifacts (feature names, reward/policy/cluster)
├── data_cache/          # Cached match and encoded data
└── README.md
```

Saved model artifacts (feature names, reward model, policy, cluster) are stored under `model_cache/` per paths defined in `config.py`.

## Features

- Smart data caching with overlap detection
- XGBoost win prediction model
- Policy gradient RL for draft strategy
- OpenDota API integration
- Configurable training parameters
