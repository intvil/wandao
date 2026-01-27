# Dota 2 Draft Agent

RL-based draft agent using a factorization machine reward model.

## Quick Start

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn requests tqdm python-dotenv

# Set up API key (optional, improves speed)
cp .env.example .env
# Edit .env and add your OpenDota API key

# Show help (no action without flags)
python3 -m wandao.cli.run_draft

# Fetch new data
python3 -m wandao.cli.run_draft --fetch-data

# Encode lineups and train EM position model
python3 -m wandao.cli.run_draft --encode-lineups --train-em

# Train factorization machine reward model
python3 -m wandao.cli.run_draft --train-fm

# Train the policy
python3 -m wandao.cli.run_draft --train-policy

# Run a sample draft with saved models
python3 -m wandao.cli.run_draft --sample-draft --load-policy

```

Feature names are stored in `model_cache/feature_names.json` and will be fetched from OpenDota on first run (use `--refresh-features` to force refresh).
Use `python3 -m wandao.utils.inspect_models` to export named EM position probabilities and FM linear weights into the paths configured in `config.py`.

## Command Line Options

```
--fetch-data              Fetch new match data from OpenDota API (batch count set in config.py)
--encode-lineups         Encode cached matches into per-side lineups (saves to data_cache/pub_lineups.fea)
--train-em               Train EM position model (saves to configured position probs path)
--train-fm               Train factorization machine on lineup+position features
--train-policy           Train draft policy via RL (iterations set in config.py)
--load-policy            Load saved policy before training/sampling
--sample-draft           Run a sample greedy draft with saved models
--refresh-features       Force refresh hero feature_names.json from OpenDota API
```

## Examples

```bash
# Full pipeline from scratch
python3 -m wandao.cli.run_draft --fetch-data --encode-lineups --train-em --train-fm --train-policy
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
├── models/              # Model definitions
│   ├── policy.py
│   ├── reward_model.py
│   └── position_em.py
├── training/            # RL training utilities
│   └── training.py
├── utils/               # Helpers (feature names, hero names)
│   ├── utils.py
│   └── inspect_models.py
├── pipelines/           # Orchestration
│   └── main.py
├── model_cache/         # Saved model artifacts (feature names, reward/policy)
├── data_cache/          # Cached match and encoded data
└── README.md
```

Saved model artifacts (feature names, reward model, policy) are stored under `model_cache/` per paths defined in `config.py`.

## Features

- Smart data caching with overlap detection
- Factorization machine win prediction model
- Policy gradient RL for draft strategy
- OpenDota API integration
- Configurable training parameters
