# Fragment Assembly as Goal-Reaching: A Reinforcement Learning Approach to Targeted Molecular Design

Please see [Paper.pdf](Paper.pdf) for the write-up.

Code for assembling molecules from BRICS-compatible fragments toward a target vector of molecular properties. The implementation includes fingerprint and graph neural network policies, A2C and PPO training with hindsight experience replay, successor features, and training-free planning methods.

## Source layout

```text
env/                  # Assembly actions, descriptors, rewards, and replay
models/               # Fingerprint, GNN, and successor-feature models
training/             # A2C, PPO, GNN wrappers, and successor-feature training
eval/                 # Policy evaluation, planners, and visualization
scripts/              # Demonstration and runtime path setup
paths.py              # Configurable local data/output defaults
```

See [architecture](docs/architecture.md) and [data formats](docs/data-format.md).

## Local setup

Python 3.11 is recommended. From this repository's root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Separately supplied chemistry dependency

Obtain the small-molecule modification toolkit separately. Its root must contain `chem/`, `core/`, and `utils/`. Point to it explicitly:

```bash
export MFA_TOOLKIT_DIR=/absolute/path/to/small-molecule-mod-toolkit
```

The sibling `small-molecule-mod-toolkit` folder is also discovered automatically. Keep it outside this repository. 

### Data and outputs

Supply `fragments.parquet` and `parents.parquet` using the schemas in [data-format.md](docs/data-format.md). These files are not included.

```bash
export MFA_DATA_DIR=/absolute/path/to/dataset
export MFA_RESULTS_DIR=/absolute/path/to/output
```

Defaults are `data/` and `artifacts/` at the repository root. Both are excluded from Git. The main demo, training, and evaluation commands also accept `--fragments_parquet` and `--parents_parquet` overrides.

## Entry points

The following commands illustrate the interface once the external toolkit and data are supplied.

```bash
# Random-policy demo; images go to artifacts/demo by default
python scripts/demo.py --seed 42 --n_frags 50 --steps 4

# Fingerprint PPO
python training/train_ppo.py --device cpu --seed 1

# Fingerprint A2C; GNN wrappers and successor-feature training are adjacent
python training/a2c.py --device cpu --seed 1

# Fingerprint policy evaluation with random and immediate-termination baselines
python eval/evaluate.py --device cpu \
  --checkpoint artifacts/checkpoints_ppo/ckpt_ep10000.pt

# Training-free additive-property planner and quotient-state beam search
python eval/fast_planner.py --help
python eval/qtabs.py --help
```

Use `--help` for the full options. Training defaults filter fragments with `--min_frag_count 5000`; lower threshold explicitly for a small dataset. W&B tracking is optional (`requirements-optional.txt` and `--use_wandb`).

## Scope and attribution

The included demo and policy evaluators do not cover every checkpoint/model variant. In particular, `eval/evaluate.py` and `eval/checkpoints.py` use fingerprint actors. Learned and training-free approaches should be evaluated separately under matched data, target, and compute settings. No pretrained 
weights, results, or reproduction are included in this copy.

## License

Copyright 2026 Asmani Yamin.

The source code is licensed under the Apache License, Version 2.0. 

You may use, modify, and distribute the code, including for commercial purposes, subject to the license terms. When distributing the code or derivative works, include the license, preserve applicable copyright and attribution notices (including [NOTICE](NOTICE)), and identify modified files as required by section 4.

If you use this code or derivatives in research, please cite *Fragment Assembly as Goal-Reaching: A Reinforcement Learning Approach to Targeted Molecular Design* by Asmani Yamin. See [CITATION.cff](CITATION.cff) for citation metadata and [Paper.pdf](Paper.pdf) for the write-up.
