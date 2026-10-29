# Architecture

The environment builds a molecular graph by attaching compatible BRICS fragments to open attachment sites. A policy can choose an attachment action or terminate an episode. Rewards compare achieved molecular properties with a normalized goal vector.

- `env/env.py` defines molecular state, compatible attachment actions, transitions, and termination.
- `env/properties.py`, `features.py`, and `rewards.py` define descriptors, encodings, normalization, and reward variants.
- `models/base.py` implements fingerprint actor/critic networks; `models/gnn.py` encodes graph structure.
- `training/a2c.py` and `training/ppo.py` train actor/critic policies. Replay and hindsight relabeling are in `env/replay.py`.
- `models/successor.py` and `training/sf.py` implement successor-feature learning.
- `env/fast_properties.py` estimates properties from fragment contributions and calibrated bond corrections. `eval/fast_planner.py` uses this model for greedy planning.
- `eval/qtabs.py` implements quotient-state tensor beam search. This is a training-free planning method.

The external small-molecule toolkit provides the `MolGraph` structures and chemistry operations imported through `core.*`, `chem.*`, and `utils.*`. It is intentionally not
in this repository. Set `MFA_TOOLKIT_DIR` when obtaining it separately.
