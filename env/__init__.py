# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: Apache-2.0

from .data import (
    GOAL_DIM, PROP_NAMES, PROP_MIN, PROP_MAX, PROP_RANGE,
    normalize_props, denormalize_props, FragInfo,
    load_fragment_library, load_target_distribution, sample_target,
)
from .properties import (
    DEFAULT_PROPERTY_NAMES,
    PROPERTY_SPECS,
    parse_property_names,
    property_bounds,
    property_indices,
)
from .features import (
    FP_BITS, BRICS_DIM, ACTION_FEAT_DIM, STATE_DIM,
    state_dim,
    molgraph_to_fp, smiles_to_fp, brics_onehot,
    state_features, action_features,
    compute_raw_properties, compute_norm_properties,
)
from .rewards import (
    OPEN_DUMMY_PENALTY,
    DEFAULT_PROPERTY_SURROGATE_CFG,
    parse_reward_vector,
    build_env_reward_config,
    reward_from_context,
)
from .env import MolEnv, Action, TERMINATE, StepResult, _normalise_brics_smiles
from .replay import ReplayBuffer, Episode, Transition
