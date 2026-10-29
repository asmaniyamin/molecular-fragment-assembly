# Input data

## `fragments.parquet`

The fragment loader reads these four columns:

| Column | Meaning |
|---|---|
| `frag_smiles` | BRICS fragment SMILES with labeled dummy attachment atoms, such as `[3*]OCC` |
| `count` | Numeric occurrence count used for filtering and ranking |
| `attach_labels` | Comma-separated attachment label identifiers, such as `3,16` |
| `n_attach` | Number of dummy attachment atoms |

Only rows with at least one attachment and `count >= min_frag_count` are retained. The loader keeps the highest-frequency requested number of fragments. The environment normalizes BRICS isotope-style dummy labels to its internal representation.

## `parents.parquet`

Targets can be supplied as raw descriptor columns matching `--goal_properties`; the default columns are `sLogP`, `QED`, and `TPSA`.

Alternatively, provide a SMILES column so missing descriptors can be recomputed. Recognized names include `smiles`, `canonical_smiles`, `parent_smiles`, `SMILES`, `parent_smiles_canon`, `parent_smiles_mapped`, and `frag_smiles`.

Descriptors are normalized with fixed bounds in `env/properties.py`, including sLogP [-5, 10], QED [0, 1], and TPSA [0, 200]. Input descriptor columns must contain raw values, not already normalized values.
