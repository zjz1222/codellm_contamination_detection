# Data

`clone/{family}/input/{lang}/` contains frozen training/evaluation code and generated variants. `label=1` denotes a member-source sample; `label=0` denotes a nonmember-source sample. Variant labels are `original`, `Type I`, `Type II`, and `Type III` (also stored as `Type_III`).

`features/clone/{family}/input_processed/{lang}/` contains cached token features. Files ending in `.pkl` are Parquet, with numeric feature columns, `target`, and `clone_type`.

`manifest.json` records retained artifacts and hashes. Private paths, identifiers, email addresses and credential-like literals in code samples are redacted without changing row order or labels. Numeric feature caches retain historical values; re-extraction from sanitized text can therefore differ. See `../THIRD_PARTY.md` for sources.
