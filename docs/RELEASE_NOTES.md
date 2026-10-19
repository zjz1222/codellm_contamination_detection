Code and processed datasets for **Syntax-Aware Detection of Training Data Contamination in Code LLMs**, accepted to **Findings of EMNLP 2026**.

- Main contamination-detection experiment for eight code LLMs, Python/C++, and four code transformation groups.
- Cached features, LightGBM training/evaluation, baseline implementations, and the original paper.
- Private identifiers, machine paths, and credential-like literals sanitized; unused auxiliary studies removed.

The local Table 1 classifier reruns obtain macro AUC 74.63% on Python and 71.35% on C++ (paper: 74.8% and 72.8%). Baseline comparisons use historical measurements; full GPU baseline inference was not rerun. See `docs/TABLE1_CHECK.md` for details.

Assets:

- `codemia-code.tar.gz`: implementation, instructions, and paper.
- `codemia-data.tar.gz`: processed datasets, cached features, and historical results.
- `checksums.json`: SHA-256 hashes and sizes.

Both archives extract into `codemia/`. Install with `pip install -r requirements-core.txt`, then follow the README. Data sources and redistribution notes are in `THIRD_PARTY.md`; a project code license has not been selected.
