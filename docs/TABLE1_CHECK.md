# Table 1 check

Only Table 1 on page 7 of `paper.pdf` is checked. The original PDF is unchanged. Values below are percentages; differences are percentage points.

The experiment structure matches: `starcoderbase`, Python/C++, Original/Type-I/Type-II/Type-III, AUC/ACC/TPR@1%, and an equal-weight macro average. Both local classifier runs were freshly fitted from the bundled numeric features. AUC and ACC are close to the paper; some low-FPR TPR values differ more.

## Ours: paper / local fresh fit

| Language | Group | AUC | ACC | TPR@1% |
| --- | --- | --- | --- | --- |
| Python | Original | 76.5 / 76.30 | 70.7 / 70.70 | 1.6 / 1.60 |
| Python | Type-I | 74.6 / 74.32 | 68.2 / 67.95 | 1.9 / 3.10 |
| Python | Type-II | 74.6 / 73.94 | 68.4 / 67.70 | 1.4 / 2.60 |
| Python | Type-III | 73.6 / 73.95 | 68.4 / 69.40 | 2.4 / 2.20 |
| Python | Macro Average | 74.8 / 74.63 | 68.3 / 68.94 | 2.0 / 2.38 |
| C++ | Original | 73.7 / 72.85 | 69.2 / 68.10 | 6.8 / 2.20 |
| C++ | Type-I | 72.5 / 71.40 | 67.4 / 65.97 | 2.6 / 2.53 |
| C++ | Type-II | 72.3 / 70.67 | 67.1 / 65.70 | 2.9 / 3.60 |
| C++ | Type-III | 71.9 / 70.47 | 67.1 / 65.00 | 4.8 / 3.60 |
| C++ | Macro Average | 72.8 / 71.35 | 67.6 / 66.19 | 2.6 / 2.98 |

Local macro AUC differs by 0.17 points for Python and 1.45 points for C++. Every local group AUC exceeds the corresponding historical baseline group AUC. The largest TPR discrepancy is C++ Original: 6.8% in the paper versus 2.2% locally. The fresh fits reproduce the retained historical Ours files at their stored precision.

## Baselines

Compared all 225 baseline metric cells against the historical `starcoderbase_full_results.txt` files and Python `starcoderbase_synprune_result.txt`. PPL, Zlib, Min-10/20%, Min-10/20%++, and ReCaLL differ by at most 0.1 points after rounding. Python SynPrune group cells agree within rounding; its stored `AVG` is a pooled result (65.92 / 63.08 / 2.10), whereas the paper reports a macro average (66.1 / 63.4 / 1.8). The arithmetic mean of its stored group metrics is 66.09 / 63.43 / 1.80, agreeing with the paper. Historical records support these comparisons; full GPU inference was not rerun.

## Reproduction details

- Each feature split has 4,000 rows, with 1,000 per code group and 500 examples per label per group. Training samples 1,000 rows with seed 42. Evaluation covers all 4,000 rows.
- The retained LightGBM configuration uses 50 leaves, learning rate 0.1, L1 0.2, L2 5, path smooth 2, extra trees, up to 500 rounds and 50-round early stopping. The same evaluation set is also used for early stopping. The paper describes three repetitions; the supplied main script performs one seeded fit.
- ACC maximizes balanced accuracy over ROC thresholds; TPR uses strict FPR < 0.01. Numeric caches remain unchanged after text anonymization.
- Printed paper macro cells are not all arithmetic means of the printed groups: Python Ours ACC averages to 68.93 rather than 68.3; C++ Ours TPR averages to 4.28 rather than 2.6. The paper and stored results were not edited to force agreement.
- The lean launcher routes GPU baselines to clone validation data, normalizes Type_III, and reports a metric macro average. Scoring functions are retained; existing short-sequence Min-K++, ReCaLL context/masking and SynPrune indexing behavior are unchanged. These GPU paths have group/aggregation checks, not a full inference rerun.
- Feature extraction keeps the original token-probability and syntax-feature functions, processes every row, and writes to a separate feature directory. Re-extraction can differ because code text was anonymized.

Run commands are in the root README. Cell-level comparison and fit metadata are in `results/validation/table1_check.json`.
