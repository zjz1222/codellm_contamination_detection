# Data sources

| Family | Member source |
| --- | --- |
| Magicoder | `ise-uiuc/Magicoder-OSS-Instruct-75K` |
| OctoPack | `bigcode/commitpackft` |
| OpenCoder | `OpenCoder-LLM/opencoder-sft-stage1` |
| StarCoder | `bigcode/starcoderdata`, associated with The Stack |

Nonmember examples originate from GitHub; clone variants were generated from the selected code. Model weights and upstream raw corpora are acquired separately. Model identifiers are listed in `configs/models.json`.

Source labels identify member/nonmember pools, rather than independently proving inclusion or exclusion in a model's training data. Historical snapshots, per-file licenses, and clone parent identifiers are not fully available. Upstream repository and dataset terms continue to apply. Names and contact information in imported comments are redacted; the supplied paper is preserved unchanged.

This release keeps the main clone-detection pipeline. Auxiliary studies, unused baselines, duplicate result formats, and historical checkpoints are omitted. Cached numeric features are preserved; re-extracting features from sanitized code may change measurements. Historical failed generations and duplicate/conflicting examples remain in the frozen main dataset.

No project code license has been selected.
