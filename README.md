# CodeMIA

Official implementation of **Syntax-Aware Detection of Training Data Contamination in Code LLMs**, accepted to **Findings of EMNLP 2026**. The main experiment compares contamination detection on Original, Type-I, Type-II, and Type-III code in Python and C++.

## Installation

Python 3.10+. Clone the repository, then run commands from its root:

```bash
git clone https://github.com/zjz1222/codellm_contamination_detection.git
cd codellm_contamination_detection
```

Code and data archives are also available on the [Releases page](https://github.com/zjz1222/codellm_contamination_detection/releases).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
```

## Run the main experiment

Processed datasets and cached token features are included. No LLM weights or GPU are needed to train and evaluate the classifier:

```bash
# Table 1: starcoderbase, Python and C++
python scripts/run.py clone --family starcoder --model starcoderbase --lang python
python scripts/run.py clone --family starcoder --model starcoderbase --lang cpp
```

Each command samples 1,000 training examples, trains LightGBM, and reports AUC, balanced accuracy, and TPR at FPR < 1% for each code group and their macro average. Results and fitted models are saved to `outputs/`. An existing fitted model is reused; choose a new `--output-dir` for a fresh fit.

Other models are listed in `configs/models.json`. For example:

```bash
python scripts/run.py clone --family magicoder --model Magicoder-CL-7B --lang python --output-dir outputs/new-run
```

## Extract features and run baselines

Install the inference dependencies and place target model weights under `$CODEMIA_MODEL_ROOT/<model-name>`:

```bash
pip install -r requirements-inference.txt
export CODEMIA_MODEL_ROOT=/path/to/models
python scripts/run.py features --family starcoder --model starcoderbase --lang python
python scripts/run.py baselines --family starcoder --model starcoderbase --lang python
python scripts/run.py synprune --family starcoder --model starcoderbase --lang python
```

These commands require compatible GPU hardware. Feature extraction saves to `outputs/features/`; pass that directory with `clone --feature-dir outputs/features` to train on newly extracted features. Baselines include PPL, Zlib, Min-K%, Min-K%++, and ReCaLL; SynPrune supports Python.

## Files

- `paper.pdf`: original paper.
- `src/`: feature extraction, LightGBM, and baseline implementations.
- `data/clone/` and `data/features/clone/`: processed code and cached features for eight models and two languages. Historical `.pkl` files use Parquet format.
- `results/legacy/clone/`: historical full-precision measurements.
- [Table 1 check](docs/TABLE1_CHECK.md): paper comparison, local rerun, and reproduction details.

```bash
python scripts/check_release.py
python -m unittest discover -s tests -v
python scripts/package_release.py --output-dir dist
```

Data sources and redistribution notes are in [THIRD_PARTY.md](THIRD_PARTY.md). A code license has not been selected.
