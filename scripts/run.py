#!/usr/bin/env python3
"""Train/evaluate the main CodeMIA experiment, extract features, or run baselines."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "clone": "trainer/lightgbm_train.py",
    "features": "trainer/extract_features.py",
    "baselines": "methods/eval.py",
    "synprune": "methods/eval_synprune.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--family", required=True, choices=["magicoder", "octopack", "opencoder", "starcoder"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--lang", choices=["python", "cpp"], default="python")
    parser.add_argument("--mode", choices=["combine", "log_prob", "normalized_log_prob"], default="combine")
    parser.add_argument("--output-dir", help="Directory for results and fitted models (default: outputs)")
    parser.add_argument("--feature-dir", help="Feature root with family/input_processed/language subdirectories")
    args = parser.parse_args()
    models = json.loads((ROOT / "configs/models.json").read_text())["families"]
    if args.model not in models[args.family]:
        parser.error("--model must belong to --family; see configs/models.json")
    if args.task != "clone" and args.mode != "combine":
        parser.error("--mode applies to classifier training; extraction always writes both feature types")
    if args.task == "synprune" and args.lang != "python":
        parser.error("SynPrune supports Python")
    output = Path(args.output_dir or os.environ.get("CODEMIA_OUTPUT_ROOT", ROOT / "outputs")).resolve()
    os.environ["CODEMIA_OUTPUT_ROOT"] = str(output)
    if args.feature_dir:
        os.environ["CODEMIA_FEATURE_ROOT"] = str(Path(args.feature_dir).resolve())
    # Avoid oversubscribing CPUs on large inference servers; callers can override this.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    path = ROOT / "src" / TASKS[args.task]
    sys.path[:0] = [str(path.parent), str(ROOT / "src"), str(ROOT / "src/trainer")]
    module = runpy.run_path(str(path))
    if args.task == "clone":
        feature_root = Path(os.environ.get("CODEMIA_FEATURE_ROOT", ROOT / "data/features/clone"))
        for split in ["train", "val"]:
            required = feature_root / f"{args.family}/input_processed/{args.lang}/{args.model}_top0.1_{split}_data_clone.pkl"
            if not required.is_file():
                parser.error(f"Missing feature file: {required.name}")
        module["test_lightgbm"](args.family, args.model, args.lang, args.mode, module["get_type_feature"](args.lang))
    else:
        from experiment import extract, baselines
        (extract if args.task == "features" else baselines)(module, args, ROOT, output)


if __name__ == "__main__":
    main()
