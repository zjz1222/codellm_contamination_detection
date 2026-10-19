"""Run feature extraction and grouped baselines using the retained scoring functions."""
import asyncio
import json
from pathlib import Path
import random

GROUPS = ["original", "Type I", "Type II", "Type III"]


def extract(module, args, root, output):
    model = Path(module["_release_path"](f"models/{args.model}"))
    for split in ["train", "val"]:
        source = root / f"data/clone/{args.family}/input/{args.lang}/{args.model}_{split}_data_clone.jsonl"
        target = output / f"features/{args.family}/input_processed/{args.lang}/{args.model}_top0.1_{split}_data_clone.pkl"
        target.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(module["dataset_process"](
            str(model), str(source), str(target), module["get_type_feature"](args.lang),
            module["get_max_length"](args.model), "combine", 0.1,
        ))


def grouped_scores(module, model, tokenizer, rows, max_length, task):
    """Keep code groups separate and average their metrics with equal weight."""
    import numpy as np
    prefix = None
    if task == "baselines":
        candidates = [r["code"] for r in rows if r["label"] == 0 and r["type"] == "original"]
        prefix = random.sample(candidates, k=1)
        avg_length = int(np.mean([len(tokenizer.encode(r["code"])) for r in rows]))
        prefix = module["process_prefix"](tokenizer, prefix, avg_length, False, 1, max_length)
    tables = []
    for group in GROUPS:
        selected = [r for r in rows if r["type"].replace("_", " ") == group and not r["code"].startswith("Error")]
        codes = [r["code"] for r in selected]
        labels = [r["label"] for r in selected]
        if set(labels) != {0, 1}:
            raise ValueError(f"Both labels are required in {group}")
        if task == "baselines":
            scores = module["predict"](model, tokenizer, codes, max_length, prefix=prefix)
        else:
            scores = module["eval"](model, tokenizer, codes, max_length)
        table = module["calculate"](scores, labels).copy()
        for column in ["auc", "accuracy", "tpr01"]:
            table[column] = table[column].map(lambda v: float(v.rstrip("%"))/100 if isinstance(v, str) else float(v))
        table.insert(0, "Clone Type", group)
        tables.append(table)
    import pandas as pd
    combined = pd.concat(tables, ignore_index=True)
    mean = combined.groupby("method", sort=False)[["auc", "accuracy", "tpr01"]].mean().reset_index()
    mean.insert(0, "Clone Type", "Macro Average")
    return pd.concat([combined, mean], ignore_index=True)


def baselines(module, args, root, output):
    if args.task == "synprune" and args.lang != "python":
        raise ValueError("SynPrune supports Python")
    random.seed(42)
    if "fix_seed" in module:
        module["fix_seed"](42)
    source = root / f"data/clone/{args.family}/input/{args.lang}/{args.model}_val_data_clone.jsonl"
    with source.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    model, tokenizer = module["get_transformers_model"](module["_release_path"](f"models/{args.model}"))
    table = grouped_scores(module, model, tokenizer, rows, module["get_max_length"](args.model), args.task)
    target = output / f"{args.family}/results/{args.lang}/{args.model}_{args.task}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(target, index=False)
    print(table.to_string(index=False))
