"""Map original project-relative paths to the release layout; no experiment logic."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def release_path(relative):
    relative = str(relative).strip("/")
    parts = relative.split("/")
    output_root = Path(os.environ.get("CODEMIA_OUTPUT_ROOT", ROOT / "outputs"))
    if relative == "data":
        path = ROOT / "data/clone"
    elif parts[0] == "data" and len(parts) >= 3:
        family, kind = parts[1:3]
        if kind == "input_processed":
            path = Path(os.environ.get("CODEMIA_FEATURE_ROOT", ROOT / "data/features/clone")) / "/".join(parts[1:])
        elif kind in ("results", "lightgbm_models", "catboost_models"):
            path = output_root / "/".join(parts[1:])
        else:
            path = ROOT / "data/clone" / "/".join(parts[1:])
    elif relative.startswith("models/"):
        path = Path(os.environ.get("CODEMIA_MODEL_ROOT", ROOT / "models")) / relative.removeprefix("models/")
    elif relative.startswith("datasets/"):
        path = Path(os.environ.get("CODEMIA_DATASET_ROOT", ROOT / "raw/datasets")) / relative.removeprefix("datasets/")
    elif relative.startswith("train_data_process/github"):
        path = ROOT / "raw/github" / parts[-1]
    else:
        path = output_root / relative
    # Original scripts do not consistently create result/model directories.
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)
