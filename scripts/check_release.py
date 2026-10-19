#!/usr/bin/env python3
"""Verify frozen artifact hashes, Python syntax, and release privacy."""
import ast
import json
from pathlib import Path
from check_privacy import scan
from release_io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    for item in manifest["artifacts"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            errors.append({"file": item["path"], "error": "artifact hash mismatch"})
    sources = list(ROOT.glob("src/**/*.py")) + list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("tests/*.py"))
    for path in sources:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            errors.append({"file": str(path.relative_to(ROOT)), "error": "Python syntax"})
    privacy = scan()
    errors.extend(privacy["errors"])
    report = {"status": "pass" if not errors else "fail", "frozen_artifacts": len(manifest["artifacts"]),
              "python_files": len(sources), "privacy": privacy["inspected"], "errors": errors}
    write_json(report, ROOT / "results/validation/current_release.json")
    print(json.dumps(report, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
