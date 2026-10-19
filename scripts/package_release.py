#!/usr/bin/env python3
"""Create separate deterministic code/data archives without generated working outputs."""
import argparse
import gzip
import json
from pathlib import Path
import tarfile
from release_io import sha256, write_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output-dir", default="dist")
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
out = Path(args.output_dir).resolve()
out.mkdir(parents=True, exist_ok=True)
code_items = [".github", "src", "scripts", "tests", "configs", "docs", "README.md", "paper.pdf", "CITATION.cff", "THIRD_PARTY.md", "requirements-core.txt", "requirements-inference.txt", ".gitignore", "results/validation/environment.json", "results/validation/cleanup.json", "results/validation/table1_check.json"]
data_items = ["data", "results/legacy"]
checksums = {}
for name, items in [("codemia-code.tar.gz", code_items), ("codemia-data.tar.gz", data_items)]:
    files = []
    for item in items:
        path = root / item
        files.extend([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()))
    with (out/name).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for path in files:
                    if "__pycache__" in path.parts or path.suffix == ".pyc" or any(part.endswith(".egg-info") for part in path.parts):
                        continue
                    info = archive.gettarinfo(str(path), arcname="codemia/" + str(path.relative_to(root)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as f:
                        archive.addfile(info, f)
    checksums[name] = {"sha256":sha256(out/name), "bytes":(out/name).stat().st_size}
write_json(checksums, out/"checksums.json")
print(checksums)
