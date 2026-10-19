"""Scan release files and archives without printing matched private values.

The supplied paper PDF is intentionally exempt. This is a pattern-based check,
not a claim that every natural-language name in a code corpus can be recognized.
"""
import argparse
from collections import Counter
from io import BytesIO
import getpass
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private_home": re.compile(rb"/(?:home|Users)/[A-Za-z0-9][A-Za-z0-9_.-]*(?=[/\\\s\"'])"),
    "credential": re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})\b"),
    "email": re.compile(rb"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])"),
}
ASSIGNMENT = re.compile(rb"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|password|client[_-]?secret)\b(?:\\?[\"'])?\s*[:=]\s*\\?[\"']([^\"'\\\r\n]{4,})")


def scan(root=ROOT, deny_names=(), archives=True):
    errors = []
    counts = Counter()
    current_user = getpass.getuser()
    names = [n.encode().lower() for n in deny_names if n]
    if current_user not in {"root", "user", "runner", "codex"}:
        names.append(current_user.encode().lower())

    def inspect(name, payload):
        counts["files"] += 1
        if name.endswith(".pdf"):
            counts["paper_exempt"] += 1
            return
        if name.endswith(".jsonl"):
            # Decode escaped newlines/quotes before inspecting source-code strings.
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, list):
                    for item in value:
                        yield from strings(item)
                elif isinstance(value, dict):
                    for key, item in value.items():
                        yield key
                        yield from strings(item)
            try:
                payload = "\n".join(text for line in payload.decode("utf-8").splitlines() if line.strip()
                                    for text in strings(json.loads(line))).encode("utf-8")
            except (UnicodeError, ValueError):
                errors.append({"file": name, "error": "invalid JSONL"})
                return
            counts["jsonl"] += 1
        if payload[:4] == b"PAR1":
            # Inspect compressed text columns and schema metadata, not just raw bytes.
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.ParquetFile(BytesIO(payload))
            columns = [field.name for field in table.schema_arrow
                       if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
            content = table.schema_arrow.to_string(show_schema_metadata=True).encode()
            if columns:
                content += json.dumps(table.read(columns=columns).to_pydict()).encode()
            payload = content
            counts["parquet"] += 1
        for label, pattern in PATTERNS.items():
            matches = list(pattern.finditer(payload))
            if label == "email":
                matches = [m for m in matches if m.group() != b"redacted@example.invalid"]
            if matches:
                errors.append({"file": name, "error": label, "count": len(matches)})
        lowered = payload.lower()
        literal_count = 0
        for match in ASSIGNMENT.finditer(lowered):
            value = match.group(1)
            if not (value.startswith((b"redacted", b"your_", b"example", b"<", b"${")) or value in {b"empty", b"token", b"none", b"null"}):
                literal_count += 1
        if literal_count:
            errors.append({"file": name, "error": "literal credential assignment", "count": literal_count})
        if any(n in lowered for n in names):
            errors.append({"file": name, "error": "private identifier"})

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            errors.append({"file": str(relative), "error": "unexpected symlink"})
            continue
        if not path.is_file():
            continue
        if path.name.endswith(".tar.gz"):
            if not archives:
                continue
            with tarfile.open(path, "r:gz") as archive:
                counts["archives"] += 1
                for member in archive:
                    if member.uname or member.gname or member.uid or member.gid:
                        errors.append({"file": str(relative), "error": "archive owner metadata"})
                    if member.issym() or member.islnk():
                        errors.append({"file": str(relative), "error": "archive link"})
                    if member.isfile():
                        inspect(str(relative)+":"+member.name, archive.extractfile(member).read())
            continue
        inspect(str(relative), path.read_bytes())
    return {"status": "pass" if not errors else "fail", "inspected": dict(counts), "errors": errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deny-name", action="append", default=[])
    args = parser.parse_args()
    report = scan(deny_names=args.deny_name)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))
