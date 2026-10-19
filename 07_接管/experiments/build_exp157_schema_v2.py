#!/usr/bin/env python3
"""Normalize EXP-157 to one loadable Alpaca schema and build its mixed smoke slice."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "07_接管/experiments/data/exp157_answer_only_r0_compact_v1.jsonl"
OUTPUT = ROOT / "07_接管/experiments/data/exp157_answer_only_r0_compact_v2.jsonl"
MANIFEST = ROOT / "07_接管/experiments/manifests/exp157_answer_only_r0_compact_v2.manifest.json"
SMOKE = ROOT / "07_接管/experiments/data/exp157_answer_only_r0_smoke_v2.jsonl"
SMOKE_MANIFEST = ROOT / "07_接管/experiments/manifests/exp157_answer_only_r0_smoke_v2.manifest.json"
SOURCE_SHA = "15a2914c23b6d6a8ebeaf650d3412db2df9546f4625e2ee243f1e2705721791b"
ROWS = 176867
PUBLIC_ROWS = 32705
R0_ROWS = 144162
SID_RE = re.compile(r"<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
SMOKE_LINES = set(range(1, 9)) | set(range(PUBLIC_ROWS + 1, PUBLIC_ROWS + 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink(): raise SystemExit(f"REFUSE: {path} exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA: raise SystemExit("FAIL: v1 source SHA drift")
    if any(path.exists() or path.is_symlink() for path in (OUTPUT, MANIFEST, SMOKE, SMOKE_MANIFEST)):
        raise SystemExit("REFUSE: a v2 output already exists")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    digest = hashlib.sha256(); smoke_chunks = []; rows = 0; r0_sids = set(); terminal_c = set(); output_bytes = 0
    try:
        with os.fdopen(fd, "wb") as target, SOURCE.open(encoding="utf-8") as source:
            for rows, line in enumerate(source, 1):
                old = json.loads(line)
                normalized = {
                    "instruction": old["instruction"],
                    "input": old["input"],
                    "output": old["output"],
                    "system": old.get("system") or "",
                    "history": old["history"],
                }
                if not all(isinstance(normalized[key], str) for key in ("instruction", "input", "output", "system")) or normalized["history"] != []:
                    raise SystemExit(f"FAIL: row contract drift at {rows}")
                if rows > PUBLIC_ROWS:
                    matches = SID_RE.findall(normalized["output"])
                    if len(matches) != 1: raise SystemExit(f"FAIL: R0 SID drift at {rows}")
                    sid = "/".join(matches[0])
                    if sid in r0_sids: raise SystemExit(f"FAIL: duplicate R0 SID at {rows}")
                    r0_sids.add(sid); terminal_c.add(int(matches[0][2]))
                raw = (json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                target.write(raw); digest.update(raw); output_bytes += len(raw)
                if rows in SMOKE_LINES: smoke_chunks.append(raw)
            target.flush(); os.fsync(target.fileno())
        if rows != ROWS or len(r0_sids) != R0_ROWS or len(terminal_c) != 4162 or len(smoke_chunks) != 16:
            raise SystemExit("FAIL: v2 composition drift")
        os.link(temporary, OUTPUT); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
    output = {"path": str(OUTPUT.relative_to(ROOT)), "lines": rows, "size_bytes": output_bytes, "sha256": digest.hexdigest()}
    manifest = {
        "schema": "llm-rec.takeover.exp157-answer-only-r0-compact",
        "schema_version": 2,
        "logical_name": "exp157_answer_only_r0_compact_v2",
        "output": output,
        "composition": {"public_answer_only_rows": PUBLIC_ROWS, "c_balanced_r0_rows": R0_ROWS, "unique_r0_sids": len(r0_sids), "unique_terminal_c": len(terminal_c)},
        "schema_contract": {"columns_in_order": ["instruction", "input", "output", "system", "history"], "system_present_on_every_row": True, "empty_system_used_for_legacy_public_rows": True},
        "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": SOURCE_SHA},
        "builder_sha256": sha256(Path(__file__).resolve()),
    }
    publish(MANIFEST, (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    smoke_payload = b"".join(smoke_chunks)
    publish(SMOKE, smoke_payload)
    smoke_manifest = {
        "schema": "llm-rec.takeover.exp157-smoke-dataset",
        "schema_version": 2,
        "output": {"path": str(SMOKE.relative_to(ROOT)), "lines": 16, "size_bytes": len(smoke_payload), "sha256": hashlib.sha256(smoke_payload).hexdigest()},
        "composition": {"public_answer_only_rows": 8, "c_balanced_r0_rows": 8, "source_line_numbers": sorted(SMOKE_LINES)},
        "source": output,
        "schema_contract": manifest["schema_contract"],
        "builder_sha256": manifest["builder_sha256"],
    }
    publish(SMOKE_MANIFEST, (json.dumps(smoke_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"verdict": "PASS", "full": output, "smoke": smoke_manifest["output"]}, sort_keys=True))


if __name__ == "__main__": main()
