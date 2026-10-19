#!/usr/bin/env python3
"""Independently verify the immutable EXP-157 compact continuation dataset."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "07_接管/experiments/data/exp157_answer_only_r0_compact_v2.jsonl"
MANIFEST = ROOT / "07_接管/experiments/manifests/exp157_answer_only_r0_compact_v2.manifest.json"
REGISTRY = ROOT / "07_接管/experiments/dataset_registry_exp157_answer_only_r0_compact/dataset_info.json"
EXPECTED_SHA = "78ee84eefeadb677b8f627073a71fe100223c85fc684213a4ea77da857653ef9"
EXPECTED_ROWS = 176867
PUBLIC_ROWS = 32705
EXPECTED_R0 = 144162
EXPECTED_C = 4162
SID_RE = re.compile(r"<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = registry.get("exp157_answer_only_r0_compact_v2", {})
    if Path(entry.get("file_name", "")).resolve() != DATA.resolve():
        fail("registry path drift")
    if entry.get("columns", {}).get("system") != "system":
        fail("system column is not mapped")
    if manifest.get("output", {}).get("sha256") != EXPECTED_SHA:
        fail("manifest SHA binding drift")
    if manifest.get("output", {}).get("lines") != EXPECTED_ROWS:
        fail("manifest row binding drift")

    digest = hashlib.sha256()
    rows = 0
    r0_sids: set[str] = set()
    terminal_c: set[int] = set()
    with DATA.open("rb") as handle:
        for rows, raw in enumerate(handle, 1):
            digest.update(raw)
            if not raw.endswith(b"\n"):
                fail(f"missing newline at row {rows}")
            row = json.loads(raw)
            if tuple(row) != ("instruction", "input", "output", "system", "history") or not isinstance(row.get("system"), str):
                fail(f"uniform schema drift at row {rows}")
            if row.get("history") != [] or not all(isinstance(row.get(key), str) for key in ("instruction", "input", "output")):
                fail(f"row contract drift at row {rows}")
            if rows <= PUBLIC_ROWS:
                continue
            matches = SID_RE.findall(row["output"])
            if len(matches) != 1 or not row["input"].endswith("/no_think"):
                fail(f"R0 contract drift at row {rows}")
            sid = "/".join(matches[0])
            if sid in r0_sids:
                fail(f"duplicate R0 SID at row {rows}")
            r0_sids.add(sid)
            terminal_c.add(int(matches[0][2]))

    if rows != EXPECTED_ROWS or digest.hexdigest() != EXPECTED_SHA:
        fail("dataset byte or row drift")
    if len(r0_sids) != EXPECTED_R0 or len(terminal_c) != EXPECTED_C:
        fail("R0 coverage drift")
    print(json.dumps({"verdict": "PASS", "rows": rows, "sha256": digest.hexdigest(), "unique_r0_sids": len(r0_sids), "unique_terminal_c": len(terminal_c)}, sort_keys=True))


if __name__ == "__main__":
    main()
