#!/usr/bin/env python3
"""Build the compact answer-only plus c-balanced R0 continuation dataset."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "07_接管/experiments/data/exp142_public091_answer_only_c48c42a_v1.jsonl"
PUBLIC_MANIFEST = ROOT / "07_接管/experiments/manifests/exp142_public091_answer_only_c48c42a_v1.manifest.json"
CBALANCED = ROOT / "07_接管/experiments/data/r0_cbalanced_replay_v1.jsonl"
CBALANCED_MANIFEST = ROOT / "07_接管/experiments/manifests/r0_cbalanced_replay_v1.manifest.json"
EVAL = ROOT / "07_接管/experiments/data/r0_aligned_replay_v1_eval.jsonl"
OUTPUT = ROOT / "07_接管/experiments/data/exp157_answer_only_r0_compact_v1.jsonl"
MANIFEST = ROOT / "07_接管/experiments/manifests/exp157_answer_only_r0_compact_v1.manifest.json"

PUBLIC_SHA = "ad4f2fe884e190b255c9270d94401e8f5760cc7e57bac6781bb3d539ff2ebe5a"
PUBLIC_LINES = 32705
CBALANCED_SHA = "fa1efd9c7f29b9bca64202c00bd844bae273e462199ed3e466c1fc71bd42e97d"
CBALANCED_LINES = 617979
REPLAY_PREFIX = 473817
SELECTED_LINES = 144162
EVAL_SHA = "d3e7dab4f0d1570b92aa83fa1f3310d16444b47b670c91c8fdb50d383a3b90a1"
SID_RE = re.compile(r"<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")
KEYS = ("instruction", "input", "output", "system", "history")


class BuildError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def load_row(raw: bytes, where: str) -> dict:
    require(raw.endswith(b"\n"), f"missing newline at {where}")
    row = json.loads(raw)
    require(isinstance(row, dict), f"non-object row at {where}")
    require(set(row) in (set(KEYS), set(KEYS) - {"system"}), f"schema drift at {where}")
    require(row.get("history") == [], f"history drift at {where}")
    for key in ("instruction", "input", "output"):
        require(isinstance(row.get(key), str), f"invalid {key} at {where}")
    if "system" in row:
        require(isinstance(row["system"], str), f"invalid system at {where}")
    return row


def scan_eval_sids() -> set[str]:
    require(sha256(EVAL) == EVAL_SHA, "evaluation exclusion SHA drift")
    result: set[str] = set()
    with EVAL.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            row = load_row(raw, f"eval:{line_number}")
            matches = SID_RE.findall(row["output"])
            require(len(matches) == 1, f"evaluation SID drift at line {line_number}")
            result.add("/".join(matches[0]))
    require(result, "empty evaluation SID set")
    return result


def source_record(path: Path, role: str) -> dict:
    return {
        "role": role,
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def publish() -> dict:
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), f"refusing to overwrite {OUTPUT}")
    require(not MANIFEST.exists() and not MANIFEST.is_symlink(), f"refusing to overwrite {MANIFEST}")
    require(sha256(PUBLIC) == PUBLIC_SHA, "public answer-only SHA drift")
    require(sha256(CBALANCED) == CBALANCED_SHA, "c-balanced SHA drift")
    public_manifest = json.loads(PUBLIC_MANIFEST.read_text(encoding="utf-8"))
    cbalanced_manifest = json.loads(CBALANCED_MANIFEST.read_text(encoding="utf-8"))
    require(public_manifest["output"]["sha256"] == PUBLIC_SHA, "public manifest binding drift")
    require(cbalanced_manifest["output"]["sha256"] == CBALANCED_SHA, "c-balanced manifest binding drift")
    require(cbalanced_manifest["composition"]["replay_rows"] == REPLAY_PREFIX, "replay boundary drift")
    eval_sids = scan_eval_sids()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    digest = hashlib.sha256()
    public_rows = r0_rows = output_bytes = 0
    r0_sids: set[str] = set()
    c_values: set[int] = set()
    try:
        with os.fdopen(fd, "wb") as target:
            with PUBLIC.open("rb") as source:
                for public_rows, raw in enumerate(source, 1):
                    load_row(raw, f"public:{public_rows}")
                    target.write(raw); digest.update(raw); output_bytes += len(raw)
            with CBALANCED.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    if line_number <= REPLAY_PREFIX:
                        continue
                    r0_rows += 1
                    row = load_row(raw, f"r0:{line_number}")
                    matches = SID_RE.findall(row["output"])
                    require(len(matches) == 1, f"R0 SID drift at source line {line_number}")
                    sid = "/".join(matches[0])
                    require(sid not in r0_sids, f"duplicate selected R0 SID at source line {line_number}")
                    require(sid not in eval_sids, f"evaluation SID leak at source line {line_number}")
                    r0_sids.add(sid); c_values.add(int(matches[0][2]))
                    target.write(raw); digest.update(raw); output_bytes += len(raw)
            target.flush(); os.fsync(target.fileno())
        require(public_rows == PUBLIC_LINES, f"public row count {public_rows}")
        require(r0_rows == SELECTED_LINES, f"selected R0 row count {r0_rows}")
        require(REPLAY_PREFIX + r0_rows == CBALANCED_LINES, "c-balanced boundary mismatch")
        os.link(temporary, OUTPUT); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass

    manifest = {
        "schema": "llm-rec.takeover.exp157-answer-only-r0-compact",
        "schema_version": 1,
        "logical_name": "exp157_answer_only_r0_compact_v1",
        "output": {
            "path": str(OUTPUT.relative_to(ROOT)),
            "lines": public_rows + r0_rows,
            "size_bytes": output_bytes,
            "sha256": digest.hexdigest(),
        },
        "composition": {
            "public_answer_only_rows": public_rows,
            "c_balanced_r0_rows": r0_rows,
            "unique_r0_sids": len(r0_sids),
            "unique_terminal_c": len(c_values),
        },
        "isolation": {"evaluation_sid_overlap": 0, "evaluation_sids": len(eval_sids)},
        "transform": "Exact public answer-only rows followed by the selected-R0 suffix of the locked c-balanced dataset; old v8t replay prefix omitted.",
        "sources": [
            source_record(PUBLIC, "public_answer_only"),
            source_record(PUBLIC_MANIFEST, "public_manifest"),
            source_record(CBALANCED, "c_balanced"),
            source_record(CBALANCED_MANIFEST, "c_balanced_manifest"),
            source_record(EVAL, "evaluation_exclusion"),
            source_record(Path(__file__).resolve(), "builder"),
        ],
    }
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{MANIFEST.name}.", dir=MANIFEST.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, MANIFEST); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass
    return manifest


if __name__ == "__main__":
    report = publish()
    print(json.dumps({"status": "PASS", "output": report["output"], "composition": report["composition"]}, ensure_ascii=False))
