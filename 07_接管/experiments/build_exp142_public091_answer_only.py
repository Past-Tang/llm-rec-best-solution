#!/usr/bin/env python3
"""Build the receipt-independent EXP-142 public091 answer-only dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import build_exp132_public091 as base


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/exp142_public091_answer_only_c48c42a_v1.jsonl"
MANIFEST = ROOT / "manifests/exp142_public091_answer_only_c48c42a_v1.manifest.json"
REGISTRY = ROOT / "dataset_registry_exp142_public091_answer_only/dataset_info.json"
EVIDENCE = ROOT.parent / "analysis/external_evidence/czxivy_onereason_competition_c48c42a.json"
EXPECTED_SHA256 = "ad4f2fe884e190b255c9270d94401e8f5760cc7e57bac6781bb3d539ff2ebe5a"
EXPECTED_BYTES = 218_959_628
EXPECTED_LINES = 32_705
EXPECTED_GROUPS = {"rec": 18_651, "item": 9_684, "user": 2_792, "world": 1_578}
EXTERNAL_COMMIT = "c48c42a49d5b9608988c71f58013ebcca54a8d9e"
EXTERNAL_BUILDER_BLOB = "88c96b00056f0faee611260d8d6686067437d5b6"
THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


class BuildError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build() -> dict:
    source = base.validate_source()
    require(EVIDENCE.is_file() and not EVIDENCE.is_symlink(), f"missing evidence: {EVIDENCE}")
    evidence = json.loads(EVIDENCE.read_text())
    require(evidence.get("commit") == EXTERNAL_COMMIT, "external commit drift")
    builder = evidence.get("files", {}).get("scripts/data/prepare_onereason_sft_answer_only.py", {})
    require(builder.get("git_blob_sha1") == EXTERNAL_BUILDER_BLOB, "external builder blob drift")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent)
    groups: Counter[str] = Counter()
    filled_thoughts = 0
    original_chars = 0
    cleaned_chars = 0
    lines = 0
    try:
        with base.SOURCE_PATH.open("r", encoding="utf-8") as src, os.fdopen(fd, "w", encoding="utf-8") as dst:
            for lines, line in enumerate(src, 1):
                row = base.parse_source_row(line, lines)
                group, _, thought_mode = base.inspect_row(row, lines)
                match = base.THINK_RE.fullmatch(row["response"])
                require(match is not None, f"invalid think wrapper at line {lines}")
                final = match.group(2).strip()
                require(bool(final), f"empty final answer at line {lines}")
                require(not THINK_TAG.search(final), f"remaining think tag at line {lines}")
                converted = {
                    "instruction": row["system"],
                    "input": row["prompt"],
                    "output": final,
                    "history": [],
                }
                dst.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
                groups[group] += 1
                filled_thoughts += thought_mode == "filled"
                original_chars += len(row["response"])
                cleaned_chars += len(final)
            dst.flush()
            os.fsync(dst.fileno())
        temporary = Path(temporary_name)
        require(lines == EXPECTED_LINES, f"line count drift: {lines}")
        require(dict(groups) == EXPECTED_GROUPS, f"group drift: {dict(groups)}")
        require(temporary.stat().st_size == EXPECTED_BYTES, "output byte count drift")
        require(sha256_file(temporary) == EXPECTED_SHA256, "output hash drift")
        if OUTPUT.exists():
            require(OUTPUT.is_file() and not OUTPUT.is_symlink(), "unsafe existing output")
            require(OUTPUT.stat().st_size == EXPECTED_BYTES and sha256_file(OUTPUT) == EXPECTED_SHA256, "existing output conflicts")
            temporary.unlink()
        else:
            os.replace(temporary, OUTPUT)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    manifest = {
        "schema": "llm-rec.takeover.exp142-public091-answer-only",
        "schema_version": 1,
        "logical_name": "exp142_public091_answer_only_c48c42a_v1",
        "source": source,
        "output": {"path": str(OUTPUT), "lines": lines, "size_bytes": EXPECTED_BYTES, "sha256": EXPECTED_SHA256},
        "composition": {"groups": dict(groups), "filled_thought_rows_removed": filled_thoughts, "empty_thought_wrappers_removed": lines - filled_thoughts},
        "audit": {
            "empty_outputs": 0,
            "remaining_think_tags": 0,
            "original_response_chars": original_chars,
            "cleaned_response_chars": cleaned_chars,
            "mean_original_response_chars": original_chars / lines,
            "mean_cleaned_response_chars": cleaned_chars / lines,
        },
        "transform": "Preserve source order and prompts; replace each response with the non-empty final text after its balanced </think> wrapper.",
        "external_evidence": {
            "repository": evidence.get("repository"),
            "commit": EXTERNAL_COMMIT,
            "answer_only_builder_git_blob_sha1": EXTERNAL_BUILDER_BLOB,
            "evidence_path": str(EVIDENCE),
            "evidence_sha256": sha256_file(EVIDENCE),
            "official_score_present": False,
        },
    }
    registry = {
        manifest["logical_name"]: {
            "file_name": str(OUTPUT),
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output", "history": "history"},
        }
    }
    atomic_write(MANIFEST, canonical(manifest))
    atomic_write(REGISTRY, canonical(registry))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        print("FAIL: pass --build")
        return 2
    try:
        print(json.dumps(build(), ensure_ascii=False, indent=2, sort_keys=True))
    except (BuildError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
