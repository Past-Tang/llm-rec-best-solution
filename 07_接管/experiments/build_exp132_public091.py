#!/usr/bin/env python3
"""Build the immutable public-0.91 dataset used by EXP-132.

The source is the Apache-2.0 Frinkleko dataset at a pinned Hugging Face
revision.  This builder only performs the field mapping published by
lldois/llm4rec; it does not filter, reshuffle, relabel, or synthesize rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "07_接管/experiments"
SOURCE_PATH = (
    EXPERIMENT_ROOT
    / "data/external/frinkleko_kuaishou_llmrec_sft_baseline_091/train.jsonl"
)
OUTPUT_PATH = EXPERIMENT_ROOT / "data/exp132_public091_exact_v1.jsonl"
MANIFEST_PATH = EXPERIMENT_ROOT / "manifests/exp132_public091_exact_v1.manifest.json"
REGISTRY_PATH = (
    EXPERIMENT_ROOT / "dataset_registry_exp132_public091/dataset_info.json"
)

SOURCE_REVISION = "762fab2e4c917032f31a78bbaf0d3bc72380897c"
SOURCE_URL = (
    "https://hf-mirror.com/datasets/Frinkleko/"
    "kuaishou-llmrec-sft-baseline-0.91/resolve/"
    f"{SOURCE_REVISION}/train.jsonl"
)
SOURCE_SHA256 = "4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8"
SOURCE_BYTES = 242_712_094
SOURCE_LINES = 32_705
OUTPUT_SHA256 = "cdedea13c560d3453697f4a3c9b96a2303bd708c84833034fa953d16482b9925"
EXPECTED_GROUPS = {"rec": 18_651, "item": 9_684, "user": 2_792, "world": 1_578}
EXPECTED_ROUTES = {"no_think": 20_934, "think": 11_771}
EXPECTED_THOUGHT_MODES = {"empty": 20_934, "filled": 11_771}

THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*(.*?)\s*\Z", re.S)
ITEMIC_RE = re.compile(
    r"<\|(?:prod|video|ad|living)_begin\|><s_a_\d+><s_b_\d+><s_c_\d+>"
)
WORLD_FINAL_RE = re.compile(r"正确答案是\s*\([A-D]\)")
RECOMMENDATION_PREFIXES = frozenset(
    {
        "请阅读下面的用户行为记录，并推断该用户在各推荐场景中的目标内容。",
        "下面给出用户历史行为线索：",
        "已知如下用户跨场景行为：",
        "以下是一个用户的多域历史行为信息：",
    }
)


class BuildError(RuntimeError):
    """Raised when the pinned source or conversion contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, lines: int | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid regular file: {path}")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if lines is not None:
        record["lines"] = lines
    return record


def parse_source_row(line: str, line_number: int) -> dict[str, str]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise BuildError(f"invalid JSON at source line {line_number}: {error}") from error
    require(
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict),
        f"unexpected source schema at line {line_number}",
    )
    row = value[0]
    require(
        set(row) == {"system", "prompt", "response"},
        f"unexpected source fields at line {line_number}: {sorted(row)}",
    )
    require(
        all(isinstance(row[key], str) for key in ("system", "prompt", "response")),
        f"non-string source field at line {line_number}",
    )
    return row


def inspect_row(row: Mapping[str, str], line_number: int) -> tuple[str, str, str]:
    prompt = row["prompt"]
    match = THINK_RE.fullmatch(row["response"])
    require(match is not None, f"missing or unbalanced think wrapper at line {line_number}")
    thought = match.group(1).strip()
    final = match.group(2).strip()

    if "/no_think" in prompt:
        route = "no_think"
    elif "/think" in prompt:
        route = "think"
    else:
        route = "none"
    require(route != "none", f"missing route suffix at line {line_number}")
    require(
        (route == "no_think" and not thought) or (route == "think" and bool(thought)),
        f"route/thought mismatch at line {line_number}",
    )

    first_line = prompt.splitlines()[0] if prompt else ""
    if first_line in RECOMMENDATION_PREFIXES:
        group = "rec"
        require(
            len(ITEMIC_RE.findall(final)) == 1,
            f"recommendation row lacks one final SID at line {line_number}",
        )
    elif WORLD_FINAL_RE.fullmatch(re.sub(r"\s+", "", final)):
        group = "world"
    else:
        try:
            parsed_final = json.loads(final)
        except (TypeError, json.JSONDecodeError):
            parsed_final = None
        group = "user" if isinstance(parsed_final, (dict, list)) else "item"
    return group, route, "filled" if thought else "empty"


def convert_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "instruction": row["system"],
        "input": row["prompt"],
        "output": row["response"],
        "history": [],
    }


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_source() -> dict[str, Any]:
    require(SOURCE_PATH.is_file() and not SOURCE_PATH.is_symlink(), f"missing source: {SOURCE_PATH}")
    record = file_record(SOURCE_PATH)
    require(record["size_bytes"] == SOURCE_BYTES, "source byte count mismatch")
    require(record["sha256"] == SOURCE_SHA256, "source SHA-256 mismatch")
    with SOURCE_PATH.open("rb") as handle:
        lines = sum(1 for _ in handle)
    require(lines == SOURCE_LINES, "source line count mismatch")
    record["lines"] = lines
    return record


def build() -> dict[str, Any]:
    source_record = validate_source()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{OUTPUT_PATH.name}.", suffix=".tmp", dir=OUTPUT_PATH.parent
    )
    groups: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    thought_modes: Counter[str] = Counter()
    line_count = 0
    try:
        with SOURCE_PATH.open("r", encoding="utf-8") as source, os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as output:
            for line_count, line in enumerate(source, 1):
                row = parse_source_row(line, line_count)
                group, route, thought_mode = inspect_row(row, line_count)
                groups[group] += 1
                routes[route] += 1
                thought_modes[thought_mode] += 1
                output.write(
                    json.dumps(
                        convert_row(row), ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        require(line_count == SOURCE_LINES, "converted line count mismatch")
        require(dict(groups) == EXPECTED_GROUPS, f"group count mismatch: {dict(groups)}")
        require(dict(routes) == EXPECTED_ROUTES, f"route count mismatch: {dict(routes)}")
        require(
            dict(thought_modes) == EXPECTED_THOUGHT_MODES,
            f"thought-mode count mismatch: {dict(thought_modes)}",
        )
        require(sha256_file(temporary) == OUTPUT_SHA256, "converted SHA-256 mismatch")
        if OUTPUT_PATH.exists():
            require(
                OUTPUT_PATH.is_file()
                and not OUTPUT_PATH.is_symlink()
                and sha256_file(OUTPUT_PATH) == OUTPUT_SHA256,
                "existing output conflicts with the immutable build",
            )
            temporary.unlink()
        else:
            os.replace(temporary, OUTPUT_PATH)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    output_record = file_record(OUTPUT_PATH, lines=SOURCE_LINES)
    manifest = {
        "schema": "llm-rec.takeover.exp132-public091-dataset",
        "schema_version": 1,
        "logical_name": "exp132_public091_exact_v1",
        "license": "Apache-2.0",
        "source": {
            **source_record,
            "repository": "Frinkleko/kuaishou-llmrec-sft-baseline-0.91",
            "revision": SOURCE_REVISION,
            "url": SOURCE_URL,
        },
        "output": output_record,
        "composition": {
            "groups": dict(groups),
            "routes": dict(routes),
            "thought_modes": dict(thought_modes),
        },
        "transform": (
            "Lossless field mapping system/prompt/response to "
            "instruction/input/output with history=[]; source order retained."
        ),
        "external_recipe_provenance": {
            "repository": "lldois/llm4rec",
            "git_commit": "61d1ce765209",
            "dataset_hash_claim_matches": True,
            "converted_hash_claim_matches": True,
        },
        "builder": file_record(Path(__file__)),
    }
    registry = {
        "exp132_public091_exact_v1": {
            "file_name": str(OUTPUT_PATH.resolve()),
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history",
            },
        }
    }
    atomic_write(MANIFEST_PATH, canonical_json_bytes(manifest))
    atomic_write(REGISTRY_PATH, canonical_json_bytes(registry))
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="perform the immutable build")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.build:
        print("FAIL: pass --build to create or verify the generated dataset")
        return 2
    try:
        manifest = build()
    except (BuildError, OSError, UnicodeError) as error:
        print(f"FAIL: {error}")
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
