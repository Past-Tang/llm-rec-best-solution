#!/usr/bin/env python3
"""Build the answer-only, unambiguous and baseline-novel EXP-161 R0 source."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "07_接管/experiments/data/external/unambiguous_r0/all_stage1.jsonl"
SOURCE_SHA = "0a139236c0a06667682103503d4ef9b29860ce4248d657fdaf258ee5502ecb86"
BASELINES = (
    REPO / "07_接管/experiments/data/exp132_public091_exact_v1.jsonl",
    REPO / "07_接管/experiments/data/exp157_answer_only_r0_compact_v2.jsonl",
    REPO / "07_接管/experiments/data/r0_aligned_multidomain_replay_v1.jsonl",
)
EVAL = REPO / "07_接管/experiments/data/r0_aligned_replay_v1_eval.jsonl"
EVAL_SHA = "d3e7dab4f0d1570b92aa83fa1f3310d16444b47b670c91c8fdb50d383a3b90a1"
OUTPUT = REPO / "07_接管/experiments/data/exp161_external_unambiguous_r0_v1.jsonl"
MANIFEST = REPO / "07_接管/experiments/manifests/exp161_external_unambiguous_r0_v1.manifest.json"
THINK_RE = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
SID_RE = re.compile(r"^<\|(living|prod|video|ad)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
KEYS = ("instruction", "input", "output", "system", "history")


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def normalize_prompt(value: str) -> str:
    value = value.replace("/no_think", "")
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def final_answer(value: str) -> str:
    return THINK_RE.sub("", value).strip()


def pair_digest(prompt: str, answer: str) -> bytes:
    raw = json.dumps((prompt, answer), ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.blake2b(raw, digest_size=16).digest()


def load_alpaca(raw: bytes, where: str) -> tuple[str, str]:
    require(raw.endswith(b"\n"), f"missing newline at {where}")
    row = json.loads(raw)
    require(isinstance(row, dict), f"non-object row at {where}")
    require({"instruction", "input", "output"}.issubset(row), f"schema drift at {where}")
    for key in ("instruction", "input", "output"):
        require(isinstance(row[key], str), f"non-string {key} at {where}")
    prompt = row["input"] if not row["instruction"] else row["instruction"] + ("\n" + row["input"] if row["input"] else "")
    return normalize_prompt(prompt), final_answer(row["output"])


def load_external(raw: bytes, where: str) -> tuple[str, str, str]:
    require(raw.endswith(b"\n"), f"missing newline at {where}")
    value = json.loads(raw)
    require(isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict), f"not singleton object array at {where}")
    row = value[0]
    require(set(row) == {"system", "prompt", "response"}, f"schema drift at {where}")
    require(all(isinstance(row[key], str) for key in row), f"non-string field at {where}")
    return row["system"].strip(), normalize_prompt(row["prompt"]), final_answer(row["response"])


def baseline_pairs() -> set[bytes]:
    result: set[bytes] = set()
    for path in BASELINES:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                prompt, answer = load_alpaca(raw, f"{path}:{line_number}")
                if SID_RE.fullmatch(answer):
                    result.add(pair_digest(prompt, answer))
                    # Public Alpaca data sometimes wraps the actual input in a fixed instruction.
                    row = json.loads(raw)
                    result.add(pair_digest(normalize_prompt(row["input"]), answer))
    require(result, "empty baseline pair index")
    return result


def eval_sids() -> set[str]:
    require(sha256(EVAL) == EVAL_SHA, "evaluation exclusion SHA drift")
    result = set()
    with EVAL.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            _, answer = load_alpaca(raw, f"eval:{line_number}")
            require(SID_RE.fullmatch(answer) is not None, f"evaluation answer drift at line {line_number}")
            result.add(answer)
    require(result, "empty evaluation SID exclusion")
    return result


def source_record(path: Path, role: str) -> dict:
    return {"role": role, "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def publish() -> dict:
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), f"refusing to overwrite {OUTPUT}")
    require(not MANIFEST.exists() and not MANIFEST.is_symlink(), f"refusing to overwrite {MANIFEST}")
    require(sha256(SOURCE) == SOURCE_SHA, "external source SHA drift")
    excluded_pairs = baseline_pairs()
    excluded_eval_sids = eval_sids()

    answers_by_prompt: dict[str, set[str]] = collections.defaultdict(set)
    direct_rows = 0
    with SOURCE.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            _, prompt, answer = load_external(raw, f"source:{line_number}")
            if SID_RE.fullmatch(answer):
                direct_rows += 1
                answers_by_prompt[prompt].add(answer)
    ambiguous_prompts = {prompt for prompt, answers in answers_by_prompt.items() if len(answers) != 1}
    require(direct_rows == 32081, f"direct row count drift: {direct_rows}")
    require(len(answers_by_prompt) == 10926, f"unique prompt count drift: {len(answers_by_prompt)}")
    require(len(ambiguous_prompts) == 1914, f"ambiguous prompt count drift: {len(ambiguous_prompts)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    output_hash = hashlib.sha256()
    seen_prompts: set[str] = set()
    counts = collections.Counter()
    output_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as target, SOURCE.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                system, prompt, answer = load_external(raw, f"source:{line_number}")
                match = SID_RE.fullmatch(answer)
                if match is None:
                    counts["excluded_non_direct"] += 1
                    continue
                if prompt in ambiguous_prompts:
                    counts["excluded_ambiguous_rows"] += 1
                    continue
                if prompt in seen_prompts:
                    counts["excluded_duplicate_rows"] += 1
                    continue
                seen_prompts.add(prompt)
                if answer in excluded_eval_sids:
                    counts["excluded_eval_sid"] += 1
                    continue
                if pair_digest(prompt, answer) in excluded_pairs:
                    counts["excluded_baseline_pair"] += 1
                    continue
                require(all(0 <= int(value) <= 8191 for value in match.groups()[1:]), f"SID range drift at line {line_number}")
                row = {"instruction": "", "input": prompt, "output": answer, "system": system, "history": []}
                payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                target.write(payload); output_hash.update(payload); output_bytes += len(payload)
                counts["included_rows"] += 1
                counts[f"included_{match.group(1)}"] += 1
            target.flush(); os.fsync(target.fileno())
        require(counts["included_rows"] > 1000, "insufficient novel unambiguous rows")
        os.link(temporary, OUTPUT); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    manifest = {
        "schema": "llm-rec.takeover.exp161-external-unambiguous-r0-dataset",
        "schema_version": 1,
        "logical_name": "exp161_external_unambiguous_r0_v1",
        "output": {"path": str(OUTPUT.relative_to(REPO)), "lines": counts["included_rows"], "size_bytes": output_bytes, "sha256": output_hash.hexdigest()},
        "composition": dict(sorted(counts.items())),
        "source_scan": {"direct_rows": direct_rows, "unique_direct_prompts": len(answers_by_prompt), "ambiguous_prompts": len(ambiguous_prompts), "unambiguous_prompts": len(answers_by_prompt) - len(ambiguous_prompts)},
        "isolation": {"evaluation_sid_overlap": 0, "excluded_eval_sids": len(excluded_eval_sids)},
        "transform": "Final-answer-only singleton SID rows; one answer per globally unambiguous normalized prompt; exact baseline pairs, duplicates, and public evaluation SIDs excluded.",
        "sources": [
            source_record(SOURCE, "external_stage1"),
            *(source_record(path, "baseline_exclusion") for path in BASELINES),
            source_record(EVAL, "evaluation_sid_exclusion"),
            source_record(Path(__file__).resolve(), "builder"),
        ],
    }
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{MANIFEST.name}.", dir=MANIFEST.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, MANIFEST); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return manifest


if __name__ == "__main__":
    try:
        value = publish()
    except (OSError, UnicodeError, json.JSONDecodeError, BuildError) as error:
        print(f"FAIL: {error}")
        raise SystemExit(1)
    print(json.dumps({"status": "PASS", "output": value["output"], "composition": value["composition"]}, ensure_ascii=False, sort_keys=True))
