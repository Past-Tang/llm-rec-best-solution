#!/usr/bin/env python3
"""Build a replay-preserving R0 dataset with a capped terminal-code frequency."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs/r0_cbalanced_replay_source_config.json"
DEFAULT_OUTPUT = HERE / "data/r0_cbalanced_replay_v1.jsonl"
DEFAULT_MANIFEST = HERE / "manifests/r0_cbalanced_replay_v1.manifest.json"
DEFAULT_STATE = HERE / "manifests/r0_cbalanced_replay_v1.build_state.json"
REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")
SID_RE = re.compile(
    r"<\|video_begin\|><s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class BuildError(RuntimeError):
    """Raised when an input lock or the c-balanced contract is violated."""


@dataclass(frozen=True)
class Candidate:
    line_number: int
    sid: str
    a: int
    b: int
    c: int
    rank: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as error:
        raise BuildError(f"path escapes repository: {path}") from error


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing to write through symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def publish_new(path: Path, payload: bytes) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise BuildError(f"refusing to overwrite: {path}") from error
        os.unlink(temporary)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def file_record(path: Path, role: str | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path}")
    record: dict[str, Any] = {
        "path": relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        record["role"] = role
    return record


def resolve_repo_path(raw_path: str, label: str) -> Path:
    require(isinstance(raw_path, str) and raw_path, f"{label}.path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    relative_path(path)
    return path.resolve()


def validate_file_spec(spec: Any, label: str, *, count_key: str | None = None) -> None:
    require(isinstance(spec, dict), f"{label} must be an object")
    resolve_repo_path(spec.get("path", ""), label)
    size = spec.get("size_bytes")
    digest = spec.get("sha256")
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0, f"{label}.size_bytes")
    require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None, f"{label}.sha256")
    if count_key is not None:
        count = spec.get(count_key)
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0, f"{label}.{count_key}")


def load_config(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid source config: {error}") from error
    require(isinstance(config, dict) and config.get("schema_version") == 1, "unsupported config")
    require(config.get("logical_name") == "r0_cbalanced_replay_v1", "logical name drift")
    validate_file_spec(config.get("source_dataset"), "source_dataset", count_key="lines")
    validate_file_spec(config.get("source_manifest"), "source_manifest")
    validate_file_spec(config.get("evaluation_exclusion"), "evaluation_exclusion", count_key="lines")
    validate_file_spec(config.get("evaluation_manifest"), "evaluation_manifest")
    selection = config.get("selection")
    require(isinstance(selection, dict), "selection is missing")
    for key in ("replay_prefix_rows", "max_unique_sids_per_c", "max_views_per_sid"):
        value = selection.get(key)
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"selection.{key}")
    require(selection["max_views_per_sid"] == 1, "this builder requires one selected view per SID")
    require(selection["replay_prefix_rows"] < config["source_dataset"]["lines"], "replay prefix covers source")
    salt = selection.get("selection_salt")
    require(isinstance(salt, str) and salt, "selection salt is missing")
    return config


def verify_locked_file(spec: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_repo_path(spec["path"], label)
    actual = file_record(path, label)
    require(actual["size_bytes"] == spec["size_bytes"], f"{label} byte drift")
    require(actual["sha256"] == spec["sha256"], f"{label} SHA-256 drift")
    return path, actual


def verify_manifest_binding(
    manifest_path: Path,
    dataset_path: Path,
    dataset_spec: Mapping[str, Any],
    label: str,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid {label}: {error}") from error
    output = manifest.get("output") if isinstance(manifest, dict) else None
    require(isinstance(output, dict), f"{label} has no output lock")
    declared = resolve_repo_path(output.get("path", ""), f"{label}.output")
    require(declared == dataset_path, f"{label} output path mismatch")
    require(output.get("sha256") == dataset_spec["sha256"], f"{label} output SHA mismatch")
    require(output.get("lines") == dataset_spec["lines"], f"{label} output line mismatch")
    expected_bytes = output.get("bytes", output.get("size_bytes"))
    require(expected_bytes == dataset_spec["size_bytes"], f"{label} output byte mismatch")


def parse_record(raw: bytes, location: str, *, require_r0: bool) -> tuple[dict[str, Any], re.Match[str] | None]:
    require(raw.endswith(b"\n"), f"row lacks newline at {location}")
    try:
        record = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid JSON at {location}") from error
    require(
        isinstance(record, dict)
        and tuple(record) == REQUIRED_KEYS
        and record.get("history") == []
        and all(isinstance(record.get(key), str) for key in REQUIRED_KEYS[:-1]),
        f"row contract drift at {location}",
    )
    matches = list(SID_RE.finditer(record["output"]))
    if require_r0:
        require(record["instruction"] == "", f"R0 instruction drift at {location}")
        require(record["input"].endswith("/no_think"), f"R0 mode suffix drift at {location}")
        require(len(matches) == 1, f"R0 target count drift at {location}")
    return record, matches[0] if matches else None


def candidate_rank(salt: str, sid: str, raw: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(sid.encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(raw).digest())
    return digest.hexdigest()


def select_candidates(candidates: Iterable[Candidate], cap: int) -> list[Candidate]:
    require(isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, "cap must be positive")
    best_by_sid: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best_by_sid.get(candidate.sid)
        if current is None or (candidate.rank, candidate.line_number) < (current.rank, current.line_number):
            best_by_sid[candidate.sid] = candidate
    by_c: dict[int, list[Candidate]] = collections.defaultdict(list)
    for candidate in best_by_sid.values():
        by_c[candidate.c].append(candidate)
    selected: list[Candidate] = []
    for c in sorted(by_c):
        ordered = sorted(
            by_c[c],
            key=lambda item: (item.rank, item.a, item.b, item.sid, item.line_number),
        )
        selected.extend(ordered[:cap])
    return sorted(selected, key=lambda item: item.line_number)


def scan_evaluation(path: Path, expected_lines: int) -> tuple[set[str], str]:
    digest = hashlib.sha256()
    sids: set[str] = set()
    rows = 0
    with path.open("rb") as handle:
        for rows, raw in enumerate(handle, 1):
            digest.update(raw)
            _record, match = parse_record(raw, f"evaluation:{rows}", require_r0=True)
            require(match is not None, f"evaluation target missing at line {rows}")
            sids.add(match.group(0))
    require(rows == expected_lines, f"evaluation row count {rows} != {expected_lines}")
    require(len(sids) > 0, "evaluation SID set is empty")
    return sids, digest.hexdigest()


def scan_source(
    path: Path,
    expected_lines: int,
    replay_prefix_rows: int,
    evaluation_sids: set[str],
    salt: str,
) -> tuple[list[Candidate], dict[str, Any]]:
    digest = hashlib.sha256()
    rows = 0
    candidates: list[Candidate] = []
    source_r0_sids: set[str] = set()
    with path.open("rb") as handle:
        for rows, raw in enumerate(handle, 1):
            digest.update(raw)
            _record, match = parse_record(
                raw,
                f"source:{rows}",
                require_r0=rows > replay_prefix_rows,
            )
            if rows <= replay_prefix_rows:
                continue
            require(match is not None, f"source R0 target missing at line {rows}")
            sid = match.group(0)
            require(sid not in evaluation_sids, f"evaluation SID leaked into source at line {rows}")
            source_r0_sids.add(sid)
            values = {name: int(match.group(name)) for name in ("a", "b", "c")}
            require(all(0 <= value <= 8191 for value in values.values()), f"subtoken out of range at line {rows}")
            candidates.append(
                Candidate(
                    line_number=rows,
                    sid=sid,
                    a=values["a"],
                    b=values["b"],
                    c=values["c"],
                    rank=candidate_rank(salt, sid, raw),
                )
            )
    require(rows == expected_lines, f"source row count {rows} != {expected_lines}")
    return candidates, {
        "rows": rows,
        "sha256": digest.hexdigest(),
        "r0_rows": len(candidates),
        "r0_unique_sids": len(source_r0_sids),
    }


def write_output(
    source_path: Path,
    output_path: Path,
    replay_prefix_rows: int,
    selected: Sequence[Candidate],
) -> dict[str, Any]:
    selected_lines = {candidate.line_number for candidate in selected}
    require(len(selected_lines) == len(selected), "selected source lines are not unique")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require(not output_path.exists() and not output_path.is_symlink(), f"refusing to overwrite: {output_path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    digest = hashlib.sha256()
    output_rows = 0
    output_bytes = 0
    try:
        with os.fdopen(descriptor, "wb") as output, source_path.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                if line_number <= replay_prefix_rows or line_number in selected_lines:
                    output.write(raw)
                    digest.update(raw)
                    output_rows += 1
                    output_bytes += len(raw)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_name, output_path)
        except FileExistsError as error:
            raise BuildError(f"refusing to overwrite: {output_path}") from error
        os.unlink(temporary_name)
        temporary_name = ""
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    expected_rows = replay_prefix_rows + len(selected)
    require(output_rows == expected_rows, f"output row count {output_rows} != {expected_rows}")
    require(output_path.stat().st_size == output_bytes, "output byte count changed after publish")
    require(sha256_file(output_path) == digest.hexdigest(), "output SHA changed after publish")
    return {
        "path": relative_path(output_path),
        "lines": output_rows,
        "bytes": output_bytes,
        "sha256": digest.hexdigest(),
    }


def distribution_summary(selected: Sequence[Candidate]) -> dict[str, Any]:
    by_c = collections.Counter(candidate.c for candidate in selected)
    by_a = collections.Counter(candidate.a for candidate in selected)
    by_b = collections.Counter(candidate.b for candidate in selected)
    require(by_c and all(value > 0 for value in by_c.values()), "empty selected distribution")
    ordered = sorted(by_c.values())
    return {
        "unique_sids": len({candidate.sid for candidate in selected}),
        "unique_a": len(by_a),
        "unique_b": len(by_b),
        "unique_c": len(by_c),
        "unique_ab": len({(candidate.a, candidate.b) for candidate in selected}),
        "rows_per_c": {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
        },
    }


def build(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    for path in (output_path, manifest_path, sidecar_path):
        require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    atomic_write(state_path, canonical_json_bytes({"schema_version": 1, "status": "BUILDING"}))
    config = load_config(config_path)
    source_path, source_record = verify_locked_file(config["source_dataset"], "source_dataset")
    source_manifest_path, source_manifest_record = verify_locked_file(config["source_manifest"], "source_manifest")
    evaluation_path, evaluation_record = verify_locked_file(config["evaluation_exclusion"], "evaluation_exclusion")
    evaluation_manifest_path, evaluation_manifest_record = verify_locked_file(config["evaluation_manifest"], "evaluation_manifest")
    verify_manifest_binding(source_manifest_path, source_path, config["source_dataset"], "source manifest")
    verify_manifest_binding(
        evaluation_manifest_path,
        evaluation_path,
        config["evaluation_exclusion"],
        "evaluation manifest",
    )
    evaluation_sids, evaluation_digest = scan_evaluation(
        evaluation_path, config["evaluation_exclusion"]["lines"]
    )
    require(evaluation_digest == config["evaluation_exclusion"]["sha256"], "evaluation changed during scan")
    selection = config["selection"]
    candidates, source_scan = scan_source(
        source_path,
        config["source_dataset"]["lines"],
        selection["replay_prefix_rows"],
        evaluation_sids,
        selection["selection_salt"],
    )
    require(source_scan["sha256"] == config["source_dataset"]["sha256"], "source changed during scan")
    selected = select_candidates(candidates, selection["max_unique_sids_per_c"])
    require(len({candidate.sid for candidate in selected}) == len(selected), "selected SIDs are not unique")
    require({candidate.sid for candidate in selected}.isdisjoint(evaluation_sids), "selected/evaluation SID overlap")
    output = write_output(source_path, output_path, selection["replay_prefix_rows"], selected)
    distribution = distribution_summary(selected)
    require(
        distribution["rows_per_c"]["max"] <= selection["max_unique_sids_per_c"],
        "terminal-code cap was exceeded",
    )
    manifest = {
        "schema": "llm-rec.takeover.r0-cbalanced-replay-dataset",
        "schema_version": 1,
        "logical_name": config["logical_name"],
        "output": output,
        "composition": {
            "replay_rows": selection["replay_prefix_rows"],
            "selected_r0_rows": len(selected),
            "total_rows": output["lines"],
        },
        "selection": selection,
        "selected_distribution": distribution,
        "source_scan": source_scan,
        "isolation": {
            "evaluation_sids": len(evaluation_sids),
            "selected_evaluation_sid_overlap": 0,
            "one_selected_view_per_sid": True,
        },
        "sources": [
            source_record,
            source_manifest_record,
            evaluation_record,
            evaluation_manifest_record,
            file_record(config_path, "source_config"),
            file_record(Path(__file__).resolve(), "builder"),
        ],
        "data_boundary": {
            "allowed": ["audited training replay", "official public position-level alignment"],
            "forbidden": ["platform hidden inputs", "platform hidden generations", "formal evaluation logs"],
        },
    }
    payload = canonical_json_bytes(manifest)
    publish_new(manifest_path, payload)
    publish_new(sidecar_path, (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii"))
    atomic_write(
        state_path,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "PASS",
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = args.state.resolve()
    try:
        report = build(
            args.config.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
            state_path,
        )
    except (BuildError, OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        try:
            atomic_write(
                state_path,
                canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                ),
            )
        except BaseException:
            pass
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "PASS: "
        f"rows={report['output']['lines']} "
        f"selected={report['composition']['selected_r0_rows']} "
        f"unique_c={report['selected_distribution']['unique_c']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
