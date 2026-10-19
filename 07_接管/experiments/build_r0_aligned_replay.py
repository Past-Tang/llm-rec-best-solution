#!/usr/bin/env python3
"""Build a replay-preserving, public-aligned multi-view R0 dataset.

The candidate has three deliberately separate components:

* clean rows from the formal v8t training corpus;
* the already audited official alignment rows;
* additional captions joined through the public PID-to-SID tables.

The join is restricted to the video domain observed in the formal R0 logs.
All source contracts are checked before a new output is atomically published.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as parquet

import build_r0_multiview as multiview
import build_r0_scale_candidate as core


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs/r0_aligned_replay_source_config.json"
DEFAULT_TRAIN_OUTPUT = HERE / "data/r0_aligned_replay_v1.jsonl"
DEFAULT_EVAL_OUTPUT = HERE / "data/r0_aligned_replay_v1_eval.jsonl"
DEFAULT_TRAIN_MANIFEST = HERE / "manifests/r0_aligned_replay_v1.manifest.json"
DEFAULT_EVAL_MANIFEST = HERE / "manifests/r0_aligned_replay_v1_eval.manifest.json"
DEFAULT_STATE = HERE / "manifests/r0_aligned_replay_v1.build_state.json"

REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")
VIDEO_SID_RE = re.compile(r"^<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
EMPTY_THINK = core.EMPTY_THINK


class BuildError(RuntimeError):
    """Raised when a source or derived-data contract cannot be proven."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes, *, allow_identical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_identical:
        raise BuildError(f"refusing to overwrite existing output: {path}")
    if path.exists() and path.read_bytes() == payload:
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
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


def protocol_bytes(record: Mapping[str, Any]) -> bytes:
    require(tuple(record) == REQUIRED_KEYS, "record key order drift")
    require(set(record) == set(REQUIRED_KEYS), "record key set drift")
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def source_record(path: Path, role: str, *, digest: str | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe {role}: {path}")
    return {
        "role": role,
        "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
        "size_bytes": path.stat().st_size,
        "sha256": digest or sha256_file(path),
    }


def resolve_repo(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise BuildError(f"source path escapes repository: {path_text}") from error
    return path


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid source config: {error}") from error
    require(isinstance(config, dict) and config.get("schema_version") == 1, "unsupported source config")
    require(config.get("logical_name") == "r0_aligned_replay_v1", "logical name drift")
    for key in ("replay_dataset", "aligned_dataset", "holdout_exclusion"):
        spec = config.get(key)
        require(isinstance(spec, dict), f"missing {key}")
        require(isinstance(spec.get("path"), str) and spec["path"], f"invalid {key}.path")
        require(isinstance(spec.get("lines"), int) and spec["lines"] > 0, f"invalid {key}.lines")
        require(re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256"))) is not None, f"invalid {key}.sha256")
    for key in ("pid2sid", "pid2caption"):
        spec = config.get(key)
        require(isinstance(spec, dict), f"missing {key}")
        require(isinstance(spec.get("path"), str) and spec["path"], f"invalid {key}.path")
        require(isinstance(spec.get("files"), int) and spec["files"] > 0, f"invalid {key}.files")
        require(isinstance(spec.get("rows"), int) and spec["rows"] > 0, f"invalid {key}.rows")
    selection = config.get("selection")
    require(isinstance(selection, dict), "missing selection")
    require(selection.get("target_domain") == "video/video", "target domain must be video/video")
    require(isinstance(selection.get("max_views_per_sid"), int) and selection["max_views_per_sid"] >= 2, "invalid max views")
    require(isinstance(selection.get("min_caption_chars"), int) and selection["min_caption_chars"] > 0, "invalid min chars")
    require(isinstance(selection.get("max_caption_chars"), int) and selection["max_caption_chars"] >= selection["min_caption_chars"], "invalid max chars")
    require(isinstance(selection.get("eval_fraction"), float) and 0 < selection["eval_fraction"] < 1, "invalid eval fraction")
    require(isinstance(selection.get("selection_salt"), str) and selection["selection_salt"], "missing selection salt")
    return config


def verify_locked(path_text: str, expected: Mapping[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_repo(path_text)
    record = source_record(path, role)
    require(record["sha256"] == expected["sha256"], f"{role} SHA-256 drift")
    require(record["size_bytes"] == expected.get("size_bytes", record["size_bytes"]), f"{role} size drift")
    return path, record


def row_sid_tokens(record: Mapping[str, Any]) -> set[str]:
    text = json.dumps(record, ensure_ascii=False)
    return {match.group(0) for match in core.SID_RE.finditer(text)}


def direct_caption(record: Mapping[str, Any]) -> str | None:
    output = record.get("output")
    prompt = record.get("input")
    if not isinstance(output, str) or not isinstance(prompt, str):
        return None
    think = core.THINK_HEAD_RE.match(output)
    final = output[think.end() :].strip() if think else output.strip()
    if core.SID_FULL_RE.fullmatch(final) is None:
        return None
    try:
        prompt_without_mode, _mode = core.strip_mode(prompt)
        _prefix, caption = core.split_generation_prompt(prompt_without_mode)
    except core.BuildError:
        return None
    return core.normalize_caption(caption)


def scan_replay(
    path: Path,
    expected_lines: int,
) -> tuple[list[dict[str, Any]], set[str], set[str], dict[str, int], str]:
    """Read the 0.9GB source once; malformed-history rows are explicitly dropped."""

    kept: list[dict[str, Any]] = []
    replay_sids: set[str] = set()
    replay_captions: set[str] = set()
    counts = collections.Counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid replay row {line_number}") from error
            require(isinstance(record, dict) and set(record) == set(REQUIRED_KEYS), f"replay schema drift at {line_number}")
            if tuple(record) != REQUIRED_KEYS:
                counts["key_order_normalized"] += 1
                record = {key: record[key] for key in REQUIRED_KEYS}
            require(all(isinstance(record[key], (str, list)) for key in REQUIRED_KEYS), f"replay field type drift at {line_number}")
            counts["source_rows"] += 1
            replay_sids.update(row_sid_tokens(record))
            caption = direct_caption(record)
            if caption:
                replay_captions.add(caption)
            if record["history"] != []:
                counts["dropped_nonempty_history"] += 1
                continue
            kept.append(record)
            counts["kept_rows"] += 1
    require(counts["source_rows"] == expected_lines, f"replay rows {counts['source_rows']} != {expected_lines}")
    return kept, replay_sids, replay_captions, dict(sorted(counts.items())), digest.hexdigest()


def scan_aligned(
    path: Path,
    expected_lines: int,
    base_lines: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
) -> tuple[set[str], set[str], list[tuple[str, str]], dict[str, int], str]:
    target_sids: set[str] = set()
    aligned_captions: set[str] = set()
    templates: set[tuple[str, str]] = set()
    counts = collections.Counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid aligned row {line_number}") from error
            require(isinstance(record, dict) and tuple(record) == REQUIRED_KEYS, f"aligned schema drift at {line_number}")
            if line_number <= base_lines:
                counts["base_rows"] += 1
                continue
            require(record["history"] == [], f"aligned additional history at {line_number}")
            response = record["output"]
            require(isinstance(response, str) and response.startswith(EMPTY_THINK), f"aligned response drift at {line_number}")
            sid = response[len(EMPTY_THINK) :].strip()
            require(VIDEO_SID_RE.fullmatch(sid) is not None, f"aligned SID drift at {line_number}")
            require(sid not in excluded_sids, f"aligned row contains excluded SID at {line_number}")
            prompt_without_mode, mode = core.strip_mode(record["input"])
            require(mode == "/no_think", f"aligned mode drift at {line_number}")
            prefix, caption = core.split_generation_prompt(prompt_without_mode)
            caption = core.normalize_caption(caption)
            require(caption and caption not in excluded_captions, f"aligned caption exclusion drift at {line_number}")
            target_sids.add(sid)
            aligned_captions.add(caption)
            templates.add((record["system"], prefix))
            counts["additional_rows"] += 1
    require(counts["base_rows"] + counts["additional_rows"] == expected_lines, "aligned row count drift")
    require(templates, "aligned template pool is empty")
    return target_sids, aligned_captions, sorted(templates), dict(sorted(counts.items())), digest.hexdigest()


def partition_files(path: Path, expected_files: int, expected_rows: int, role: str) -> tuple[list[Path], dict[str, Any]]:
    files = sorted(path.glob("*.parquet"))
    require(len(files) == expected_files, f"{role} file count {len(files)} != {expected_files}")
    rows = sum(parquet.ParquetFile(item).metadata.num_rows for item in files)
    require(rows == expected_rows, f"{role} rows {rows} != {expected_rows}")
    return files, {
        "files": len(files),
        "rows": rows,
        "domain_filter": "video/video",
        "source_files": [source_record(item, role) for item in files],
    }


def choose_views(
    views: Mapping[str, Iterable[str]],
    aligned_captions: set[str],
    max_views: int,
    salt: str,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for sid, values in views.items():
        candidates = {core.normalize_caption(value) for value in values}
        candidates.difference_update(aligned_captions)
        candidates = {value for value in candidates if value and core.SID_LIKE_RE.search(value) is None}
        ordered = sorted(candidates, key=lambda value: (-len(value), core.stable_rank(salt, "video", sid, value), value))
        if ordered:
            selected[sid] = ordered[:max_views]
    return selected


def render_rows(selected: Mapping[str, list[str]], templates: list[tuple[str, str]], salt: str) -> Iterable[dict[str, Any]]:
    for sid in sorted(selected):
        require(isinstance(sid, str), f"selected SID is not a string: {type(sid).__name__}")
        for caption in selected[sid]:
            require(isinstance(caption, str), f"selected caption is not a string: {type(caption).__name__}")
            system, prefix = core.stable_choice(templates, f"{sid}\0{caption}", "aligned-replay-video")
            yield {
                "instruction": "",
                "input": f"{prefix}{caption}/no_think",
                "output": EMPTY_THINK + sid,
                "system": system,
                "history": [],
            }


def write_outputs(
    train_path: Path,
    eval_path: Path,
    replay_rows: list[dict[str, Any]],
    aligned_path: Path,
    aligned_base_lines: int,
    selected_rows: Mapping[str, list[str]],
    eval_sids: set[str],
    templates: list[tuple[str, str]],
    salt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for output in (train_path, eval_path):
        require(not output.exists(), f"refusing to overwrite output: {output}")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    train_tmp = train_path.with_name(f".{train_path.name}.tmp")
    eval_tmp = eval_path.with_name(f".{eval_path.name}.tmp")
    for item in (train_tmp, eval_tmp):
        if item.exists():
            item.unlink()
    train_counts = collections.Counter(replay_rows=len(replay_rows))
    eval_counts = collections.Counter()
    train_hash = hashlib.sha256()
    eval_hash = hashlib.sha256()
    train_bytes = eval_bytes = 0

    def emit(handle: Any, digest: Any, record: Mapping[str, Any], counts: collections.Counter, kind: str) -> None:
        nonlocal train_bytes, eval_bytes
        payload = protocol_bytes(record)
        handle.write(payload)
        digest.update(payload)
        counts["lines"] += 1
        counts[f"{kind}_bytes"] += len(payload)
        if kind == "train":
            train_bytes += len(payload)
        else:
            eval_bytes += len(payload)

    try:
        with train_tmp.open("wb") as train_handle, eval_tmp.open("wb") as eval_handle:
            for record in replay_rows:
                # Replay rows are kept in the training side.  Evaluation SIDs were
                # selected outside this set, so this preserves entity separation.
                emit(train_handle, train_hash, record, train_counts, "train")
            with aligned_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if line_number <= aligned_base_lines:
                        continue
                    record = json.loads(line)
                    sid = record["output"][len(EMPTY_THINK) :].strip()
                    if sid in eval_sids:
                        emit(eval_handle, eval_hash, record, eval_counts, "eval")
                    else:
                        emit(train_handle, train_hash, record, train_counts, "train")
            for record in render_rows(selected_rows, templates, salt):
                sid = record["output"][len(EMPTY_THINK) :].strip()
                if sid in eval_sids:
                    emit(eval_handle, eval_hash, record, eval_counts, "eval")
                else:
                    emit(train_handle, train_hash, record, train_counts, "train")
            train_handle.flush()
            eval_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(eval_handle.fileno())
        os.replace(train_tmp, train_path)
        os.replace(eval_tmp, eval_path)
    except BaseException:
        for item in (train_tmp, eval_tmp):
            try:
                item.unlink()
            except FileNotFoundError:
                pass
        raise
    train_counts.update({"bytes": train_bytes, "sha256": train_hash.hexdigest()})
    eval_counts.update({"bytes": eval_bytes, "sha256": eval_hash.hexdigest()})
    return dict(train_counts), dict(eval_counts)


def build(
    config_path: Path,
    train_path: Path,
    eval_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    replay_path, replay_source = verify_locked(config["replay_dataset"]["path"], config["replay_dataset"], "replay_dataset")
    aligned_path, aligned_source = verify_locked(config["aligned_dataset"]["path"], config["aligned_dataset"], "aligned_dataset")
    holdout_path, holdout_source = verify_locked(config["holdout_exclusion"]["path"], config["holdout_exclusion"], "holdout_exclusion")
    replay_rows, replay_sids, replay_captions, replay_counts, replay_digest = scan_replay(replay_path, config["replay_dataset"]["lines"])
    require(replay_digest == config["replay_dataset"]["sha256"], "replay changed during scan")
    print(json.dumps({"stage": "replay", **replay_counts}, ensure_ascii=False), flush=True)

    excluded_sids: set[str] = set()
    holdout_captions: set[str] = set()
    core.load_holdout_exclusions(
        holdout_path,
        config["holdout_exclusion"]["lines"],
        excluded_sids,
        holdout_captions,
    )
    target_sids, aligned_captions, templates, aligned_counts, aligned_digest = scan_aligned(
        aligned_path,
        config["aligned_dataset"]["lines"],
        config["aligned_dataset"]["base_lines"],
        excluded_sids,
        holdout_captions,
    )
    require(aligned_digest == config["aligned_dataset"]["sha256"], "aligned changed during scan")
    print(json.dumps({"stage": "aligned", **aligned_counts, "target_sids": len(target_sids)}, ensure_ascii=False), flush=True)
    excluded_captions = set(replay_captions)
    excluded_captions.update(holdout_captions)
    excluded_captions.update(aligned_captions)

    pid2sid_dir = resolve_repo(config["pid2sid"]["path"])
    pid2caption_dir = resolve_repo(config["pid2caption"]["path"])
    sid_files, sid_inventory = partition_files(pid2sid_dir, config["pid2sid"]["files"], config["pid2sid"]["rows"], "pid2sid")
    caption_files, caption_inventory = partition_files(pid2caption_dir, config["pid2caption"]["files"], config["pid2caption"]["rows"], "pid2caption")
    pid_to_sid, sid_stats = multiview.build_video_pid_map(sid_files, config["pid2sid"]["rows"])
    print(json.dumps({"stage": "pid2sid", **sid_stats, "target_pids": len(pid_to_sid)}, ensure_ascii=False), flush=True)
    views, ambiguous, caption_stats = multiview.collect_caption_views(
        caption_files,
        pid_to_sid,
        target_sids,
        excluded_captions,
        config["pid2caption"]["rows"],
        config["selection"]["min_caption_chars"],
        config["selection"]["max_caption_chars"],
    )
    print(json.dumps({"stage": "pid2caption", **caption_stats, "ambiguous": len(ambiguous)}, ensure_ascii=False), flush=True)
    selected = choose_views(views, aligned_captions, config["selection"]["max_views_per_sid"], config["selection"]["selection_salt"])
    require(all(isinstance(sid, str) for sid in selected), "selected view map has a non-string SID")
    candidate_sids = sorted(set(selected) - replay_sids - excluded_sids)
    require(candidate_sids, "no entity-disjoint evaluation SIDs remain")
    eval_sids = {
        sid for sid in candidate_sids
        if core.stable_rank(config["selection"]["selection_salt"], "eval", sid, "")
        % 10_000 < int(config["selection"]["eval_fraction"] * 10_000)
    }
    require(eval_sids, "evaluation split is empty")
    print(json.dumps({"stage": "selection", "selected_sids": len(selected), "selected_rows": sum(map(len, selected.values())), "eval_sids": len(eval_sids)}, ensure_ascii=False), flush=True)
    train_counts, eval_counts = write_outputs(
        train_path,
        eval_path,
        replay_rows,
        aligned_path,
        config["aligned_dataset"]["base_lines"],
        selected,
        eval_sids,
        templates,
        config["selection"]["selection_salt"],
    )
    sources = [
        source_record(Path(__file__).resolve(), "builder"),
        source_record(config_path, "source_config"),
        replay_source,
        aligned_source,
        holdout_source,
        sid_inventory,
        caption_inventory,
    ]
    common = {
        "schema": "llm-rec.takeover.r0-aligned-replay-dataset",
        "schema_version": 1,
        "logical_name": "r0_aligned_replay_v1",
        "config": source_record(config_path, "source_config"),
        "sources": sources,
        "selection": config["selection"],
        "replay": replay_counts,
        "aligned": aligned_counts,
        "public_join": {
            "target_sids": len(target_sids),
            "target_pids": len(pid_to_sid),
            "sid_stats": sid_stats,
            "caption_stats": caption_stats,
            "ambiguous_caption_count": len(ambiguous),
            "selected_sids": len(selected),
            "selected_rows": sum(map(len, selected.values())),
            "eval_sids": len(eval_sids),
            "eval_rows": sum(len(selected.get(sid, ())) for sid in eval_sids),
        },
    }
    train_manifest = {
        **common,
        "output": {"path": str(train_path.resolve()), **train_counts},
        "split": "train",
    }
    eval_manifest = {
        **common,
        "output": {"path": str(eval_path.resolve()), **eval_counts},
        "split": "evaluation",
    }
    atomic_write(train_manifest_path, canonical(train_manifest, pretty=True))
    atomic_write(eval_manifest_path, canonical(eval_manifest, pretty=True))
    for manifest_path, document in ((train_manifest_path, train_manifest), (eval_manifest_path, eval_manifest)):
        atomic_write(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), (sha256_file(manifest_path) + "\n").encode("ascii"))
    state = {
        "schema": "llm-rec.takeover.r0-aligned-replay-build-state",
        "schema_version": 1,
        "status": "PASS",
        "train_manifest": source_record(train_manifest_path, "train_manifest"),
        "eval_manifest": source_record(eval_manifest_path, "eval_manifest"),
    }
    atomic_write(state_path, canonical(state, pretty=True))
    return {"status": "PASS", "train": train_counts, "evaluation": eval_counts, "eval_sids": len(eval_sids)}


def verify_output(path: Path, manifest_path: Path) -> dict[str, Any]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = document.get("output")
    require(isinstance(expected, dict), "manifest output missing")
    require(Path(expected["path"]).resolve() == path.resolve(), "manifest path mismatch")
    digest = hashlib.sha256()
    lines = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid output row {line_number}") from error
            require(isinstance(record, dict) and tuple(record) == REQUIRED_KEYS, f"output schema drift at {line_number}")
            require(record["history"] == [], f"output history drift at {line_number}")
            lines += 1
    actual = {"lines": lines, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    require(actual == {key: expected[key] for key in ("lines", "bytes", "sha256")}, f"output lock mismatch: {actual}")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build(args.config.resolve(), args.train_output.resolve(), args.eval_output.resolve(), args.train_manifest.resolve(), args.eval_manifest.resolve(), args.state.resolve())
        else:
            result = {
                "train": verify_output(args.train_output.resolve(), args.train_manifest.resolve()),
                "evaluation": verify_output(args.eval_output.resolve(), args.eval_manifest.resolve()),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        print(f"FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
