#!/usr/bin/env python3
"""Build a public-alignment-targeted video description-to-SID dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as parquet

import build_r0_scale_candidate as core


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs/r0_aligned_video_source_config.json"
DEFAULT_OUTPUT = HERE / "data/r0_aligned_video_v1.jsonl"
DEFAULT_MANIFEST = HERE / "manifests/r0_aligned_video_v1.manifest.json"
DEFAULT_STATE = HERE / "manifests/r0_aligned_video_v1.build_state.json"
REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")
VIDEO_SID_RE = re.compile(
    r"^<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)


class BuildError(RuntimeError):
    """Raised when the public-alignment dataset contract cannot be proven."""


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
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "sort_keys": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def require_repo_path(path: Path, role: str) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise BuildError(f"{role} escapes repository: {path}") from exc


def relative_path(path: Path) -> str:
    require_repo_path(path, "path")
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def file_record(path: Path, role: str | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe source file: {path}")
    record = {
        "path": relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if role is not None:
        record["role"] = role
    return record


def atomic_write_new(path: Path, payload: bytes) -> None:
    require_repo_path(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
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


def replace_state(path: Path, payload: bytes) -> None:
    require_repo_path(path, "state")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing symlink state: {path}")
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


def load_config(path: Path) -> dict[str, Any]:
    require_repo_path(path, "source config")
    require(path.is_file() and not path.is_symlink(), f"unsafe source config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"invalid source config: {exc}") from exc
    require(
        isinstance(config, dict) and config.get("schema_version") == 1,
        "unsupported source config",
    )
    require(config.get("logical_name") == "r0_aligned_video_v1", "logical name drift")
    for key, count_key in (
        ("base_dataset", "lines"),
        ("holdout_exclusion", "lines"),
        ("official_alignment", "rows"),
    ):
        spec = config.get(key)
        require(isinstance(spec, dict), f"{key} is missing")
        require(isinstance(spec.get("path"), str) and spec["path"], f"{key}.path")
        require(
            isinstance(spec.get(count_key), int)
            and not isinstance(spec[count_key], bool)
            and spec[count_key] > 0,
            f"{key}.{count_key}",
        )
        require(
            isinstance(spec.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is not None,
            f"{key}.sha256",
        )
    selection = config.get("selection")
    require(isinstance(selection, dict), "selection is missing")
    require(selection.get("target_domain") == "video", "target domain must be video")
    for key in ("max_views_per_sid", "min_caption_chars", "max_caption_chars", "minimum_unique_sids"):
        value = selection.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid selection.{key}",
        )
    require(
        selection["max_caption_chars"] >= selection["min_caption_chars"],
        "caption bounds are reversed",
    )
    require(
        isinstance(selection.get("selection_salt"), str)
        and selection["selection_salt"],
        "selection salt is missing",
    )
    for key in ("official_seed_expected_files", "official_seed_expected_rows"):
        value = config.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid {key}",
        )
    seed_glob = config.get("official_seed_glob")
    require(
        isinstance(seed_glob, str)
        and seed_glob
        and not Path(seed_glob).is_absolute()
        and ".." not in Path(seed_glob).parts,
        "unsafe official seed glob",
    )
    return config


def verify_locked_file(spec: Mapping[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    path = core.resolve_repo_path(spec["path"])
    record = file_record(path, role)
    require(record["sha256"] == spec["sha256"], f"{role} SHA-256 drift")
    return path, record


def collect_seed_target_exclusions(
    paths: Sequence[Path],
) -> tuple[set[str], set[str]]:
    excluded_sids: set[str] = set()
    excluded_captions: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BuildError(f"invalid official seed {path}:{line_number}") from exc
                prompt = record.get("prompt") if isinstance(record, dict) else None
                response = record.get("response") if isinstance(record, dict) else None
                require(
                    isinstance(prompt, str) and isinstance(response, str),
                    f"official seed fields drift at {path}:{line_number}",
                )
                prompt_without_mode, _mode = core.strip_mode(prompt)
                think = core.THINK_HEAD_RE.match(response)
                require(think is not None, f"official seed response has no think block at {path}:{line_number}")
                final = response[think.end() :].strip()
                if core.SID_FULL_RE.fullmatch(final) is None:
                    continue
                excluded_sids.add(final)
                _prefix, caption = core.split_generation_prompt(prompt_without_mode)
                excluded_captions.add(core.normalize_caption(caption))
    return excluded_sids, excluded_captions


def scan_base_target_exclusions(
    path: Path, expected_lines: int
) -> tuple[set[str], set[str], str]:
    excluded_sids: set[str] = set()
    excluded_captions: set[str] = set()
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            rows += 1
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BuildError(f"invalid base row {path}:{line_number}") from exc
            require(
                isinstance(record, dict)
                and tuple(record) == REQUIRED_KEYS
                and record["history"] == []
                and all(isinstance(record[key], str) for key in REQUIRED_KEYS[:-1]),
                f"base row contract drift at {path}:{line_number}",
            )
            think = core.THINK_HEAD_RE.match(record["output"])
            final = record["output"][think.end() :].strip() if think else record["output"].strip()
            if core.SID_FULL_RE.fullmatch(final) is None:
                continue
            prompt_without_mode, _mode = core.strip_mode(record["input"])
            _prefix, caption = core.split_generation_prompt(prompt_without_mode)
            excluded_sids.add(final)
            excluded_captions.add(core.normalize_caption(caption))
    require(rows == expected_lines, f"base row count {rows} != {expected_lines}")
    return excluded_sids, excluded_captions, digest.hexdigest()


def collect_exclusions(
    config: Mapping[str, Any],
) -> tuple[
    Path,
    list[tuple[str, str]],
    set[str],
    set[str],
    list[dict[str, Any]],
]:
    base_path, base_record = verify_locked_file(config["base_dataset"], "base_dataset")
    holdout_path, holdout_record = verify_locked_file(
        config["holdout_exclusion"], "holdout_exclusion"
    )
    alignment_path, alignment_record = verify_locked_file(
        config["official_alignment"], "official_alignment"
    )
    seed_paths = sorted(REPO_ROOT.glob(config["official_seed_glob"]))
    require(
        len(seed_paths) == config["official_seed_expected_files"],
        "official seed file count drift",
    )
    generation, _reverse, _all_seed_sids, _all_seed_captions, seed_sources = (
        core.load_seed_assets(seed_paths, config["official_seed_expected_rows"])
    )
    excluded_sids, excluded_captions = collect_seed_target_exclusions(seed_paths)
    core.load_holdout_exclusions(
        holdout_path,
        config["holdout_exclusion"]["lines"],
        excluded_sids,
        excluded_captions,
    )
    base_sids, base_captions, digest = scan_base_target_exclusions(
        base_path, config["base_dataset"]["lines"]
    )
    excluded_sids.update(base_sids)
    excluded_captions.update(base_captions)
    require(digest == config["base_dataset"]["sha256"], "base changed during scan")
    sources = [base_record, holdout_record, alignment_record, *seed_sources]
    return (
        alignment_path,
        generation["video"],
        excluded_sids,
        excluded_captions,
        sources,
    )


def collect_alignment_views(
    path: Path,
    expected_rows: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
    minimum_chars: int,
    maximum_chars: int,
) -> tuple[dict[str, set[str]], set[str], dict[str, int]]:
    require(path.is_file() and not path.is_symlink(), f"unsafe alignment file: {path}")
    source = parquet.ParquetFile(path)
    require(source.metadata.num_rows == expected_rows, "alignment row count drift")
    required_columns = {"sid_token_list", "caption_list"}
    require(required_columns <= set(source.schema_arrow.names), "alignment columns missing")
    views: dict[str, set[str]] = collections.defaultdict(set)
    caption_owner: dict[str, str | None] = {}
    counts = collections.Counter()
    all_video_sids: set[str] = set()
    rows = 0
    for batch in source.iter_batches(
        batch_size=1024, columns=["sid_token_list", "caption_list"]
    ):
        values = batch.to_pydict()
        for sids, captions in zip(values["sid_token_list"], values["caption_list"]):
            rows += 1
            require(isinstance(sids, list) and isinstance(captions, list), "list column drift")
            require(len(sids) == len(captions), f"alignment list length mismatch at row {rows}")
            for sid, caption in zip(sids, captions):
                counts["pairs_scanned"] += 1
                if not isinstance(sid, str) or VIDEO_SID_RE.fullmatch(sid) is None:
                    counts["non_video_or_invalid_sid"] += 1
                    continue
                all_video_sids.add(sid)
                if sid in excluded_sids:
                    counts["excluded_sid_pairs"] += 1
                    continue
                if not isinstance(caption, str) or not caption.strip():
                    counts["empty_caption_pairs"] += 1
                    continue
                normalized = core.normalize_caption(caption)
                if not minimum_chars <= len(normalized) <= maximum_chars:
                    counts["caption_length_pairs"] += 1
                    continue
                if normalized in excluded_captions or core.SID_LIKE_RE.search(normalized):
                    counts["excluded_caption_pairs"] += 1
                    continue
                owner = caption_owner.get(normalized)
                if owner is None and normalized not in caption_owner:
                    caption_owner[normalized] = sid
                elif owner != sid:
                    caption_owner[normalized] = None
                views[sid].add(normalized)
    require(rows == expected_rows, "alignment iteration row count drift")
    ambiguous = {caption for caption, owner in caption_owner.items() if owner is None}
    for sid in list(views):
        views[sid].difference_update(ambiguous)
        if not views[sid]:
            del views[sid]
    counts["rows_scanned"] = rows
    counts["unique_video_sids_source"] = len(all_video_sids)
    counts["unique_video_sids_with_eligible_views"] = len(views)
    counts["ambiguous_captions"] = len(ambiguous)
    return dict(views), ambiguous, dict(sorted(counts.items()))


def select_views(
    views: Mapping[str, Iterable[str]],
    max_views_per_sid: int,
    salt: str,
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for sid in sorted(views):
        captions = sorted(
            set(views[sid]),
            key=lambda caption: (
                -len(caption),
                core.stable_rank(salt, "video", sid, caption),
                caption,
            ),
        )
        if captions:
            selected[sid] = captions[:max_views_per_sid]
    return selected


def render_rows(
    selected: Mapping[str, Sequence[str]],
    templates: Sequence[tuple[str, str]],
    salt: str,
) -> Iterable[dict[str, Any]]:
    for sid in sorted(selected):
        subtokens = [int(value) for value in VIDEO_SID_RE.fullmatch(sid).groups()]
        require(all(0 <= value <= 8191 for value in subtokens), f"SID out of range: {sid}")
        for index, caption in enumerate(selected[sid]):
            system, prefix = core.stable_choice(
                templates, f"{salt}\0{sid}\0{index}", "aligned-video-forward"
            )
            yield {
                "instruction": "",
                "input": f"{prefix}{caption}/no_think",
                "output": core.EMPTY_THINK + sid,
                "system": system,
                "history": [],
            }


def protocol_bytes(record: Mapping[str, Any]) -> bytes:
    require(tuple(record) == REQUIRED_KEYS, "record key order drift")
    return (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def prepare_plan(
    config_path: Path,
) -> tuple[dict[str, Any], Path, list[tuple[str, str]], dict[str, list[str]], dict[str, int], list[dict[str, Any]]]:
    config = load_config(config_path)
    alignment_path, templates, excluded_sids, excluded_captions, sources = (
        collect_exclusions(config)
    )
    selection = config["selection"]
    views, ambiguous, scan = collect_alignment_views(
        alignment_path,
        config["official_alignment"]["rows"],
        excluded_sids,
        excluded_captions,
        selection["min_caption_chars"],
        selection["max_caption_chars"],
    )
    selected = select_views(views, selection["max_views_per_sid"], selection["selection_salt"])
    require(
        len(selected) >= selection["minimum_unique_sids"],
        f"selected only {len(selected)} aligned video SIDs",
    )
    scan["selected_unique_sids"] = len(selected)
    scan["selected_records"] = sum(len(captions) for captions in selected.values())
    scan["selected_min_views"] = min(map(len, selected.values()))
    scan["selected_max_views"] = max(map(len, selected.values()))
    scan["ambiguous_captions_rejected"] = len(ambiguous)
    sources = [
        file_record(Path(__file__).resolve(), "builder"),
        file_record(config_path, "source_config"),
        *sources,
    ]
    base_path = core.resolve_repo_path(config["base_dataset"]["path"])
    return config, base_path, templates, selected, scan, sources


def build(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    for path in (
        output_path,
        manifest_path,
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
    ):
        require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    replace_state(state_path, canonical({"schema_version": 1, "status": "BUILDING"}, pretty=True))
    config, base_path, templates, selected, scan, sources = prepare_plan(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    digest = hashlib.sha256()
    bytes_written = 0
    rows_written = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            with base_path.open("rb") as base:
                for raw in base:
                    destination.write(raw)
                    digest.update(raw)
                    bytes_written += len(raw)
                    rows_written += 1
            for row in render_rows(selected, templates, config["selection"]["selection_salt"]):
                raw = protocol_bytes(row)
                destination.write(raw)
                digest.update(raw)
                bytes_written += len(raw)
                rows_written += 1
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    additional_records = scan["selected_records"]
    require(
        rows_written == config["base_dataset"]["lines"] + additional_records,
        "output row arithmetic mismatch",
    )
    output = {
        "path": relative_path(output_path),
        "sha256": digest.hexdigest(),
        "bytes": bytes_written,
        "lines": rows_written,
    }
    manifest = {
        "schema": "llm-rec.takeover.r0-aligned-video-dataset",
        "schema_version": 1,
        "logical_name": config["logical_name"],
        "output": output,
        "base": config["base_dataset"],
        "additional": {
            "records": additional_records,
            "unique_sids": len(selected),
            "min_views_per_sid": scan["selected_min_views"],
            "max_views_per_sid": scan["selected_max_views"],
        },
        "selection": config["selection"],
        "alignment_scan": scan,
        "sources": sorted(sources, key=lambda item: (item["role"], item["path"])),
        "data_boundary": {
            "allowed": ["official public caption/tag alignment", "official public item seeds"],
            "forbidden": ["platform hidden inputs", "platform hidden generations", "formal evaluation logs"],
        },
    }
    manifest_payload = canonical(manifest, pretty=True)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    atomic_write_new(manifest_path, manifest_payload)
    atomic_write_new(
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        (manifest_hash + "\n").encode("ascii"),
    )
    replace_state(
        state_path,
        canonical(
            {"schema_version": 1, "status": "PASS", "manifest_sha256": manifest_hash},
            pretty=True,
        ),
    )
    return manifest


def verify(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    require(output_path.is_file() and not output_path.is_symlink(), "unsafe output")
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(
        state == {"manifest_sha256": manifest_hash, "schema_version": 1, "status": "PASS"},
        "state is not bound PASS",
    )
    require(
        manifest_path.with_suffix(manifest_path.suffix + ".sha256").read_text(encoding="ascii").strip()
        == manifest_hash,
        "manifest sidecar mismatch",
    )
    require(
        manifest.get("schema") == "llm-rec.takeover.r0-aligned-video-dataset"
        and manifest.get("schema_version") == 1,
        "manifest identity drift",
    )
    output = manifest["output"]
    require(core.resolve_repo_path(output["path"]) == output_path.resolve(), "output path drift")
    require(output_path.stat().st_size == output["bytes"], "output size drift")
    require(sha256_file(output_path) == output["sha256"], "output hash drift")
    config, base_path, templates, selected, scan, sources = prepare_plan(config_path)
    require(manifest["selection"] == config["selection"], "selection drift")
    require(manifest["alignment_scan"] == scan, "alignment scan drift")
    require(
        manifest["sources"] == sorted(sources, key=lambda item: (item["role"], item["path"])),
        "source lock drift",
    )
    expected_rows = config["base_dataset"]["lines"] + scan["selected_records"]
    rows = 0
    with output_path.open("rb") as output_handle, base_path.open("rb") as base_handle:
        for base_raw in base_handle:
            rows += 1
            require(output_handle.readline() == base_raw, f"base prefix drift at row {rows}")
        for expected in render_rows(selected, templates, config["selection"]["selection_salt"]):
            rows += 1
            require(
                output_handle.readline() == protocol_bytes(expected),
                f"additional row drift at row {rows}",
            )
        require(output_handle.read(1) == b"", "unexpected trailing output")
    require(rows == expected_rows == output["lines"], "verified row arithmetic mismatch")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            verify(args.config, args.output, args.manifest, args.state)
        else:
            build(args.config, args.output, args.manifest, args.state)
    except BaseException as exc:
        try:
            preserve_pass = False
            if args.state.is_file() and not args.state.is_symlink():
                try:
                    preserve_pass = (
                        json.loads(args.state.read_text(encoding="utf-8")).get("status") == "PASS"
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
            if not preserve_pass:
                replace_state(
                    args.state,
                    canonical(
                        {
                            "schema_version": 1,
                            "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        pretty=True,
                    ),
                )
        except BaseException:
            pass
        print(f"FAIL: {exc}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
