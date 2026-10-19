#!/usr/bin/env python3
"""Build a keyed, multi-caption video R0 dataset from public sources only.

The public PID-to-SID and PID-to-caption parquet files are independently
partitioned, so this builder deliberately joins on (domain, pid).  A caption
that is observed for more than one SID is rejected from the candidate pool.
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

import build_r0_scale_candidate as core


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_ROOT = REPO_ROOT / "02_数据/data/Explorer_LLM_Rec_Competition/data"
SID_DIR = DATA_ROOT / "OneReason_Pid2Sid"
CAPTION_DIR = DATA_ROOT / "OneReason_Pid2Caption"
SID_DOMAIN = {
    "live": "living",
    "goods": "prod",
    "video/ad": "ad",
    "video/video": "video",
}
VIDEO_SID_RE = re.compile(r"^<\|video_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$")
REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True, "allow_nan": False}
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


def atomic_write(path: Path, payload: bytes, *, allow_identical: bool = False) -> None:
    require_repo_path(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BuildError(f"refusing symlink output: {path}")
    if path.exists():
        if allow_identical and path.read_bytes() == payload:
            return
        raise BuildError(f"refusing to overwrite existing output: {path}")
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


def replace_write(path: Path, payload: bytes) -> None:
    """Atomically replace a state marker owned by this builder."""
    require_repo_path(path, "state")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BuildError(f"refusing symlink state: {path}")
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


def source_file_records(paths: Iterable[Path], role: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {path}")
        records.append(
            {
                "role": role,
                "path": str(path.resolve().relative_to(REPO_ROOT.resolve())),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def load_source_config(path: Path) -> dict[str, Any]:
    require_repo_path(path, "source config")
    require(path.is_file() and not path.is_symlink(), f"unsafe source config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"invalid source config: {path}") from exc
    require(isinstance(config, dict) and config.get("schema_version") == 1, "unsupported source config")
    require(isinstance(config.get("logical_name"), str) and config["logical_name"], "logical_name is missing")
    selection = config.get("selection")
    require(isinstance(selection, dict), "selection is missing")
    for key in ("candidate_pool_size", "target_unique_sids", "max_views_per_sid"):
        value = selection.get(key)
        require(not isinstance(value, bool) and isinstance(value, int) and value > 0, f"invalid selection.{key}")
    require(selection["max_views_per_sid"] >= 2, "selection.max_views_per_sid must be at least 2")
    require(selection["target_unique_sids"] <= selection["candidate_pool_size"], "target exceeds pool")
    minimum = selection.get("min_caption_chars")
    maximum = selection.get("max_caption_chars")
    require(
        not isinstance(minimum, bool)
        and isinstance(minimum, int)
        and minimum > 0
        and not isinstance(maximum, bool)
        and isinstance(maximum, int)
        and maximum >= minimum,
        "invalid caption bounds",
    )
    require(
        isinstance(selection.get("selection_salt"), str) and selection["selection_salt"],
        "invalid selection.selection_salt",
    )
    require(
        isinstance(selection.get("max_caption_share"), (int, float))
        and 0 < float(selection["max_caption_share"])
        <= 1,
        "invalid selection.max_caption_share",
    )
    require(core.resolve_repo_path(config["pid2sid"]["path"]) == SID_DIR.resolve(), "Pid2Sid path drift")
    require(
        core.resolve_repo_path(config["pid2caption"]["path"]) == CAPTION_DIR.resolve(),
        "Pid2Caption path drift",
    )
    return config


def collect_exclusions(
    config: Mapping[str, Any],
) -> tuple[
    set[str],
    set[str],
    list[dict[str, Any]],
    dict[str, list[tuple[str, str]]],
]:
    excluded_sids: set[str] = set()
    excluded_captions: set[str] = set()
    sources: list[dict[str, Any]] = []
    base = config["base_dataset"]
    extra = config["exclusion_dataset"]
    holdout = config["holdout_exclusion"]
    for spec, role in ((base, "base_dataset"), (extra, "exclusion_dataset"), (holdout, "holdout_exclusion")):
        path = core.resolve_repo_path(spec["path"])
        sources.append(core.verify_locked_file(path, spec, role))
    seed_paths = sorted(REPO_ROOT.glob(config["official_seed_glob"]))
    require(len(seed_paths) == config["official_seed_expected_files"], "official seed file count drift")
    generation, _reverse, seed_sids, seed_caps, seed_sources = core.load_seed_assets(
        seed_paths, config["official_seed_expected_rows"]
    )
    excluded_sids.update(seed_sids)
    excluded_captions.update(seed_caps)
    sources.extend(seed_sources)
    for spec in (holdout, base, extra):
        path = core.resolve_repo_path(spec["path"])
        if spec is holdout:
            core.load_holdout_exclusions(path, spec["lines"], excluded_sids, excluded_captions)
        else:
            rows, digest = core.scan_jsonl_exclusions(path, spec["lines"], excluded_sids, excluded_captions)
            require(rows == spec["lines"] and digest == spec["sha256"], f"source drift while scanning {path}")
    return excluded_sids, excluded_captions, sources, generation


def iter_lut_candidates(
    path: Path,
    expected_rows: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
    salt: str,
    max_caption_share: float,
    min_chars: int,
    max_chars: int,
) -> tuple[list[tuple[int, str, str, int, int]], dict[str, int]]:
    columns = ["sid", "domain", "n_pid", "caption_top", "caption_share", "caption_n_distinct"]
    candidates: list[tuple[int, str, str, int, int]] = []
    counts = collections.Counter()
    rows = 0
    lut = parquet.ParquetFile(path)
    require(set(columns).issubset(lut.schema_arrow.names), "LUT lacks multi-view columns")
    for batch in lut.iter_batches(batch_size=262144, columns=columns):
        values = batch.to_pydict()
        for sid, domain, n_pid, caption, share, n_distinct in zip(*(values[key] for key in columns)):
            rows += 1
            if domain != "video":
                continue
            if not isinstance(sid, str) or VIDEO_SID_RE.fullmatch(sid) is None:
                continue
            if sid in excluded_sids or not isinstance(caption, str) or not caption:
                continue
            if not isinstance(n_pid, int) or n_pid < 2 or not isinstance(n_distinct, int) or n_distinct < 2:
                continue
            if not isinstance(share, (int, float)) or float(share) > max_caption_share:
                continue
            caption_norm = core.normalize_caption(caption)
            if not min_chars <= len(caption_norm) <= max_chars or caption_norm in excluded_captions:
                continue
            if core.SID_LIKE_RE.search(caption_norm):
                continue
            rank = core.stable_rank(salt, "video", sid, caption_norm)
            candidates.append((rank, sid, caption_norm, n_pid, n_distinct))
            counts["eligible"] += 1
    require(rows == expected_rows, f"LUT rows {rows} != {expected_rows}")
    counts["rows_scanned"] = rows
    return candidates, dict(sorted(counts.items()))


def build_video_pid_map(paths: list[Path], expected_rows: int) -> tuple[dict[int, str], dict[str, int]]:
    mapping: dict[int, str] = {}
    domain_counts = collections.Counter()
    rows = 0
    for path in sorted(paths):
        table = parquet.read_table(path, columns=["pid", "domain", "sid_three"])
        for pid, raw_domain, triple in zip(
            table["pid"].to_pylist(), table["domain"].to_pylist(), table["sid_three"].to_pylist()
        ):
            rows += 1
            domain = SID_DOMAIN.get(raw_domain)
            require(domain is not None, f"unknown source domain {raw_domain!r}")
            domain_counts[domain] += 1
            if domain != "video":
                continue
            require(isinstance(pid, int), "PID is not int")
            require(isinstance(triple, list) and len(triple) == 3, "SID triple is malformed")
            values = [int(round(value)) for value in triple]
            require(all(0 <= value <= 8191 for value in values), "SID component out of range")
            sid = f"<|video_begin|><s_a_{values[0]}><s_b_{values[1]}><s_c_{values[2]}>"
            previous = mapping.setdefault(pid, sid)
            require(previous == sid, f"conflicting SID for video PID {pid}")
    require(rows == expected_rows, f"Pid2Sid rows {rows} != {expected_rows}")
    require(len(mapping) == domain_counts["video"], "duplicate video PID detected")
    return mapping, dict(sorted(domain_counts.items()))


def collect_caption_views(
    paths: list[Path],
    pid_to_sid: Mapping[int, str],
    candidate_sids: set[str],
    excluded_captions: set[str],
    expected_rows: int,
    min_chars: int,
    max_chars: int,
) -> tuple[dict[str, set[str]], set[str], dict[str, int]]:
    views: dict[str, set[str]] = collections.defaultdict(set)
    caption_owner: dict[str, str] = {}
    ambiguous: set[str] = set()
    rows = mapped = excluded = 0
    for path in sorted(paths):
        table = parquet.read_table(path, columns=["pid", "domain", "caption"])
        for pid, raw_domain, caption in zip(
            table["pid"].to_pylist(), table["domain"].to_pylist(), table["caption"].to_pylist()
        ):
            rows += 1
            domain = SID_DOMAIN.get(raw_domain)
            require(domain is not None, f"unknown caption domain {raw_domain!r}")
            if domain != "video":
                continue
            sid = pid_to_sid.get(pid)
            require(sid is not None, f"caption PID missing from Pid2Sid: {pid}")
            if not isinstance(caption, str) or not caption:
                continue
            caption_norm = core.normalize_caption(caption)
            if not min_chars <= len(caption_norm) <= max_chars or core.SID_LIKE_RE.search(caption_norm):
                continue
            if caption_norm in excluded_captions:
                excluded += 1
                continue
            mapped += 1
            owner = caption_owner.get(caption_norm)
            if owner is None:
                caption_owner[caption_norm] = sid
            elif owner != sid:
                ambiguous.add(caption_norm)
            if sid in candidate_sids:
                views[sid].add(caption_norm)
    require(rows == expected_rows, f"Pid2Caption rows {rows} != {expected_rows}")
    for caption in ambiguous:
        caption_owner.pop(caption, None)
    for sid, captions in list(views.items()):
        views[sid] = {caption for caption in captions if caption_owner.get(caption) == sid}
    return views, ambiguous, {
        "rows_scanned": rows,
        "mapped_video_captions": mapped,
        "excluded_caption_hits": excluded,
        "ambiguous_captions": len(ambiguous),
    }


def render_rows(
    selected: Mapping[str, list[str]],
    templates: list[tuple[str, str]],
    salt: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in sorted(selected):
        for caption in sorted(selected[sid], key=lambda value: (core.stable_rank(salt, "view", sid, value), value)):
            system, prefix = core.stable_choice(templates, f"{sid}\0{caption}", "multiview-video-forward")
            rows.append(
                {
                    "instruction": "",
                    "input": f"{prefix}{caption}/no_think",
                    "output": core.EMPTY_THINK + sid,
                    "system": system,
                    "history": [],
                }
            )
    return rows


def parse_multiview_row(
    row: Mapping[str, Any],
    templates: list[tuple[str, str]],
    min_chars: int,
    max_chars: int,
) -> tuple[str, str]:
    require(tuple(row) == REQUIRED_KEYS, "additional row key order drift")
    require(row["instruction"] == "" and row["history"] == [], "additional row envelope drift")
    system = row["system"]
    prompt = row["input"]
    response = row["output"]
    require(all(isinstance(value, str) for value in (system, prompt, response)), "additional row has non-string fields")
    require(prompt.endswith("/no_think"), "additional prompt mode drift")
    matches = {
        prompt[len(prefix) : -len("/no_think")]
        for template_system, prefix in templates
        if system == template_system and prompt.startswith(prefix)
    }
    require(len(matches) == 1, "additional prompt does not match one public template")
    caption = core.normalize_caption(matches.pop())
    require(min_chars <= len(caption) <= max_chars, "additional caption length drift")
    require(core.SID_LIKE_RE.search(caption) is None, "additional caption contains SID-like text")
    require(response.startswith(core.EMPTY_THINK), "additional response think prefix drift")
    sid = response[len(core.EMPTY_THINK) :]
    require(VIDEO_SID_RE.fullmatch(sid) is not None, "additional response SID drift")
    return sid, caption


def build(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    config = load_source_config(config_path)
    sidecar_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    artifact_paths = (output_path, manifest_path, sidecar_path, state_path)
    for path in artifact_paths:
        require_repo_path(path, "build artifact")
    require(len({path.resolve() for path in artifact_paths}) == len(artifact_paths), "build artifact paths overlap")
    for path in (output_path, manifest_path, sidecar_path):
        require(not path.exists(), f"refusing to overwrite {path}")
    if state_path.exists():
        require(state_path.is_file() and not state_path.is_symlink(), "unsafe existing state")
        try:
            prior_state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BuildError("cannot validate existing build state") from exc
        require(prior_state.get("status") == "FAILED", "only a FAILED build state may be retried")
    replace_write(state_path, canonical({"schema_version": 1, "status": "BUILDING"}, pretty=True))
    excluded_sids, excluded_captions, sources, generation = collect_exclusions(config)
    lut_path = core.resolve_repo_path(config["public_lut"]["path"])
    sources.append(core.verify_locked_file(lut_path, config["public_lut"], "public_lut"))
    candidates, selection_scan = iter_lut_candidates(
        lut_path,
        config["public_lut"]["rows"],
        excluded_sids,
        excluded_captions,
        config["selection"]["selection_salt"],
        float(config["selection"]["max_caption_share"]),
        config["selection"]["min_caption_chars"],
        config["selection"]["max_caption_chars"],
    )
    candidates.sort(key=lambda item: (item[0], item[1]))
    pool = candidates[: config["selection"]["candidate_pool_size"]]
    require(len(pool) == config["selection"]["candidate_pool_size"], "multi-view candidate pool is too small")
    candidate_sids = {item[1] for item in pool}
    sid_paths = sorted(SID_DIR.glob("part-*.parquet"))
    caption_paths = sorted(CAPTION_DIR.glob("part-*.parquet"))
    require(len(sid_paths) == config["pid2sid"]["files"], "Pid2Sid file count drift")
    require(len(caption_paths) == config["pid2caption"]["files"], "Pid2Caption file count drift")
    pid_to_sid, domain_counts = build_video_pid_map(sid_paths, config["pid2sid"]["rows"])
    views, ambiguous, caption_scan = collect_caption_views(
        caption_paths,
        pid_to_sid,
        candidate_sids,
        excluded_captions,
        config["pid2caption"]["rows"],
        config["selection"]["min_caption_chars"],
        config["selection"]["max_caption_chars"],
    )
    eligible = [sid for sid, values in views.items() if len(values) >= 2]
    eligible.sort(key=lambda sid: (core.stable_rank(config["selection"]["selection_salt"], "selected", sid, ""), sid))
    target = config["selection"]["target_unique_sids"]
    require(len(eligible) >= target, f"only {len(eligible)} eligible multi-view SIDs, need {target}")
    selected_sids = eligible[:target]
    selected: dict[str, list[str]] = {}
    for sid in selected_sids:
        values = sorted(views[sid], key=lambda value: (core.stable_rank(config["selection"]["selection_salt"], "view", sid, value), value))
        selected[sid] = values[: config["selection"]["max_views_per_sid"]]
        require(len(selected[sid]) >= 2, f"selected SID lost views: {sid}")
    additional_rows = render_rows(selected, generation["video"], config["selection"]["selection_salt"])
    require(len(additional_rows) == sum(len(values) for values in selected.values()), "render count mismatch")

    base_spec = config["base_dataset"]
    base_path = core.resolve_repo_path(base_spec["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    digest = hashlib.sha256()
    bytes_written = lines = 0
    try:
        with os.fdopen(fd, "wb") as destination, base_path.open("rb") as source:
            for raw in source:
                destination.write(raw); digest.update(raw); bytes_written += len(raw); lines += 1
            for row in additional_rows:
                raw = core.protocol_record_bytes(row)
                destination.write(raw); digest.update(raw); bytes_written += len(raw); lines += 1
            destination.flush(); os.fsync(destination.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    source_records = sources + source_file_records(sid_paths, "public_pid2sid") + source_file_records(caption_paths, "public_pid2caption")
    manifest = {
        "schema": "llm-rec.takeover.r0-multiview-dataset",
        "schema_version": 1,
        "logical_name": config["logical_name"],
        "output": {"path": str(output_path.resolve().relative_to(REPO_ROOT.resolve())), "bytes": bytes_written, "lines": lines, "sha256": digest.hexdigest()},
        "base": {"path": str(base_path.resolve().relative_to(REPO_ROOT.resolve())), "lines": config["base_dataset"]["lines"], "sha256": config["base_dataset"]["sha256"]},
        "additional": {"records": len(additional_rows), "unique_sids": len(selected_sids), "min_views_per_sid": min(map(len, selected.values())), "max_views_per_sid": max(map(len, selected.values()))},
        "selection": config["selection"],
        "selection_scan": selection_scan,
        "caption_scan": caption_scan,
        "domain_counts": domain_counts,
        "ambiguous_caption_count": len(ambiguous),
        "exclusions": {"excluded_sid_count": len(excluded_sids), "excluded_caption_count": len(excluded_captions)},
        "source_files": source_records,
        "data_boundary": {"allowed": ["public Pid2Sid", "public Pid2Caption", "public sid_semantic_lut", "official public item seeds"], "forbidden": ["platform hidden inputs", "platform hidden generations", "formal evaluation logs"]},
    }
    manifest_bytes = canonical(manifest, pretty=True)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(manifest_path, manifest_bytes)
    atomic_write(sidecar_path, (manifest_hash + "\n").encode("ascii"))
    replace_write(state_path, canonical({"schema_version": 1, "status": "PASS", "manifest_sha256": manifest_hash}, pretty=True))
    return manifest


def verify(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    config = load_source_config(config_path)
    artifact_paths = (
        output_path,
        manifest_path,
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        state_path,
    )
    for path in artifact_paths:
        require_repo_path(path, "verification artifact")
    require(len({path.resolve() for path in artifact_paths}) == len(artifact_paths), "verification artifact paths overlap")
    for path, role in ((output_path, "output"), (manifest_path, "manifest"), (state_path, "state")):
        require(path.is_file() and not path.is_symlink(), f"unsafe or missing {role}: {path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    require(state == {"manifest_sha256": manifest_hash, "schema_version": 1, "status": "PASS"}, "state is not bound PASS")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    require(sidecar.is_file() and not sidecar.is_symlink(), "unsafe or missing manifest sidecar")
    require(sidecar.read_text(encoding="ascii").strip() == manifest_hash, "manifest sidecar mismatch")
    require(
        isinstance(manifest, dict)
        and manifest.get("schema") == "llm-rec.takeover.r0-multiview-dataset"
        and manifest.get("schema_version") == 1
        and manifest.get("logical_name") == config["logical_name"],
        "manifest identity drift",
    )
    output = manifest["output"]
    expected_output_path = str(output_path.resolve().relative_to(REPO_ROOT.resolve()))
    require(output.get("path") == expected_output_path, "manifest output path drift")
    require(output_path.stat().st_size == output["bytes"] and sha256_file(output_path) == output["sha256"], "output hash/size drift")
    base_spec = config["base_dataset"]
    base_path = core.resolve_repo_path(base_spec["path"])
    core.verify_locked_file(base_path, base_spec, "base_dataset")
    excluded_sids, excluded_captions, _sources, generation = collect_exclusions(config)
    sid_views: collections.Counter[str] = collections.Counter()
    caption_owner: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    rows = additional_rows = 0
    with output_path.open("rb") as output_handle, base_path.open("rb") as base_handle:
        for base_raw in base_handle:
            raw = output_handle.readline()
            require(raw == base_raw, f"base prefix drift at row {rows + 1}")
            rows += 1
        require(rows == base_spec["lines"], "base prefix line count drift")
        for raw in output_handle:
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BuildError(f"invalid additional JSON at row {rows + 1}") from exc
            require(isinstance(row, dict), f"additional row is not an object at {rows + 1}")
            sid, caption = parse_multiview_row(
                row,
                generation["video"],
                config["selection"]["min_caption_chars"],
                config["selection"]["max_caption_chars"],
            )
            require(sid not in excluded_sids, f"excluded SID reintroduced: {sid}")
            require(caption not in excluded_captions, "excluded caption reintroduced")
            owner = caption_owner.setdefault(caption, sid)
            require(owner == sid, "caption is shared across different SIDs")
            require((sid, caption) not in seen_pairs, "duplicate SID-caption pair")
            seen_pairs.add((sid, caption))
            sid_views[sid] += 1
            additional_rows += 1
            rows += 1
    require(rows == output["lines"], "output line count drift")
    require(manifest["selection"] == config["selection"], "selection config drift")
    additional = manifest["additional"]
    target = config["selection"]["target_unique_sids"]
    maximum = config["selection"]["max_views_per_sid"]
    require(additional_rows == additional["records"] == output["lines"] - base_spec["lines"], "additional row count drift")
    require(len(sid_views) == additional["unique_sids"] == target, "additional unique SID count drift")
    require(sid_views and min(sid_views.values()) >= 2 and max(sid_views.values()) <= maximum, "additional view cardinality drift")
    require(
        additional["min_views_per_sid"] == min(sid_views.values())
        and additional["max_views_per_sid"] == max(sid_views.values()),
        "manifest view cardinality drift",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
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
                    preserve_pass = json.loads(args.state.read_text(encoding="utf-8")).get("status") == "PASS"
                except (OSError, UnicodeError, json.JSONDecodeError):
                    preserve_pass = False
            if not preserve_pass:
                replace_write(args.state, canonical({"schema_version": 1, "status": "FAILED", "error": type(exc).__name__ + ": " + str(exc)}, pretty=True))
        except BaseException:
            pass
        print(f"FAIL: {exc}", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
