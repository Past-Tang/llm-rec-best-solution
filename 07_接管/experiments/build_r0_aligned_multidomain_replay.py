#!/usr/bin/env python3
"""Build entity-disjoint non-video R0 alignment data on the formal-best replay.

The base dataset is copied byte-for-byte. New examples come only from the
official position-level caption/SID alignment package and are restricted to
product, ad, and live domains. A deterministic SID-level split prevents the
same item from appearing in both train and evaluation outputs.
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
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as parquet

import build_r0_scale_candidate as core


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "configs/r0_aligned_multidomain_replay_source_config.json"
DEFAULT_TRAIN_OUTPUT = HERE / "data/r0_aligned_multidomain_replay_v1.jsonl"
DEFAULT_EVAL_OUTPUT = HERE / "data/r0_aligned_multidomain_replay_v1_eval.jsonl"
DEFAULT_TRAIN_MANIFEST = (
    HERE / "manifests/r0_aligned_multidomain_replay_v1.manifest.json"
)
DEFAULT_EVAL_MANIFEST = (
    HERE / "manifests/r0_aligned_multidomain_replay_v1_eval.manifest.json"
)
DEFAULT_STATE = (
    HERE / "manifests/r0_aligned_multidomain_replay_v1.build_state.json"
)

LOGICAL_NAME = "r0_aligned_multidomain_replay_v1"
TARGET_DOMAINS = ("prod", "ad", "living")
REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")


class BuildError(RuntimeError):
    """Raised when the data lineage or entity-separation contract fails."""


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
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def require_repo_path(path: Path, role: str) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise BuildError(f"{role} escapes repository: {path}") from error


def resolve_repo_path(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    require_repo_path(path, "source")
    return path


def relative_path(path: Path) -> str:
    require_repo_path(path, "path")
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def file_record(path: Path, role: str, *, digest: str | None = None) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe {role}: {path}")
    return {
        "role": role,
        "path": relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest or sha256_file(path),
    }


def atomic_write_new(path: Path, payload: bytes) -> None:
    require_repo_path(path, "output")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def replace_state(path: Path, payload: bytes) -> None:
    require_repo_path(path, "state")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.is_symlink(), f"refusing symlink state: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    require(record["history"] == [], "record history must be empty")
    return (
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validate_locked_file_spec(
    config: Mapping[str, Any], key: str, count_key: str | None
) -> None:
    spec = config.get(key)
    require(isinstance(spec, dict), f"missing {key}")
    require(isinstance(spec.get("path"), str) and spec["path"], f"invalid {key}.path")
    require(
        isinstance(spec.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]) is not None,
        f"invalid {key}.sha256",
    )
    size = spec.get("size_bytes")
    require(
        isinstance(size, int) and not isinstance(size, bool) and size > 0,
        f"invalid {key}.size_bytes",
    )
    if count_key is not None:
        count = spec.get(count_key)
        require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            f"invalid {key}.{count_key}",
        )


def load_config(path: Path) -> dict[str, Any]:
    require_repo_path(path, "source config")
    require(path.is_file() and not path.is_symlink(), f"unsafe source config: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid source config: {error}") from error
    require(
        isinstance(config, dict) and config.get("schema_version") == 1,
        "unsupported source config",
    )
    require(config.get("logical_name") == LOGICAL_NAME, "logical name drift")
    for key, count_key in (
        ("base_dataset", "lines"),
        ("previous_eval_exclusion", "lines"),
        ("holdout_exclusion", "lines"),
        ("official_alignment", "rows"),
        ("base_manifest", None),
        ("previous_eval_manifest", None),
    ):
        _validate_locked_file_spec(config, key, count_key)

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

    selection = config.get("selection")
    require(isinstance(selection, dict), "missing selection")
    domains = selection.get("target_domains")
    require(
        isinstance(domains, list)
        and tuple(domains) == TARGET_DOMAINS
        and len(domains) == len(set(domains)),
        f"target_domains must be {list(TARGET_DOMAINS)}",
    )
    for key in ("max_views_per_sid", "min_caption_chars", "max_caption_chars"):
        value = selection.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid selection.{key}",
        )
    require(
        selection["max_caption_chars"] >= selection["min_caption_chars"],
        "caption bounds are reversed",
    )
    fraction = selection.get("eval_fraction")
    require(
        isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and 0 < float(fraction) < 1,
        "invalid selection.eval_fraction",
    )
    minimum = selection.get("minimum_unique_sids")
    require(
        isinstance(minimum, dict) and set(minimum) == set(TARGET_DOMAINS),
        "minimum_unique_sids domain drift",
    )
    for domain, value in minimum.items():
        require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"invalid minimum_unique_sids.{domain}",
        )
    require(
        isinstance(selection.get("selection_salt"), str)
        and selection["selection_salt"],
        "missing selection salt",
    )
    return config


def verify_locked_file(
    spec: Mapping[str, Any], role: str
) -> tuple[Path, dict[str, Any]]:
    path = resolve_repo_path(spec["path"])
    record = file_record(path, role)
    require(record["size_bytes"] == spec["size_bytes"], f"{role} size drift")
    require(record["sha256"] == spec["sha256"], f"{role} SHA-256 drift")
    return path, record


def scan_jsonl_exclusions(
    path: Path,
    expected_lines: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
) -> tuple[dict[str, int], str]:
    """Scan both legacy unsuffixed replay rows and current protocol rows."""

    counts = collections.Counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid exclusion JSONL {path}:{line_number}") from error
            require(
                isinstance(record, dict) and tuple(record) == REQUIRED_KEYS,
                f"exclusion row schema/order drift: {path}:{line_number}",
            )
            require(
                record["history"] == []
                and all(isinstance(record[key], str) for key in REQUIRED_KEYS[:-1]),
                f"exclusion row field contract drift: {path}:{line_number}",
            )
            text = record["input"] + "\n" + record["output"]
            excluded_sids.update(match.group(0) for match in core.SID_RE.finditer(text))
            think = core.THINK_HEAD_RE.match(record["output"])
            final = (
                record["output"][think.end() :].strip()
                if think
                else record["output"].strip()
            )
            prompt = record["input"].rstrip()
            mode = core.MODE_SUFFIX_RE.search(prompt)
            if mode is None:
                counts["legacy_rows_without_mode_suffix"] += 1
            else:
                prompt = prompt[: mode.start()]
                counts["protocol_rows_with_mode_suffix"] += 1

            if core.SID_FULL_RE.fullmatch(final):
                counts["forward_grounding_rows"] += 1
                try:
                    _prefix, caption = core.split_generation_prompt(prompt)
                except core.BuildError:
                    counts["unparsed_forward_captions"] += 1
                else:
                    normalized = core.normalize_caption(caption)
                    if normalized:
                        excluded_captions.add(normalized)
                        counts["forward_captions_collected"] += 1
            elif len(list(core.SID_RE.finditer(record["input"]))) == 1:
                normalized = core.normalize_caption(final)
                if normalized and core.SID_LIKE_RE.search(normalized) is None:
                    excluded_captions.add(normalized)
                    counts["reverse_captions_collected"] += 1
            counts["rows"] += 1
    require(counts["rows"] == expected_lines, f"row count drift: {path}")
    return dict(sorted(counts.items())), digest.hexdigest()


def collect_exclusions(
    config: Mapping[str, Any],
) -> tuple[
    Path,
    Path,
    dict[str, list[tuple[str, str]]],
    set[str],
    set[str],
    set[str],
    dict[str, Any],
    list[dict[str, Any]],
]:
    base_path, base_source = verify_locked_file(config["base_dataset"], "base_dataset")
    previous_eval_path, previous_eval_source = verify_locked_file(
        config["previous_eval_exclusion"], "previous_eval_exclusion"
    )
    holdout_path, holdout_source = verify_locked_file(
        config["holdout_exclusion"], "holdout_exclusion"
    )
    alignment_path, alignment_source = verify_locked_file(
        config["official_alignment"], "official_alignment"
    )
    _base_manifest_path, base_manifest_source = verify_locked_file(
        config["base_manifest"], "base_manifest"
    )
    _eval_manifest_path, eval_manifest_source = verify_locked_file(
        config["previous_eval_manifest"], "previous_eval_manifest"
    )

    seed_paths = sorted(REPO_ROOT.glob(config["official_seed_glob"]))
    require(
        len(seed_paths) == config["official_seed_expected_files"],
        "official seed file count drift",
    )
    templates, _reverse, seed_sids, seed_captions, seed_sources = (
        core.load_seed_assets(seed_paths, config["official_seed_expected_rows"])
    )
    base_sids: set[str] = set()
    base_captions: set[str] = set()
    counts: dict[str, Any] = {
        "official_seed_sids": len(seed_sids),
        "official_seed_captions": len(seed_captions),
    }
    base_scan, digest = scan_jsonl_exclusions(
        base_path,
        config["base_dataset"]["lines"],
        base_sids,
        base_captions,
    )
    require(digest == config["base_dataset"]["sha256"], "base changed during scan")
    counts["base_scan"] = base_scan
    counts["base_unique_sids"] = len(base_sids)
    counts["base_unique_captions"] = len(base_captions)

    previous_eval_sids: set[str] = set()
    previous_eval_captions: set[str] = set()
    previous_eval_scan, digest = scan_jsonl_exclusions(
        previous_eval_path,
        config["previous_eval_exclusion"]["lines"],
        previous_eval_sids,
        previous_eval_captions,
    )
    require(
        digest == config["previous_eval_exclusion"]["sha256"],
        "previous evaluation changed during scan",
    )
    counts["previous_eval_scan"] = previous_eval_scan
    holdout_sids: set[str] = set()
    holdout_captions: set[str] = set()
    core.load_holdout_exclusions(
        holdout_path,
        config["holdout_exclusion"]["lines"],
        holdout_sids,
        holdout_captions,
    )
    counts["holdout_rows"] = config["holdout_exclusion"]["lines"]
    hard_excluded_sids = seed_sids | previous_eval_sids | holdout_sids
    seen_entity_sids = base_sids | hard_excluded_sids
    excluded_captions = (
        seed_captions | base_captions | previous_eval_captions | holdout_captions
    )
    counts["hard_excluded_sids_total"] = len(hard_excluded_sids)
    counts["seen_entity_sids_for_eval"] = len(seen_entity_sids)
    counts["excluded_captions_total"] = len(excluded_captions)

    sources = [
        base_source,
        previous_eval_source,
        holdout_source,
        alignment_source,
        base_manifest_source,
        eval_manifest_source,
        *seed_sources,
    ]
    return (
        base_path,
        alignment_path,
        templates,
        hard_excluded_sids,
        seen_entity_sids,
        excluded_captions,
        counts,
        sources,
    )


def collect_alignment_views(
    path: Path,
    expected_rows: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
    minimum_chars: int,
    maximum_chars: int,
) -> tuple[dict[str, dict[str, set[str]]], set[str], dict[str, Any]]:
    require(path.is_file() and not path.is_symlink(), f"unsafe alignment file: {path}")
    source = parquet.ParquetFile(path)
    require(source.metadata.num_rows == expected_rows, "alignment row count drift")
    required_columns = {"sid_token_list", "caption_list"}
    require(required_columns <= set(source.schema_arrow.names), "alignment columns missing")

    views: dict[str, collections.defaultdict[str, set[str]]] = {
        domain: collections.defaultdict(set) for domain in TARGET_DOMAINS
    }
    source_sids = {domain: set() for domain in TARGET_DOMAINS}
    caption_owner: dict[str, str | None] = {}
    domain_counts = {domain: collections.Counter() for domain in TARGET_DOMAINS}
    total_counts = collections.Counter()
    rows = 0

    for batch in source.iter_batches(
        batch_size=1024, columns=["sid_token_list", "caption_list"]
    ):
        values = batch.to_pydict()
        for sids, captions in zip(values["sid_token_list"], values["caption_list"]):
            rows += 1
            require(isinstance(sids, list) and isinstance(captions, list), "list column drift")
            require(
                len(sids) == len(captions),
                f"alignment list length mismatch at row {rows}",
            )
            for sid, caption in zip(sids, captions):
                total_counts["pairs_scanned"] += 1
                match = core.SID_FULL_RE.fullmatch(sid) if isinstance(sid, str) else None
                if match is None or match.group("domain") not in TARGET_DOMAINS:
                    total_counts["non_target_or_invalid_sid_pairs"] += 1
                    continue
                domain = match.group("domain")
                source_sids[domain].add(sid)
                domain_counts[domain]["pairs_scanned"] += 1
                if not isinstance(caption, str) or not caption.strip():
                    domain_counts[domain]["empty_caption_pairs"] += 1
                    continue
                normalized = core.normalize_caption(caption)
                if not minimum_chars <= len(normalized) <= maximum_chars:
                    domain_counts[domain]["caption_length_pairs"] += 1
                    continue
                if core.SID_LIKE_RE.search(normalized):
                    domain_counts[domain]["sid_like_caption_pairs"] += 1
                    continue

                if normalized not in caption_owner:
                    caption_owner[normalized] = sid
                elif caption_owner[normalized] != sid:
                    caption_owner[normalized] = None

                if sid in excluded_sids:
                    domain_counts[domain]["excluded_sid_pairs"] += 1
                    continue
                if normalized in excluded_captions:
                    domain_counts[domain]["excluded_caption_pairs"] += 1
                    continue
                views[domain][sid].add(normalized)

    require(rows == expected_rows, "alignment iteration row count drift")
    ambiguous = {caption for caption, owner in caption_owner.items() if owner is None}
    result: dict[str, dict[str, set[str]]] = {}
    for domain in TARGET_DOMAINS:
        for sid in list(views[domain]):
            views[domain][sid].difference_update(ambiguous)
            if not views[domain][sid]:
                del views[domain][sid]
        domain_counts[domain]["unique_sids_source"] = len(source_sids[domain])
        domain_counts[domain]["unique_sids_with_eligible_views"] = len(views[domain])
        domain_counts[domain]["eligible_captions"] = sum(
            len(captions) for captions in views[domain].values()
        )
        result[domain] = dict(views[domain])

    scan = {
        "rows_scanned": rows,
        "pairs_scanned": total_counts["pairs_scanned"],
        "non_target_or_invalid_sid_pairs": total_counts[
            "non_target_or_invalid_sid_pairs"
        ],
        "globally_ambiguous_captions": len(ambiguous),
        "domains": {
            domain: dict(sorted(domain_counts[domain].items()))
            for domain in TARGET_DOMAINS
        },
    }
    return result, ambiguous, scan


def select_views(
    views: Mapping[str, Mapping[str, Iterable[str]]],
    max_views_per_sid: int,
    salt: str,
) -> dict[str, dict[str, list[str]]]:
    selected: dict[str, dict[str, list[str]]] = {}
    for domain in TARGET_DOMAINS:
        selected[domain] = {}
        for sid in sorted(views[domain]):
            captions = sorted(
                set(views[domain][sid]),
                key=lambda caption: (
                    -len(caption),
                    core.stable_rank(salt, domain, sid, caption),
                    caption,
                ),
            )
            if captions:
                selected[domain][sid] = captions[:max_views_per_sid]
    return selected


def split_eval_sids(
    selected: Mapping[str, Mapping[str, Sequence[str]]],
    eval_fraction: float,
    salt: str,
    seen_entity_sids: set[str] | None = None,
) -> dict[str, set[str]]:
    denominator = 1_000_000
    cutoff = int(eval_fraction * denominator)
    seen_entity_sids = seen_entity_sids or set()
    eval_sids: dict[str, set[str]] = {}
    for domain in TARGET_DOMAINS:
        eval_sids[domain] = {
            sid
            for sid in selected[domain]
            if sid not in seen_entity_sids
            and core.stable_rank(salt, domain, sid, "eval-split") % denominator < cutoff
        }
        require(eval_sids[domain], f"{domain} evaluation split is empty")
        require(
            len(eval_sids[domain]) < len(selected[domain]),
            f"{domain} training split is empty",
        )
    return eval_sids


def render_rows(
    selected: Mapping[str, Mapping[str, Sequence[str]]],
    templates: Mapping[str, Sequence[tuple[str, str]]],
    salt: str,
) -> Iterable[tuple[str, str, dict[str, Any]]]:
    for domain in TARGET_DOMAINS:
        for sid in sorted(selected[domain]):
            match = core.SID_FULL_RE.fullmatch(sid)
            require(match is not None and match.group("domain") == domain, f"SID domain drift: {sid}")
            subtokens = [int(value) for value in core.SUBTOKEN_RE.findall(sid)]
            require(
                len(subtokens) == 3 and all(0 <= value <= 8191 for value in subtokens),
                f"SID out of range: {sid}",
            )
            for index, caption in enumerate(selected[domain][sid]):
                system, prefix = core.stable_choice(
                    templates[domain],
                    f"{salt}\0{domain}\0{sid}\0{index}",
                    f"aligned-multidomain-{domain}",
                )
                yield domain, sid, {
                    "instruction": "",
                    "input": f"{prefix}{caption}/no_think",
                    "output": core.EMPTY_THINK + sid,
                    "system": system,
                    "history": [],
                }


def prepare_plan(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    (
        base_path,
        alignment_path,
        templates,
        hard_excluded_sids,
        seen_entity_sids,
        excluded_captions,
        exclusion_counts,
        sources,
    ) = collect_exclusions(config)
    selection = config["selection"]
    views, ambiguous, alignment_scan = collect_alignment_views(
        alignment_path,
        config["official_alignment"]["rows"],
        hard_excluded_sids,
        excluded_captions,
        selection["min_caption_chars"],
        selection["max_caption_chars"],
    )
    selected = select_views(
        views, selection["max_views_per_sid"], selection["selection_salt"]
    )
    for domain in TARGET_DOMAINS:
        require(
            len(selected[domain]) >= selection["minimum_unique_sids"][domain],
            f"selected only {len(selected[domain])} {domain} SIDs",
        )
    eval_sids = split_eval_sids(
        selected,
        float(selection["eval_fraction"]),
        selection["selection_salt"],
        seen_entity_sids,
    )
    split_counts: dict[str, dict[str, int]] = {}
    for domain in TARGET_DOMAINS:
        train_sids = set(selected[domain]) - eval_sids[domain]
        seen_train_sids = train_sids & seen_entity_sids
        split_counts[domain] = {
            "selected_unique_sids": len(selected[domain]),
            "selected_records": sum(len(value) for value in selected[domain].values()),
            "train_unique_sids": len(train_sids),
            "train_records": sum(len(selected[domain][sid]) for sid in train_sids),
            "seen_entity_train_sids": len(seen_train_sids),
            "novel_entity_train_sids": len(train_sids - seen_entity_sids),
            "eval_unique_sids": len(eval_sids[domain]),
            "eval_records": sum(len(selected[domain][sid]) for sid in eval_sids[domain]),
        }
    sources = [
        file_record(Path(__file__).resolve(), "builder"),
        file_record(config_path, "source_config"),
        *sources,
    ]
    return {
        "config": config,
        "base_path": base_path,
        "templates": templates,
        "selected": selected,
        "eval_sids": eval_sids,
        "exclusions": exclusion_counts,
        "alignment_scan": alignment_scan,
        "ambiguous_count": len(ambiguous),
        "split_counts": split_counts,
        "sources": sorted(sources, key=lambda item: (item["role"], item["path"])),
    }


def write_outputs(
    plan: Mapping[str, Any], train_path: Path, eval_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (train_path, eval_path):
        require_repo_path(path, "output")
        require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    train_fd, train_temporary = tempfile.mkstemp(
        prefix=f".{train_path.name}.", suffix=".tmp", dir=train_path.parent
    )
    eval_fd, eval_temporary = tempfile.mkstemp(
        prefix=f".{eval_path.name}.", suffix=".tmp", dir=eval_path.parent
    )
    train_hash = hashlib.sha256()
    eval_hash = hashlib.sha256()
    train_counts = collections.Counter()
    eval_counts = collections.Counter()

    def emit(
        handle: Any,
        digest: Any,
        counts: collections.Counter,
        payload: bytes,
        domain: str | None,
    ) -> None:
        handle.write(payload)
        digest.update(payload)
        counts["lines"] += 1
        counts["bytes"] += len(payload)
        if domain is not None:
            counts[f"{domain}_records"] += 1

    base_digest = hashlib.sha256()
    base_lines = 0
    try:
        with os.fdopen(train_fd, "wb") as train_handle, os.fdopen(
            eval_fd, "wb"
        ) as eval_handle:
            with plan["base_path"].open("rb") as base_handle:
                for raw in base_handle:
                    base_digest.update(raw)
                    base_lines += 1
                    emit(train_handle, train_hash, train_counts, raw, None)
            config = plan["config"]
            require(
                base_lines == config["base_dataset"]["lines"],
                "base row count changed during copy",
            )
            require(
                base_digest.hexdigest() == config["base_dataset"]["sha256"],
                "base hash changed during copy",
            )
            for domain, sid, record in render_rows(
                plan["selected"],
                plan["templates"],
                config["selection"]["selection_salt"],
            ):
                payload = protocol_bytes(record)
                if sid in plan["eval_sids"][domain]:
                    emit(eval_handle, eval_hash, eval_counts, payload, domain)
                else:
                    emit(train_handle, train_hash, train_counts, payload, domain)
            train_handle.flush()
            eval_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(eval_handle.fileno())
        os.replace(train_temporary, train_path)
        os.replace(eval_temporary, eval_path)
    except BaseException:
        for temporary in (train_temporary, eval_temporary):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise

    train_counts["base_records"] = base_lines
    train_counts["sha256"] = train_hash.hexdigest()
    eval_counts["base_records"] = 0
    eval_counts["sha256"] = eval_hash.hexdigest()
    expected_train = base_lines + sum(
        counts["train_records"] for counts in plan["split_counts"].values()
    )
    expected_eval = sum(
        counts["eval_records"] for counts in plan["split_counts"].values()
    )
    require(train_counts["lines"] == expected_train, "train row arithmetic mismatch")
    require(eval_counts["lines"] == expected_eval, "evaluation row arithmetic mismatch")
    return dict(sorted(train_counts.items())), dict(sorted(eval_counts.items()))


def build(
    config_path: Path,
    train_path: Path,
    eval_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    for path in (
        train_path,
        eval_path,
        train_manifest_path,
        eval_manifest_path,
        train_manifest_path.with_suffix(train_manifest_path.suffix + ".sha256"),
        eval_manifest_path.with_suffix(eval_manifest_path.suffix + ".sha256"),
    ):
        require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    replace_state(
        state_path,
        canonical({"schema_version": 1, "status": "BUILDING"}, pretty=True),
    )
    plan = prepare_plan(config_path)
    train_counts, eval_counts = write_outputs(plan, train_path, eval_path)
    common = {
        "schema": "llm-rec.takeover.r0-aligned-multidomain-replay-dataset",
        "schema_version": 1,
        "logical_name": LOGICAL_NAME,
        "selection": plan["config"]["selection"],
        "exclusions": plan["exclusions"],
        "alignment_scan": plan["alignment_scan"],
        "globally_ambiguous_captions": plan["ambiguous_count"],
        "split_counts": plan["split_counts"],
        "sources": plan["sources"],
        "data_boundary": {
            "allowed": [
                "official public position-level caption/SID alignment",
                "official public item seeds",
                "previous audited training and public holdout artifacts",
            ],
            "forbidden": [
                "platform hidden inputs",
                "platform hidden generations",
                "formal evaluation logs",
            ],
        },
    }
    train_manifest = {
        **common,
        "split": "train",
        "output": {"path": relative_path(train_path), **train_counts},
    }
    eval_manifest = {
        **common,
        "split": "evaluation",
        "output": {"path": relative_path(eval_path), **eval_counts},
    }
    train_payload = canonical(train_manifest, pretty=True)
    eval_payload = canonical(eval_manifest, pretty=True)
    train_manifest_hash = hashlib.sha256(train_payload).hexdigest()
    eval_manifest_hash = hashlib.sha256(eval_payload).hexdigest()
    atomic_write_new(train_manifest_path, train_payload)
    atomic_write_new(eval_manifest_path, eval_payload)
    atomic_write_new(
        train_manifest_path.with_suffix(train_manifest_path.suffix + ".sha256"),
        (train_manifest_hash + "\n").encode("ascii"),
    )
    atomic_write_new(
        eval_manifest_path.with_suffix(eval_manifest_path.suffix + ".sha256"),
        (eval_manifest_hash + "\n").encode("ascii"),
    )
    replace_state(
        state_path,
        canonical(
            {
                "schema_version": 1,
                "status": "PASS",
                "train_manifest_sha256": train_manifest_hash,
                "eval_manifest_sha256": eval_manifest_hash,
            },
            pretty=True,
        ),
    )
    return {
        "status": "PASS",
        "train": train_counts,
        "evaluation": eval_counts,
        "split_counts": plan["split_counts"],
    }


def verify_output(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    output = manifest.get("output")
    require(isinstance(output, dict), "manifest output missing")
    require(resolve_repo_path(output["path"]) == path.resolve(), "output path drift")
    require(path.is_file() and not path.is_symlink(), f"unsafe output: {path}")
    digest = hashlib.sha256()
    lines = 0
    domain_records = collections.Counter()
    base_records = int(output["base_records"])
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid output row {path}:{line_number}") from error
            require(
                isinstance(record, dict) and tuple(record) == REQUIRED_KEYS,
                f"output schema drift at {path}:{line_number}",
            )
            require(record["history"] == [], f"output history drift at {path}:{line_number}")
            if line_number > base_records:
                response = record["output"]
                require(
                    isinstance(response, str) and response.startswith(core.EMPTY_THINK),
                    f"new output response drift at {path}:{line_number}",
                )
                sid = response[len(core.EMPTY_THINK) :].strip()
                match = core.SID_FULL_RE.fullmatch(sid)
                require(
                    match is not None and match.group("domain") in TARGET_DOMAINS,
                    f"new output SID drift at {path}:{line_number}",
                )
                domain_records[match.group("domain")] += 1
            lines += 1
    actual = {
        "lines": lines,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    require(
        actual == {key: output[key] for key in ("lines", "bytes", "sha256")},
        f"output lock mismatch: {actual}",
    )
    for domain in TARGET_DOMAINS:
        require(
            domain_records[domain] == output.get(f"{domain}_records", 0),
            f"{domain} output count drift",
        )
    return actual


def verify(
    train_path: Path,
    eval_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    train_bytes = train_manifest_path.read_bytes()
    eval_bytes = eval_manifest_path.read_bytes()
    train_hash = hashlib.sha256(train_bytes).hexdigest()
    eval_hash = hashlib.sha256(eval_bytes).hexdigest()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    require(
        state
        == {
            "eval_manifest_sha256": eval_hash,
            "schema_version": 1,
            "status": "PASS",
            "train_manifest_sha256": train_hash,
        },
        "state is not a bound PASS",
    )
    require(
        train_manifest_path.with_suffix(train_manifest_path.suffix + ".sha256")
        .read_text(encoding="ascii")
        .strip()
        == train_hash,
        "train manifest sidecar mismatch",
    )
    require(
        eval_manifest_path.with_suffix(eval_manifest_path.suffix + ".sha256")
        .read_text(encoding="ascii")
        .strip()
        == eval_hash,
        "eval manifest sidecar mismatch",
    )
    train_manifest = json.loads(train_bytes)
    eval_manifest = json.loads(eval_bytes)
    for manifest, split in ((train_manifest, "train"), (eval_manifest, "evaluation")):
        require(
            manifest.get("schema")
            == "llm-rec.takeover.r0-aligned-multidomain-replay-dataset"
            and manifest.get("schema_version") == 1
            and manifest.get("logical_name") == LOGICAL_NAME
            and manifest.get("split") == split,
            f"{split} manifest identity drift",
        )
        for source in manifest["sources"]:
            source_path = resolve_repo_path(source["path"])
            current = file_record(source_path, source["role"])
            require(current == source, f"source drift: {source['path']}")
    return {
        "status": "PASS",
        "train": verify_output(train_path, train_manifest),
        "evaluation": verify_output(eval_path, eval_manifest),
    }


def main(argv: Sequence[str] | None = None) -> int:
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
            result = build(
                args.config.resolve(),
                args.train_output.resolve(),
                args.eval_output.resolve(),
                args.train_manifest.resolve(),
                args.eval_manifest.resolve(),
                args.state.resolve(),
            )
        else:
            result = verify(
                args.train_output.resolve(),
                args.eval_output.resolve(),
                args.train_manifest.resolve(),
                args.eval_manifest.resolve(),
                args.state.resolve(),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as error:
        if args.command == "build":
            try:
                preserve_pass = False
                if args.state.is_file() and not args.state.is_symlink():
                    try:
                        preserve_pass = (
                            json.loads(args.state.read_text(encoding="utf-8")).get("status")
                            == "PASS"
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
                                "error": f"{type(error).__name__}: {error}",
                            },
                            pretty=True,
                        ),
                    )
            except BaseException:
                pass
        print(f"FAIL: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
