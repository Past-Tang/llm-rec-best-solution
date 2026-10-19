#!/usr/bin/env python3
"""Build and verify the isolated 100K-LUT R0 scale candidate dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import os
import re
import stat
import struct
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs/r0_scale100_source_config.json"
DEFAULT_OUTPUT = HERE / "data/r0_scale100_protocol.jsonl"
DEFAULT_MANIFEST = HERE / "manifests/r0_scale100_protocol.manifest.json"
DEFAULT_STATE = HERE / "manifests/r0_scale100_protocol.build_state.json"
DOMAINS = ("ad", "living", "prod", "video")
REVERSE_DOMAINS = {"ad", "prod", "video"}
REQUIRED_KEYS = ("instruction", "input", "output", "system", "history")
EMPTY_THINK = "<think>\n</think>\n"
SID_PATTERN = (
    r"<\|(?P<domain>prod|ad|video|living)_begin\|>"
    r"<s_a_\d+><s_b_\d+><s_c_\d+>"
)
SID_RE = re.compile(SID_PATTERN)
SID_FULL_RE = re.compile(rf"^{SID_PATTERN}$")
SID_LIKE_RE = re.compile(r"<\|\w+_begin\|>|<s_[abc]_\d+>")
THINK_HEAD_RE = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)
MODE_SUFFIX_RE = re.compile(r"/(?:no_)?think\s*$")
SUBTOKEN_RE = re.compile(r"<s_[abc]_(\d+)>")
LUT_COLUMNS = ("sid", "domain", "n_pid", "caption_top", "caption_share")


class BuildError(RuntimeError):
    """Raised when the scale-data contract cannot be proven."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any], *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {"ensure_ascii": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def protocol_record_bytes(record: Mapping[str, Any]) -> bytes:
    """Encode one Alpaca record without losing the registry-declared key order."""

    if set(record) != set(REQUIRED_KEYS):
        raise BuildError("protocol record keys differ from the required schema")
    ordered = {key: record[key] for key in REQUIRED_KEYS}
    return (
        json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def resolve_repo_path(path_text: str) -> Path:
    raw = Path(path_text)
    path = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise BuildError(f"path escapes repository: {path_text}") from error
    return path


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def source_record(role: str, path: Path, digest: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"missing {role}: {path}")
    return {
        "role": role,
        "path": relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest or sha256_file(path),
    }


def normalize_caption(text: str) -> str:
    if not isinstance(text, str):
        raise BuildError("caption must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def strip_mode(prompt: str) -> tuple[str, str]:
    stripped = prompt.rstrip()
    match = MODE_SUFFIX_RE.search(stripped)
    if match is None:
        raise BuildError("official prompt has no think-mode suffix")
    return stripped[: match.start()], match.group(0).strip()


def split_generation_prompt(prompt: str) -> tuple[str, str]:
    index = prompt.find("token")
    if not 0 <= index < 70:
        raise BuildError("official generation template cannot be split")
    end = index + len("token")
    separators = '：:.。，, \n"\u201c\u201d'
    while end < len(prompt) and prompt[end] in separators:
        end += 1
    if end == len(prompt):
        raise BuildError("official generation prompt has no caption")
    return prompt[:end], prompt[end:]


def stable_rank(salt: str, domain: str, sid: str, caption: str) -> int:
    payload = "\0".join((salt, domain, sid, caption)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def stable_choice(values: Sequence[Any], identity: str, namespace: str) -> Any:
    if not values:
        raise BuildError(f"empty template pool for {namespace}")
    digest = hashlib.sha256(f"{namespace}\0{identity}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid source config: {error}") from error
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise BuildError("source config schema_version must be 1")

    for name, count_key in (
        ("base_dataset", "lines"),
        ("holdout_exclusion", "lines"),
        ("public_lut", "rows"),
    ):
        spec = config.get(name)
        if not isinstance(spec, dict):
            raise BuildError(f"{name} must be an object")
        if not isinstance(spec.get("path"), str) or not spec["path"]:
            raise BuildError(f"{name}.path must be a non-empty string")
        digest = spec.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BuildError(f"{name}.sha256 must be a lowercase SHA-256")
        count = spec.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise BuildError(f"{name}.{count_key} must be a positive integer")

    for key in ("official_seed_expected_files", "official_seed_expected_rows"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BuildError(f"{key} must be a positive integer")
    seed_glob = config.get("official_seed_glob")
    if (
        not isinstance(seed_glob, str)
        or not seed_glob
        or Path(seed_glob).is_absolute()
        or ".." in Path(seed_glob).parts
    ):
        raise BuildError("official_seed_glob must be a repository-relative glob")

    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise BuildError("selection config must be an object")
    quotas = selection.get("additional_unique_sid_quota")
    if not isinstance(quotas, dict) or set(quotas) != set(DOMAINS):
        raise BuildError("selection quotas must cover exactly four domains")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in quotas.values()):
        raise BuildError("selection quotas must be positive integers")
    fraction = selection.get("top_n_pid_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0 < float(fraction) < 1
    ):
        raise BuildError("top_n_pid_fraction must be between zero and one")
    oversample = selection.get("oversample_factor")
    if isinstance(oversample, bool) or not isinstance(oversample, int) or oversample < 1:
        raise BuildError("oversample_factor must be a positive integer")
    salt = selection.get("selection_salt")
    if not isinstance(salt, str) or not salt:
        raise BuildError("selection_salt must be a non-empty string")
    minimum = selection.get("min_caption_chars")
    maximum = selection.get("max_caption_chars")
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or minimum < 1
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < minimum
    ):
        raise BuildError("caption character bounds are invalid")
    share = selection.get("min_caption_share")
    if (
        isinstance(share, bool)
        or not isinstance(share, (int, float))
        or not math.isfinite(float(share))
        or not 0 <= float(share) <= 1
    ):
        raise BuildError("min_caption_share must be between zero and one")
    return config


def verify_locked_file(path: Path, expected: Mapping[str, Any], role: str) -> dict[str, Any]:
    if not path.is_file():
        raise BuildError(f"missing {role}: {path}")
    digest = sha256_file(path)
    if digest != expected.get("sha256"):
        raise BuildError(f"{role} SHA-256 drift: {path}")
    return source_record(role, path, digest)


def load_seed_assets(
    paths: Sequence[Path], expected_rows: int
) -> tuple[
    dict[str, list[tuple[str, str]]],
    dict[str, list[tuple[str, str, str]]],
    set[str],
    set[str],
    list[dict[str, Any]],
]:
    generation: dict[str, set[tuple[str, str]]] = {domain: set() for domain in DOMAINS}
    reverse: dict[str, set[tuple[str, str, str]]] = {
        domain: set() for domain in REVERSE_DOMAINS
    }
    excluded_sids: set[str] = set()
    excluded_captions: set[str] = set()
    sources: list[dict[str, Any]] = []
    rows = 0
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise BuildError(f"invalid official seed {path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise BuildError(f"official seed is not an object: {path}:{line_number}")
                system = record.get("system")
                prompt = record.get("prompt")
                response = record.get("response")
                if not all(isinstance(value, str) for value in (system, prompt, response)):
                    raise BuildError(f"official seed has non-string fields: {path}:{line_number}")
                prompt_without_mode, mode = strip_mode(prompt)
                think = THINK_HEAD_RE.match(response)
                if think is None:
                    raise BuildError(f"official seed response has no think block: {path}:{line_number}")
                final = response[think.end() :].strip()
                excluded_sids.update(match.group(0) for match in SID_RE.finditer(prompt + response))
                sid_match = SID_FULL_RE.fullmatch(final)
                if sid_match is not None:
                    domain = sid_match.group("domain")
                    prefix, caption = split_generation_prompt(prompt_without_mode)
                    generation[domain].add((system, prefix))
                    excluded_captions.add(normalize_caption(caption))
                else:
                    prompt_sids = list(SID_RE.finditer(prompt_without_mode))
                    if len(prompt_sids) != 1:
                        raise BuildError(f"reverse seed must contain one SID: {path}:{line_number}")
                    sid = prompt_sids[0]
                    domain = sid.group("domain")
                    if domain not in REVERSE_DOMAINS:
                        raise BuildError(f"unexpected reverse seed domain: {domain}")
                    reverse[domain].add(
                        (system, prompt_without_mode[: sid.start()], prompt_without_mode[sid.end() :])
                    )
                    excluded_captions.add(normalize_caption(final))
                if mode == "/no_think" and response[: think.end()] != EMPTY_THINK:
                    raise BuildError("official /no_think response prefix drifted")
                rows += 1
        sources.append(source_record("official_public_item_seed", path, digest.hexdigest()))
    if rows != expected_rows:
        raise BuildError(f"official seed row count {rows} != {expected_rows}")
    if any(not generation[domain] for domain in DOMAINS) or any(
        not reverse[domain] for domain in REVERSE_DOMAINS
    ):
        raise BuildError("official template pool is incomplete")
    return (
        {domain: sorted(values) for domain, values in generation.items()},
        {domain: sorted(values) for domain, values in reverse.items()},
        excluded_sids,
        excluded_captions,
        sources,
    )


def scan_jsonl_exclusions(
    path: Path,
    expected_lines: int,
    excluded_sids: set[str],
    excluded_captions: set[str],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid base JSONL {path}:{line_number}") from error
            if not isinstance(record, dict) or tuple(record) != REQUIRED_KEYS:
                raise BuildError(f"base row schema/order drift: {path}:{line_number}")
            if record["history"] != [] or not all(
                isinstance(record[key], str) for key in REQUIRED_KEYS[:-1]
            ):
                raise BuildError(f"base row field contract drift: {path}:{line_number}")
            text = record["input"] + "\n" + record["output"]
            excluded_sids.update(match.group(0) for match in SID_RE.finditer(text))
            think = THINK_HEAD_RE.match(record["output"])
            final = record["output"][think.end() :].strip() if think else record["output"].strip()
            prompt, _mode = strip_mode(record["input"])
            if SID_FULL_RE.fullmatch(final):
                _prefix, caption = split_generation_prompt(prompt)
                caption = normalize_caption(caption)
                if caption:
                    excluded_captions.add(caption)
            elif len(list(SID_RE.finditer(record["input"]))) == 1:
                caption = normalize_caption(final)
                if caption and not SID_LIKE_RE.search(caption):
                    excluded_captions.add(caption)
            rows += 1
    if rows != expected_lines:
        raise BuildError(f"base row count {rows} != {expected_lines}")
    return rows, digest.hexdigest()


def load_holdout_exclusions(
    path: Path, expected_lines: int, excluded_sids: set[str], excluded_captions: set[str]
) -> int:
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BuildError(f"invalid holdout row {line_number}") from error
            sid = record.get("expected_sid")
            caption = record.get("caption")
            if not isinstance(sid, str) or SID_FULL_RE.fullmatch(sid) is None:
                raise BuildError(f"invalid holdout SID at row {line_number}")
            excluded_sids.add(sid)
            excluded_captions.add(normalize_caption(caption))
            rows += 1
    if rows != expected_lines:
        raise BuildError(f"holdout row count {rows} != {expected_lines}")
    return rows


def iter_lut(path: Path) -> Iterator[tuple[Any, ...]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise BuildError("pyarrow is required; activate conda base") from error
    parquet_file = parquet.ParquetFile(path)
    if parquet_file.metadata.num_rows <= 0:
        raise BuildError("public LUT is empty")
    missing = set(LUT_COLUMNS) - set(parquet_file.schema_arrow.names)
    if missing:
        raise BuildError(f"public LUT lacks columns: {sorted(missing)}")
    for batch in parquet_file.iter_batches(batch_size=262_144, columns=list(LUT_COLUMNS)):
        values = batch.to_pydict()
        yield from zip(*(values[column] for column in LUT_COLUMNS))


def select_candidates(
    lut_path: Path,
    selection: Mapping[str, Any],
    excluded_sids: set[str],
    excluded_captions: set[str],
    expected_rows: int,
) -> tuple[dict[str, list[tuple[str, str, int]]], dict[str, Any]]:
    quotas = selection["additional_unique_sid_quota"]
    oversample = int(selection["oversample_factor"])
    salt = selection["selection_salt"]
    top_heaps: dict[str, list[tuple[int, int, str, str, str]]] = {
        domain: [] for domain in DOMAINS
    }
    random_heaps: dict[str, list[tuple[int, str, str, str, int]]] = {
        domain: [] for domain in DOMAINS
    }
    eligible = collections.Counter()
    rejected = collections.Counter()
    rows = 0
    for sid, domain, n_pid, caption, caption_share in iter_lut(lut_path):
        rows += 1
        if domain not in quotas:
            rejected["domain"] += 1
            continue
        if not isinstance(sid, str) or SID_FULL_RE.fullmatch(sid) is None:
            rejected["sid"] += 1
            continue
        if sid in excluded_sids:
            rejected["excluded_sid"] += 1
            continue
        if isinstance(n_pid, bool) or not isinstance(n_pid, int) or n_pid < 1:
            rejected["n_pid"] += 1
            continue
        if (
            isinstance(caption_share, bool)
            or not isinstance(caption_share, (int, float))
            or not math.isfinite(float(caption_share))
            or float(caption_share) < float(selection["min_caption_share"])
        ):
            rejected["caption_share"] += 1
            continue
        if not isinstance(caption, str) or not caption:
            rejected["caption"] += 1
            continue
        caption_norm = normalize_caption(caption)
        if not int(selection["min_caption_chars"]) <= len(caption_norm) <= int(
            selection["max_caption_chars"]
        ):
            rejected["caption_length"] += 1
            continue
        if SID_LIKE_RE.search(caption_norm) or caption_norm in excluded_captions:
            rejected["excluded_caption"] += 1
            continue
        sid_match = SID_FULL_RE.fullmatch(sid)
        if sid_match is None or sid_match.group("domain") != domain:
            rejected["sid_domain"] += 1
            continue
        rank = stable_rank(salt, domain, sid, caption_norm)
        eligible[domain] += 1
        limit = int(quotas[domain]) * oversample
        top_entry = (n_pid, -rank, sid, caption_norm, caption)
        top_heap = top_heaps[domain]
        if len(top_heap) < limit:
            heapq.heappush(top_heap, top_entry)
        elif top_entry > top_heap[0]:
            heapq.heapreplace(top_heap, top_entry)
        random_entry = (-rank, sid, caption_norm, caption, n_pid)
        random_heap = random_heaps[domain]
        if len(random_heap) < limit:
            heapq.heappush(random_heap, random_entry)
        elif random_entry > random_heap[0]:
            heapq.heapreplace(random_heap, random_entry)
    if rows != expected_rows:
        raise BuildError(f"public LUT row count {rows} != {expected_rows}")

    selected: dict[str, list[tuple[str, str, int]]] = {}
    global_sids = set(excluded_sids)
    global_captions = set(excluded_captions)
    for domain in DOMAINS:
        quota = int(quotas[domain])
        top_target = math.floor(quota * float(selection["top_n_pid_fraction"]))
        records: list[tuple[str, str, int]] = []

        def accept(sid: str, caption_norm: str, caption: str, n_pid: int) -> bool:
            if sid in global_sids or caption_norm in global_captions:
                return False
            records.append((sid, caption, n_pid))
            global_sids.add(sid)
            global_captions.add(caption_norm)
            return True

        top_candidates = sorted(
            top_heaps[domain], key=lambda item: (-item[0], -item[1], item[2])
        )
        if top_target > 0:
            for n_pid, _negative_rank, sid, caption_norm, caption in top_candidates:
                accept(sid, caption_norm, caption, n_pid)
                if len(records) >= top_target:
                    break
        random_candidates = sorted(
            random_heaps[domain], key=lambda item: (-item[0], item[1])
        )
        for _negative_rank, sid, caption_norm, caption, n_pid in random_candidates:
            if len(records) >= quota:
                break
            accept(sid, caption_norm, caption, n_pid)
        if len(records) < quota:
            for n_pid, _negative_rank, sid, caption_norm, caption in top_candidates:
                accept(sid, caption_norm, caption, n_pid)
                if len(records) >= quota:
                    break
        if len(records) != quota:
            raise BuildError(f"selected {len(records)}/{quota} for {domain}")
        selected[domain] = records
    return selected, {
        "rows_scanned": rows,
        "eligible_by_domain": dict(sorted(eligible.items())),
        "rejection_counts": dict(sorted(rejected.items())),
        "selected_unique_sids_by_domain": {
            domain: len(records) for domain, records in selected.items()
        },
    }


def render_additional_rows(
    selected: Mapping[str, Sequence[tuple[str, str, int]]],
    generation_templates: Mapping[str, Sequence[tuple[str, str]]],
    reverse_templates: Mapping[str, Sequence[tuple[str, str, str]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = collections.Counter()
    for domain in DOMAINS:
        for sid, caption, _n_pid in selected[domain]:
            subtokens = [int(value) for value in SUBTOKEN_RE.findall(sid)]
            if len(subtokens) != 3 or not all(0 <= value <= 8191 for value in subtokens):
                raise BuildError(f"SID subtokens are out of range: {sid}")
            system, prefix = stable_choice(
                generation_templates[domain], sid, f"forward-{domain}"
            )
            rows.append(
                {
                    "instruction": "",
                    "input": f"{prefix}{caption}/no_think",
                    "output": EMPTY_THINK + sid,
                    "system": system,
                    "history": [],
                }
            )
            counts[f"forward_{domain}"] += 1
            if domain in REVERSE_DOMAINS:
                system, prefix, suffix = stable_choice(
                    reverse_templates[domain], sid, f"reverse-{domain}"
                )
                rows.append(
                    {
                        "instruction": "",
                        "input": f"{prefix}{sid}{suffix}/no_think",
                        "output": EMPTY_THINK + caption,
                        "system": system,
                        "history": [],
                    }
                )
                counts[f"reverse_{domain}"] += 1
    expected_rows = sum(len(values) for values in selected.values()) + sum(
        len(selected[domain]) for domain in REVERSE_DOMAINS
    )
    if len(rows) != expected_rows:
        raise BuildError("rendered row count does not match selected SID count")
    return rows, dict(sorted(counts.items()))


def build(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    atomic_write(state_path, canonical_json_bytes({"schema_version": 1, "status": "BUILDING"}, pretty=True))
    config = load_config(config_path)
    base_spec = config["base_dataset"]
    holdout_spec = config["holdout_exclusion"]
    lut_spec = config["public_lut"]
    base_path = resolve_repo_path(base_spec["path"])
    holdout_path = resolve_repo_path(holdout_spec["path"])
    lut_path = resolve_repo_path(lut_spec["path"])
    sources = [
        source_record("builder", Path(__file__).resolve()),
        source_record("source_config", config_path),
    ]
    sources.append(verify_locked_file(base_path, base_spec, "base_dataset"))
    sources.append(verify_locked_file(holdout_path, holdout_spec, "holdout_exclusion"))
    sources.append(verify_locked_file(lut_path, lut_spec, "public_lut"))

    seed_paths = sorted(REPO_ROOT.glob(config["official_seed_glob"]))
    if len(seed_paths) != config["official_seed_expected_files"]:
        raise BuildError("official seed file count drifted")
    generation, reverse, excluded_sids, excluded_captions, seed_sources = load_seed_assets(
        seed_paths, int(config["official_seed_expected_rows"])
    )
    sources.extend(seed_sources)
    load_holdout_exclusions(
        holdout_path, int(holdout_spec["lines"]), excluded_sids, excluded_captions
    )
    base_rows, base_hash = scan_jsonl_exclusions(
        base_path,
        int(base_spec["lines"]),
        excluded_sids,
        excluded_captions,
    )
    if base_hash != base_spec["sha256"]:
        raise BuildError("base dataset changed during exclusion scan")
    selected, selection_stats = select_candidates(
        lut_path,
        config["selection"],
        excluded_sids,
        excluded_captions,
        int(lut_spec["rows"]),
    )
    additional_rows, render_counts = render_additional_rows(selected, generation, reverse)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as destination:
            with base_path.open("rb") as source:
                for raw in source:
                    destination.write(raw)
                    digest.update(raw)
                    byte_count += len(raw)
            for row in additional_rows:
                raw = protocol_record_bytes(row)
                destination.write(raw)
                digest.update(raw)
                byte_count += len(raw)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    output_rows = base_rows + len(additional_rows)
    output = {
        "path": relative_path(output_path),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "lines": output_rows,
    }
    manifest = {
        "schema": "llm-rec.takeover.r0-scale-dataset",
        "schema_version": 1,
        "logical_name": "r0_scale100_protocol",
        "output": output,
        "base": {"path": relative_path(base_path), "sha256": base_hash, "lines": base_rows},
        "additional": {
            "records": len(additional_rows),
            "unique_sids": sum(len(values) for values in selected.values()),
            "render_counts": render_counts,
        },
        "selection": config["selection"],
        "selection_scan": selection_stats,
        "exclusions": {
            "scope": ["official public seeds", "all base-training SIDs/captions", "public R0 holdout"],
            "excluded_sid_count": len(excluded_sids),
            "excluded_caption_count": len(excluded_captions),
        },
        "sources": sorted(sources, key=lambda item: (item["role"], item["path"])),
        "data_boundary": {
            "allowed": ["public sid_semantic_lut", "official public item seeds"],
            "forbidden": ["platform hidden inputs", "platform hidden generations", "formal evaluation logs"],
        },
    }
    manifest_bytes = canonical_json_bytes(manifest, pretty=True)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    atomic_write(manifest_path, manifest_bytes)
    atomic_write(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), (manifest_hash + "\n").encode("ascii"))
    atomic_write(
        state_path,
        canonical_json_bytes(
            {"schema_version": 1, "status": "PASS", "manifest_sha256": manifest_hash},
            pretty=True,
        ),
    )
    return manifest


def verify(config_path: Path, output_path: Path, manifest_path: Path, state_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid generated artifacts: {error}") from error
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if state.get("status") != "PASS" or state.get("manifest_sha256") != manifest_hash:
        raise BuildError("build state is not bound PASS")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if sidecar.read_text(encoding="ascii").strip() != manifest_hash:
        raise BuildError("manifest sidecar hash mismatch")
    output = manifest.get("output")
    if not isinstance(output, dict) or resolve_repo_path(output["path"]) != output_path.resolve():
        raise BuildError("manifest output path mismatch")
    if output_path.stat().st_size != output["bytes"] or sha256_file(output_path) != output["sha256"]:
        raise BuildError("output size/hash mismatch")
    rows = 0
    with output_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BuildError(f"invalid output JSONL row {line_number}") from error
            if not isinstance(row, dict) or tuple(row) != REQUIRED_KEYS:
                raise BuildError(f"output row schema/order mismatch at {line_number}")
            if row["history"] != [] or not all(isinstance(row[key], str) for key in REQUIRED_KEYS[:-1]):
                raise BuildError(f"output row field mismatch at {line_number}")
            rows += 1
    if rows != output["lines"]:
        raise BuildError("output line count mismatch")
    for source in manifest["sources"]:
        path = resolve_repo_path(source["path"])
        if path.stat().st_size != source["size_bytes"] or sha256_file(path) != source["sha256"]:
            raise BuildError(f"source drift: {path}")
    if manifest["selection"] != config["selection"]:
        raise BuildError("selection config differs from manifest")
    return manifest


def write_failed_state(path: Path, error: BaseException) -> None:
    try:
        atomic_write(
            path,
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                pretty=True,
            ),
        )
    except BaseException:
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    for path, root, label in (
        (config_path, HERE / "configs", "config"),
        (output_path, HERE / "data", "output"),
        (manifest_path, HERE / "manifests", "manifest"),
        (state_path, HERE / "manifests", "state"),
    ):
        try:
            path.relative_to(root.resolve())
        except ValueError:
            print(f"FAIL: {label} path escapes takeover directory", file=os.sys.stderr)
            return 1
    try:
        manifest = (
            verify(config_path, output_path, manifest_path, state_path)
            if args.verify
            else build(config_path, output_path, manifest_path, state_path)
        )
    except (BuildError, OSError, ValueError, KeyError, TypeError) as error:
        if not args.verify:
            write_failed_state(state_path, error)
        print(f"FAIL: {error}", file=os.sys.stderr)
        return 1
    action = "verified" if args.verify else "built"
    print(
        f"PASS: {action} {manifest['output']['lines']} rows; "
        f"sha256={manifest['output']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
