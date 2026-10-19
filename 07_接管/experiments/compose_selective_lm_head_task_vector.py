#!/usr/bin/env python3
"""Compose a three-source task vector on selected lm_head rows only.

Formula on selected rows::

    output = target + alpha * (expert - ancestor)

Every non-selected lm_head row and every other tensor is copied exactly from
``target``.  The saved artifact is rebuilt and compared tensor-by-tensor before
the PASS manifest is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import merge_model_weights as common


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "selective_task_vector_models"
MANIFEST_ROOT = HERE / "manifests"
LM_HEAD = "lm_head.weight"
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
SID_RANGES = {"a": (151669, 159861), "b": (159861, 168053), "c": (168053, 176245)}


class SelectiveTaskVectorError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise SelectiveTaskVectorError(message)


def parse_groups(text: str) -> tuple[str, ...]:
    raw = tuple(part.strip() for part in text.split(",") if part.strip())
    require(raw and len(raw) == len(set(raw)), "groups must be non-empty and unique")
    require(not (set(raw) - set(SID_RANGES)), "unknown SID group")
    return tuple(group for group in SID_RANGES if group in raw)


def parse_alpha(text: str) -> tuple[Decimal, float]:
    try:
        decimal = Decimal(text)
    except InvalidOperation as error:
        raise SelectiveTaskVectorError("alpha must be decimal") from error
    require(decimal.is_finite() and Decimal("-1") <= decimal <= Decimal("1") and decimal != 0, "alpha outside [-1,1] or zero")
    value = float(decimal)
    require(math.isfinite(value) and value != 0.0, "unsafe alpha representation")
    return decimal, value


def resolve_model(path: Path, role: str) -> Path:
    try:
        return common.resolve_model_dir(path, role)
    except common.MergeError as error:
        raise SelectiveTaskVectorError(str(error)) from error


def verify_configs(dirs: Sequence[Path]) -> dict[str, Any]:
    keys = ("architectures", "hidden_size", "model_type", "num_hidden_layers", "tie_word_embeddings", "vocab_size")
    configs = [common.load_json_object(path / "config.json", "model config") for path in dirs]
    for key in keys:
        require(len({json.dumps(config.get(key), sort_keys=True) for config in configs}) == 1, f"model config differs: {key}")
    return {key: configs[0].get(key) for key in keys}


def compose_head(
    ancestor: torch.Tensor,
    target: torch.Tensor,
    expert: torch.Tensor,
    groups: Sequence[str],
    alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    require(ancestor.ndim == target.ndim == expert.ndim == 2, "lm_head must be rank two")
    require(ancestor.shape == target.shape == expert.shape, "lm_head shape differs")
    require(all(tensor.dtype.is_floating_point for tensor in (ancestor, target, expert)), "lm_head is not floating point")
    require(all(torch.isfinite(tensor).all().item() for tensor in (ancestor, target, expert)), "non-finite lm_head source")
    output = target.clone().contiguous()
    stats: dict[str, Any] = {"selected_rows": 0, "changed_rows_after_cast": 0, "groups": {}}
    for group in groups:
        start, end = SID_RANGES[group]
        require(end <= target.shape[0], f"{group} range exceeds vocabulary")
        expected = (target[start:end].float() + alpha * (expert[start:end].float() - ancestor[start:end].float())).to(target.dtype).contiguous()
        require(torch.isfinite(expected).all().item(), f"non-finite result in group {group}")
        changed = int((expected != target[start:end]).any(dim=1).sum().item())
        output[start:end] = expected
        stats["selected_rows"] += end - start
        stats["changed_rows_after_cast"] += changed
        stats["groups"][group] = {"start": start, "end_exclusive": end, "rows": end - start, "changed_rows_after_cast": changed}
    require(stats["changed_rows_after_cast"] > 0, "task vector has no effective selected-row change")
    return output, stats


def compose_weights(
    ancestor_path: Path,
    target_path: Path,
    expert_path: Path,
    groups: Sequence[str],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(ancestor_path, framework="pt", device="cpu") as ancestor_handle, safe_open(target_path, framework="pt", device="cpu") as target_handle, safe_open(expert_path, framework="pt", device="cpu") as expert_handle:
        names = list(target_handle.keys())
        require(LM_HEAD in names and set(names) == set(ancestor_handle.keys()) == set(expert_handle.keys()), "tensor inventory differs")
        metadata = target_handle.metadata() or {}
        element_count = 0
        dtype_counts: dict[str, int] = {}
        head_stats = None
        for name in names:
            ancestor = ancestor_handle.get_tensor(name)
            target = target_handle.get_tensor(name)
            expert = expert_handle.get_tensor(name)
            require(ancestor.shape == target.shape == expert.shape, f"tensor shape differs: {name}")
            require(torch.isfinite(target).all().item(), f"target tensor non-finite: {name}")
            if name == LM_HEAD:
                tensor, head_stats = compose_head(ancestor, target, expert, groups, alpha)
            else:
                require(torch.isfinite(ancestor).all().item() and torch.isfinite(expert).all().item(), f"source tensor non-finite: {name}")
                tensor = target.clone().contiguous()
            tensors[name] = tensor
            element_count += tensor.numel()
            dtype_name = str(tensor.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    require(head_stats is not None, "lm_head was not composed")
    return tensors, {"tensor_count": len(tensors), "element_count": element_count, "dtype_counts": dict(sorted(dtype_counts.items())), "metadata": metadata, "head": head_stats}


def verify_saved(
    output_path: Path,
    ancestor_path: Path,
    target_path: Path,
    expert_path: Path,
    groups: Sequence[str],
    alpha: float,
    expected_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    rebuilt, inventory = compose_weights(ancestor_path, target_path, expert_path, groups, alpha)
    with safe_open(output_path, framework="pt", device="cpu") as saved:
        require((saved.metadata() or {}) == expected_inventory["metadata"], "saved metadata drift")
        require(list(saved.keys()) == list(rebuilt), "saved tensor order drift")
        for name, expected in rebuilt.items():
            require(torch.equal(saved.get_tensor(name), expected), f"saved tensor differs: {name}")
    del rebuilt
    for key in ("tensor_count", "element_count", "dtype_counts", "head"):
        require(inventory[key] == expected_inventory[key], f"inventory drift: {key}")
    return {key: inventory[key] for key in ("tensor_count", "element_count", "dtype_counts", "head")}


def build(ancestor: Path, target: Path, expert: Path, output_name: str, groups_text: str, alpha_text: str) -> dict[str, Any]:
    require(SAFE_NAME.fullmatch(output_name) is not None, "unsafe output name")
    groups = parse_groups(groups_text)
    decimal_alpha, alpha = parse_alpha(alpha_text)
    ancestor_dir, target_dir, expert_dir = (resolve_model(path, role) for path, role in ((ancestor, "ancestor"), (target, "target"), (expert, "expert")))
    require(len({ancestor_dir, target_dir, expert_dir}) == 3, "source directories must differ")
    compatibility = verify_configs((ancestor_dir, target_dir, expert_dir))
    sources = {role: common.file_record(directory / "model.safetensors") for role, directory in (("ancestor", ancestor_dir), ("target", target_dir), ("expert", expert_dir))}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = (OUTPUT_ROOT / output_name).resolve()
    manifest_path = MANIFEST_ROOT / f"{output_name}.selective_task_vector_manifest.json"
    state_path = MANIFEST_ROOT / f"{output_name}.selective_task_vector_state.json"
    require(not output_dir.exists() and not manifest_path.exists() and not state_path.exists(), "refusing existing output")
    common.atomic_write(state_path, common.canonical_json_bytes({"schema_version": 1, "status": "BUILDING"}))
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=OUTPUT_ROOT))
    try:
        tensors, inventory = compose_weights(ancestor_dir / "model.safetensors", target_dir / "model.safetensors", expert_dir / "model.safetensors", groups, alpha)
        save_file(tensors, temporary / "model.safetensors", metadata=inventory["metadata"])
        del tensors
        for name in common.ANCILLARY_FILES:
            shutil.copy2(target_dir / name, temporary / name)
        verified = verify_saved(temporary / "model.safetensors", ancestor_dir / "model.safetensors", target_dir / "model.safetensors", expert_dir / "model.safetensors", groups, alpha, inventory)
        for role, directory in (("ancestor", ancestor_dir), ("target", target_dir), ("expert", expert_dir)):
            require(common.file_record(directory / "model.safetensors") == sources[role], f"source changed: {role}")
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    report = {
        "schema": "llm-rec.takeover.selective-lm-head-task-vector",
        "schema_version": 1,
        "output_name": output_name,
        "formula": "selected lm_head rows = target + alpha * (expert - ancestor); all other values = target",
        "groups": list(groups),
        "alpha": str(decimal_alpha),
        "sources": {role: {"directory": common.relative_path(directory), "model": sources[role]} for role, directory in (("ancestor", ancestor_dir), ("target", target_dir), ("expert", expert_dir))},
        "compatibility": compatibility,
        "invariants": ["all 310 non-lm_head tensors equal target", "all non-selected lm_head rows equal target", "selected rows obey the declared formula after target-dtype cast", "all output tensors are finite"],
        "output": {"directory": common.relative_path(output_dir), "artifacts": {name: common.file_record(output_dir / name) for name in common.REQUIRED_MODEL_FILES}, "inventory": verified},
        "builder": common.file_record(Path(__file__).resolve()),
        "operations_performed": {"network": False, "training": False, "upload": False, "formal_evaluation": False},
    }
    payload = common.canonical_json_bytes(report)
    digest = hashlib.sha256(payload).hexdigest()
    common.atomic_write(manifest_path, payload)
    common.atomic_write(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), (digest + "\n").encode("ascii"))
    common.atomic_write(state_path, common.canonical_json_bytes({"schema_version": 1, "status": "PASS", "manifest_sha256": digest, "output_model_sha256": report["output"]["artifacts"]["model.safetensors"]["sha256"]}))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ancestor", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--groups", default="a,b,c")
    parser.add_argument("--alpha", required=True)
    args = parser.parse_args()
    try:
        report = build(args.ancestor, args.target, args.expert, args.output_name, args.groups, args.alpha)
    except (SelectiveTaskVectorError, common.MergeError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps({"verdict": "PASS", "output": report["output"]["directory"], "head": report["output"]["inventory"]["head"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
