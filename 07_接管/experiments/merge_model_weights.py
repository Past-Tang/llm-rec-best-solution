#!/usr/bin/env python3
"""Create an audited linear weight interpolation between compatible models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MERGED_ROOT = HERE / "merged_models"
MANIFEST_ROOT = HERE / "manifests"
FAQ_PATH = (
    REPO_ROOT
    / "01_文档/LLM-Rec信息与自动化工具/FAQ_extracted_20260714.txt"
)
TRAINING_CONFIG_ROOTS = (
    REPO_ROOT / "04_作战/rft_v2",
    HERE / "configs",
)
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
REQUIRED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
ANCILLARY_FILES = REQUIRED_MODEL_FILES[1:]


class MergeError(RuntimeError):
    """Raised when a model interpolation cannot be proven safe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError as error:
        raise MergeError(f"path escapes repository: {path}") from error


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MergeError(f"required regular file is missing: {path}")
    return {
        "path": relative_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


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


def parse_alpha(text: str) -> tuple[Decimal, float]:
    try:
        decimal = Decimal(text)
    except InvalidOperation as error:
        raise MergeError("alpha must be a decimal number") from error
    if not decimal.is_finite() or not Decimal(0) < decimal < Decimal(1):
        raise MergeError("alpha must be strictly between zero and one")
    value = float(decimal)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise MergeError("alpha cannot be represented safely")
    return decimal, value


def resolve_model_dir(path: Path, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise MergeError(f"{role} model escapes the repository") from error
    if not resolved.is_dir() or resolved.is_symlink():
        raise MergeError(f"{role} model directory is missing or a symlink")
    for name in REQUIRED_MODEL_FILES:
        file_record(resolved / name)
    return resolved


def verify_ancillary_compatibility(base_dir: Path, expert_dir: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in ANCILLARY_FILES:
        base = file_record(base_dir / name)
        expert = file_record(expert_dir / name)
        if base["sha256"] != expert["sha256"] or base["bytes"] != expert["bytes"]:
            raise MergeError(f"model runtime artifact differs: {name}")
        records[name] = base
    return records


def require_finite_json(value: Any, location: str = "value") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MergeError(f"non-finite JSON number at {location}")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            require_finite_json(nested, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            require_finite_json(nested, f"{location}[{index}]")
        return
    raise MergeError(f"unsupported JSON value at {location}: {type(value).__name__}")


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MergeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MergeError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise MergeError(f"{label} must be a JSON object: {path}")
    require_finite_json(value, label)
    return value


def resolve_declared_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{label} is missing")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise MergeError(f"{label} escapes the repository: {resolved}") from error
    return resolved


def find_same_lineage_training(base_dir: Path, expert_dir: Path) -> dict[str, Any]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for root in TRAINING_CONFIG_ROOTS:
        if not root.is_dir():
            continue
        for config_path in sorted(root.glob("*.yaml")):
            if not config_path.is_file() or config_path.is_symlink():
                continue
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                continue
            if not isinstance(config, dict):
                continue
            try:
                initializer = resolve_declared_path(
                    config.get("model_name_or_path"), "model_name_or_path"
                )
                output = resolve_declared_path(config.get("output_dir"), "output_dir")
            except MergeError:
                continue
            if initializer == base_dir and output == expert_dir:
                matches.append((config_path.resolve(), config))
    if len(matches) != 1:
        raise MergeError(
            "expert must have exactly one training config initialized from base; "
            f"found {len(matches)}"
        )

    config_path, config = matches[0]
    if config.get("stage") != "sft" or config.get("finetuning_type") != "full":
        raise MergeError("same-lineage interpolation requires full-parameter SFT")
    if config.get("do_train") is not True:
        raise MergeError("expert config is not a training run")

    trainer_state_path = expert_dir / "trainer_state.json"
    trainer_state = load_json_object(trainer_state_path, "expert trainer state")
    global_step = trainer_state.get("global_step")
    max_steps = trainer_state.get("max_steps")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 1
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or global_step != max_steps
    ):
        raise MergeError("expert trainer state does not prove completed optimization")

    dataset_dir = resolve_declared_path(config.get("dataset_dir"), "dataset_dir")
    registry_path = dataset_dir / "dataset_info.json"
    registry = load_json_object(registry_path, "dataset registry")
    alias = config.get("dataset")
    if not isinstance(alias, str) or not isinstance(registry.get(alias), dict):
        raise MergeError("expert dataset alias is not registered")
    dataset_path = resolve_declared_path(
        registry[alias].get("file_name"), "dataset file"
    )

    result_files = {}
    for name in ("train_results.json", "all_results.json"):
        path = expert_dir / name
        load_json_object(path, f"expert {name}")
        result_files[name] = file_record(path)
    return {
        "classification": "same-lineage checkpoint interpolation",
        "expert_initializer_equals_base": True,
        "training_config": file_record(config_path),
        "training_method": {
            "stage": "sft",
            "finetuning_type": "full",
            "global_step": global_step,
            "max_steps": max_steps,
        },
        "dataset": {
            "alias": alias,
            "registry": file_record(registry_path),
            "artifact": file_record(dataset_path),
        },
        "trainer_state": file_record(trainer_state_path),
        "result_files": result_files,
    }


def lock_official_merge_policy() -> dict[str, Any]:
    if not FAQ_PATH.is_file() or FAQ_PATH.is_symlink():
        raise MergeError("official FAQ extraction is missing or unsafe")
    try:
        lines = FAQ_PATH.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MergeError("official FAQ extraction is unreadable") from error
    matches = [
        line.strip()
        for line in lines
        if "不鼓励使用模型融合进行提分" in line
        and "提供单模型训练方案进行代码审核以及复现" in line
    ]
    if len(set(matches)) != 1:
        raise MergeError(
            "official FAQ must contain exactly one distinct merge-policy statement"
        )
    return {
        "source": file_record(FAQ_PATH),
        "statement": matches[0],
        "classification": "discouraged_not_prohibited",
        "final_review_obligation": "reproducible single-model training solution",
    }


def interpolate_tensor(
    name: str, base: torch.Tensor, expert: torch.Tensor, alpha: float
) -> torch.Tensor:
    if base.shape != expert.shape or base.dtype != expert.dtype:
        raise MergeError(f"tensor layout differs: {name}")
    if not base.dtype.is_floating_point:
        if not torch.equal(base, expert):
            raise MergeError(f"non-floating tensor differs: {name}")
        return base.clone().contiguous()
    if not torch.isfinite(base).all().item() or not torch.isfinite(expert).all().item():
        raise MergeError(f"source tensor is non-finite: {name}")
    merged = torch.lerp(base.float(), expert.float(), alpha).to(base.dtype).contiguous()
    if not torch.isfinite(merged).all().item():
        raise MergeError(f"merged tensor is non-finite: {name}")
    return merged


def load_and_merge_weights(
    base_path: Path, expert_path: Path, alpha: float
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(base_path, framework="pt", device="cpu") as base_handle, safe_open(
        expert_path, framework="pt", device="cpu"
    ) as expert_handle:
        base_names = list(base_handle.keys())
        expert_names = list(expert_handle.keys())
        if base_names != expert_names:
            raise MergeError("safetensors tensor names or ordering differ")
        base_metadata = base_handle.metadata() or {}
        expert_metadata = expert_handle.metadata() or {}
        if base_metadata != expert_metadata:
            raise MergeError("safetensors metadata differs")
        element_count = 0
        dtype_counts: dict[str, int] = {}
        for name in base_names:
            base = base_handle.get_tensor(name)
            expert = expert_handle.get_tensor(name)
            tensors[name] = interpolate_tensor(name, base, expert, alpha)
            element_count += base.numel()
            dtype_name = str(base.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    return tensors, {
        "tensor_count": len(tensors),
        "element_count": element_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "metadata": base_metadata,
    }


def verify_saved_weights(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    tensor_count = 0
    element_count = 0
    dtype_counts: dict[str, int] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        if (handle.metadata() or {}) != expected["metadata"]:
            raise MergeError("saved safetensors metadata drifted")
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all().item():
                raise MergeError(f"saved tensor is non-finite: {name}")
            tensor_count += 1
            element_count += tensor.numel()
            dtype_name = str(tensor.dtype).removeprefix("torch.")
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    actual = {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }
    if actual != {key: expected[key] for key in actual}:
        raise MergeError("saved safetensors inventory differs from the merged inventory")
    return actual


def build_merge(base: Path, expert: Path, output_name: str, alpha_text: str) -> dict[str, Any]:
    if SAFE_NAME.fullmatch(output_name) is None:
        raise MergeError("output name is unsafe")
    decimal_alpha, alpha = parse_alpha(alpha_text)
    base_dir = resolve_model_dir(base, "base")
    expert_dir = resolve_model_dir(expert, "expert")
    if base_dir == expert_dir:
        raise MergeError("base and expert model directories must differ")
    ancillary = verify_ancillary_compatibility(base_dir, expert_dir)
    lineage = find_same_lineage_training(base_dir, expert_dir)
    official_policy = lock_official_merge_policy()
    base_model_lock = file_record(base_dir / "model.safetensors")
    expert_model_lock = file_record(expert_dir / "model.safetensors")

    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    output_dir = (MERGED_ROOT / output_name).resolve()
    manifest_path = MANIFEST_ROOT / f"{output_name}.merge_manifest.json"
    state_path = MANIFEST_ROOT / f"{output_name}.merge_state.json"
    if output_dir.exists() or manifest_path.exists() or state_path.exists():
        raise MergeError("refusing to overwrite an existing merge artifact")
    atomic_write(
        state_path,
        canonical_json_bytes({"schema_version": 1, "status": "BUILDING"}),
    )

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_name}.", dir=MERGED_ROOT))
    try:
        merged, inventory = load_and_merge_weights(
            base_dir / "model.safetensors",
            expert_dir / "model.safetensors",
            alpha,
        )
        save_file(
            merged,
            temporary_dir / "model.safetensors",
            metadata=inventory["metadata"],
        )
        del merged
        for name in ANCILLARY_FILES:
            shutil.copy2(base_dir / name, temporary_dir / name)
        verified = verify_saved_weights(temporary_dir / "model.safetensors", inventory)
        for name in REQUIRED_MODEL_FILES:
            if not (temporary_dir / name).is_file():
                raise MergeError(f"temporary merge lacks {name}")
        if file_record(base_dir / "model.safetensors") != base_model_lock:
            raise MergeError("base weights changed during interpolation")
        if file_record(expert_dir / "model.safetensors") != expert_model_lock:
            raise MergeError("expert weights changed during interpolation")
        verify_ancillary_compatibility(base_dir, expert_dir)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    report = {
        "schema": "llm-rec.takeover.linear-model-merge",
        "schema_version": 1,
        "output_name": output_name,
        "formula": "output = (1 - alpha) * base + alpha * expert",
        "alpha": str(decimal_alpha),
        "base": {
            "directory": relative_path(base_dir),
            "model": base_model_lock,
        },
        "expert": {
            "directory": relative_path(expert_dir),
            "model": expert_model_lock,
        },
        "compatibility": {
            "ancillary_files_byte_identical": True,
            "ancillary_files": ancillary,
            "tensor_names_shapes_dtypes_identical": True,
        },
        "lineage": lineage,
        "output": {
            "directory": relative_path(output_dir),
            "artifacts": {
                name: file_record(output_dir / name) for name in REQUIRED_MODEL_FILES
            },
            "inventory": verified,
            "all_tensors_finite": True,
        },
        "builder": file_record(Path(__file__).resolve()),
        "operations_performed": {
            "network": False,
            "training": False,
            "upload": False,
            "formal_evaluation": False,
        },
        "competition_compliance": {
            "official_policy": official_policy,
            "local_policy_scope": "preliminary same-lineage single-output experiment",
            "formal_submission_eligible_under_local_policy": True,
            "required_acknowledgement": "ALLOW_DISCOURAGED_SAME_LINEAGE_MERGE",
            "reason": (
                "The official FAQ discourages but does not prohibit model merging. "
                "This artifact is restricted to a reproducible interpolation between "
                "one model and its directly fine-tuned descendant."
            ),
        },
    }
    manifest_payload = canonical_json_bytes(report)
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    atomic_write(manifest_path, manifest_payload)
    atomic_write(
        manifest_path.with_suffix(manifest_path.suffix + ".sha256"),
        (manifest_hash + "\n").encode("ascii"),
    )
    atomic_write(
        state_path,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "status": "PASS",
                "manifest_sha256": manifest_hash,
            }
        ),
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--alpha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = MANIFEST_ROOT / f"{args.output_name}.merge_state.json"
    try:
        report = build_merge(args.base, args.expert, args.output_name, args.alpha)
    except (MergeError, OSError, RuntimeError, ValueError) as error:
        if SAFE_NAME.fullmatch(args.output_name):
            try:
                current_status = None
                if state_path.is_file():
                    try:
                        current_status = json.loads(
                            state_path.read_text(encoding="utf-8")
                        ).get("status")
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        current_status = "INVALID"
                if current_status in {None, "BUILDING"}:
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
        f"PASS: merged {report['output_name']} alpha={report['alpha']} "
        f"tensors={report['output']['inventory']['tensor_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
