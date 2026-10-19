#!/usr/bin/env python3
"""Launch one manifest-bound LLaMA-Factory training run with fail-closed state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = Path(__file__).resolve()
EXPERIMENT_ROOT = REPO_ROOT / "07_接管/experiments"
OUTPUT_ROOT = EXPERIMENT_ROOT / "outputs"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
TRAINING_ROOT = (
    REPO_ROOT
    / "02_数据/data/Explorer_LLM_Rec_Competition/demo/LLaMA-Factory"
)
CLI_PATH = TRAINING_ROOT / ".venv/bin/llamafactory-cli"
PYTHON_PATH = TRAINING_ROOT / ".venv/bin/python3"
RESCUE_BEAR_CLI = EXPERIMENT_ROOT / "llamafactory_cli_rescue_bear.py"
RESCUE_BEAR_PATCH = EXPERIMENT_ROOT / "liger_rescue_bear_patch.py"
LIGER_FUSED_CE_PATH = (
    TRAINING_ROOT
    / ".venv/lib/python3.11/site-packages/liger_kernel/ops/fused_linear_cross_entropy.py"
)
RUNTIME_PATHS = (
    TRAINING_ROOT / "pyproject.toml",
    TRAINING_ROOT / "src/llamafactory/cli.py",
    TRAINING_ROOT / "src/llamafactory/launcher.py",
    TRAINING_ROOT / "src/llamafactory/hparams/parser.py",
    TRAINING_ROOT / "src/llamafactory/data/loader.py",
    TRAINING_ROOT / "src/llamafactory/data/template.py",
    TRAINING_ROOT / "src/llamafactory/data/processor/supervised.py",
    TRAINING_ROOT / "src/llamafactory/model/loader.py",
    TRAINING_ROOT / "src/llamafactory/model/adapter.py",
    TRAINING_ROOT / "src/llamafactory/train/sft/workflow.py",
    TRAINING_ROOT / "src/llamafactory/train/sft/trainer.py",
    TRAINING_ROOT / "src/llamafactory/train/trainer_utils.py",
    LIGER_FUSED_CE_PATH,
    RESCUE_BEAR_CLI,
    RESCUE_BEAR_PATCH,
)
REQUIRED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,95}\Z")
ALLOWED_CUDA_VISIBLE_DEVICES = frozenset({"0", "1", "0,1,2,3"})


class LaunchError(RuntimeError):
    """Raised when training cannot be launched without violating the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchError(f"missing required file: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_file_locks(value: Any, location: str = "inputs") -> int:
    """Recheck every nested file record captured before the trainer started."""

    verified = 0
    if isinstance(value, Mapping):
        required = {"path", "size_bytes", "sha256"}
        if required <= set(value):
            path = value.get("path")
            size = value.get("size_bytes")
            digest = value.get("sha256")
            if (
                not isinstance(path, str)
                or not path
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size < 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise LaunchError(f"invalid file lock at {location}")
            actual = file_record(Path(path))
            if actual["size_bytes"] != size or actual["sha256"] != digest:
                raise LaunchError(f"locked input drifted at {location}: {path}")
            verified += 1
        for key, nested in value.items():
            verified += verify_file_locks(nested, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            verified += verify_file_locks(nested, f"{location}[{index}]")
    return verified


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
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


def resolve_within(path_text: str, root: Path, label: str) -> Path:
    path = Path(path_text).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise LaunchError(f"{label} escapes {root}: {path}") from error
    return path


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchError(f"training config does not exist: {path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise LaunchError(f"invalid training config: {error}") from error
    if not isinstance(config, dict):
        raise LaunchError("training config must be a mapping")
    required = {
        "model_name_or_path",
        "dataset",
        "dataset_dir",
        "output_dir",
        "val_size",
        "eval_strategy",
        "overwrite_output_dir",
        "gradient_accumulation_steps",
    }
    missing = required - set(config)
    if missing:
        raise LaunchError(f"training config lacks required fields: {sorted(missing)}")
    if config["overwrite_output_dir"] is not False:
        raise LaunchError("overwrite_output_dir must be false")
    if not isinstance(config["val_size"], (int, float)):
        raise LaunchError("val_size must be numeric")
    eval_alias = config.get("eval_dataset")
    if eval_alias is None:
        if not 0 < float(config["val_size"]) < 1:
            raise LaunchError("val_size must be between zero and one without eval_dataset")
    else:
        if not isinstance(eval_alias, str) or not eval_alias:
            raise LaunchError("eval_dataset must be a non-empty registry alias")
        if float(config["val_size"]) != 0:
            raise LaunchError("val_size must be zero when eval_dataset is provided")
    if config["eval_strategy"] != "steps":
        raise LaunchError("eval_strategy must be steps")
    return config


def bind_dataset_manifest(dataset_path: Path) -> dict[str, Any]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted((EXPERIMENT_ROOT / "manifests").glob("*.manifest.json")):
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LaunchError(f"invalid dataset manifest {manifest_path}: {error}") from error
        output = document.get("output") if isinstance(document, dict) else None
        if not isinstance(output, dict) or not isinstance(output.get("path"), str):
            continue
        declared_path = Path(output["path"])
        if not declared_path.is_absolute():
            declared_path = REPO_ROOT / declared_path
        if declared_path.resolve() == dataset_path.resolve():
            matches.append((manifest_path, output))
    if len(matches) != 1:
        raise LaunchError(
            f"dataset must have exactly one authoritative manifest: "
            f"{dataset_path} has {len(matches)}"
        )
    manifest_path, output = matches[0]
    expected_hash = output.get("sha256")
    expected_bytes = output.get("bytes", output.get("size_bytes"))
    expected_lines = output.get("lines")
    if (
        not isinstance(expected_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 1
        or isinstance(expected_lines, bool)
        or not isinstance(expected_lines, int)
        or expected_lines < 1
    ):
        raise LaunchError(f"dataset manifest has an incomplete output lock: {manifest_path}")
    actual = file_record(dataset_path)
    with dataset_path.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    if (
        actual["sha256"] != expected_hash
        or actual["size_bytes"] != expected_bytes
        or line_count != expected_lines
    ):
        raise LaunchError(f"dataset differs from authoritative manifest: {dataset_path}")
    return {
        "manifest": file_record(manifest_path),
        "artifact": {**actual, "lines": line_count},
    }


def resolve_registered_dataset(
    registry: Mapping[str, Any], alias: str, label: str
) -> tuple[Path, dict[str, Any]]:
    """Resolve one registry alias and bind it to its sole immutable manifest."""

    if not isinstance(alias, str) or not alias or alias not in registry:
        raise LaunchError(f"{label} alias is not registered: {alias}")
    entry = registry[alias]
    if not isinstance(entry, Mapping):
        raise LaunchError(f"{label} registry entry is not an object: {alias}")
    dataset_path = resolve_within(
        entry.get("file_name", ""), EXPERIMENT_ROOT / "data", label
    )
    return dataset_path, bind_dataset_manifest(dataset_path)


def verify_safetensors_layout(path: Path) -> dict[str, int]:
    size = path.stat().st_size
    if size < 10:
        raise LaunchError("model.safetensors is truncated")
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if not 2 <= header_size <= size - 8:
            raise LaunchError("invalid safetensors header length")
        try:
            header = json.loads(handle.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LaunchError("invalid safetensors header JSON") from error
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        raise LaunchError("safetensors contains no tensors")
    ranges = []
    for tensor in tensors:
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
        ):
            raise LaunchError("invalid safetensors tensor offsets")
        ranges.append((offsets[0], offsets[1]))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor or end < start:
            raise LaunchError("safetensors payload has gaps or overlaps")
        cursor = end
    if cursor != size - 8 - header_size:
        raise LaunchError("safetensors payload length mismatch")
    return {"tensor_count": len(tensors), "payload_bytes": cursor}


def validate_inputs(
    config_path: Path, cuda_visible_devices: str
) -> tuple[dict[str, Any], Path]:
    if cuda_visible_devices not in ALLOWED_CUDA_VISIBLE_DEVICES:
        raise LaunchError(
            "CUDA_VISIBLE_DEVICES must be exactly 0, 1, or the reserved "
            "four-GPU set 0,1,2,3"
        )
    config = load_config(config_path)
    output_dir = resolve_within(config["output_dir"], OUTPUT_ROOT, "output_dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LaunchError(f"output directory is not empty: {output_dir}")

    dataset_dir = resolve_within(
        config["dataset_dir"], EXPERIMENT_ROOT, "dataset_dir"
    )
    registry_path = dataset_dir / "dataset_info.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchError(f"invalid private dataset registry: {error}") from error
    alias = config["dataset"]
    dataset_path, dataset_binding = resolve_registered_dataset(
        registry, alias, "dataset"
    )
    evaluation_path: Path | None = None
    evaluation_binding: dict[str, Any] | None = None
    eval_alias = config.get("eval_dataset")
    if eval_alias is not None:
        evaluation_path, evaluation_binding = resolve_registered_dataset(
            registry, eval_alias, "eval_dataset"
        )
        if evaluation_path == dataset_path:
            raise LaunchError("dataset and eval_dataset must be different artifacts")

    model_dir = Path(config["model_name_or_path"]).resolve()
    model_artifacts = {
        name: file_record(model_dir / name) for name in REQUIRED_MODEL_FILES
    }
    verify_safetensors_layout(model_dir / "model.safetensors")
    if not CLI_PATH.is_file() or not os.access(CLI_PATH, os.X_OK):
        raise LaunchError(f"LLaMA-Factory CLI is unavailable: {CLI_PATH}")
    runtime = {str(path.relative_to(REPO_ROOT)): file_record(path) for path in RUNTIME_PATHS}
    return (
        {
            "config": file_record(config_path),
            "config_values": config,
            "dataset_registry": file_record(registry_path),
            "launcher": file_record(LAUNCHER_PATH),
            "dataset": dataset_binding,
            "evaluation_dataset": evaluation_binding,
            "initializer": {
                "directory": str(model_dir),
                "artifacts": model_artifacts,
            },
            "runtime": {
                **runtime,
                "llamafactory_cli": file_record(CLI_PATH),
            },
        },
        output_dir,
    )


def validate_bear_parameters(
    weights: Sequence[float], tau: float, topk: int, mode: str = "current"
) -> dict[str, Any]:
    if len(weights) != 3:
        raise LaunchError("BEAR requires exactly three SID-step weights")
    normalized_weights = []
    for step, value in zip(("a", "b", "c"), weights):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LaunchError(f"BEAR weight {step} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0.0 or number > 16.0:
            raise LaunchError(f"BEAR weight {step} must be finite and within [0, 16]")
        normalized_weights.append(number)
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise LaunchError("BEAR tau must be numeric")
    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 < tau <= 16.0:
        raise LaunchError("BEAR tau must be finite and within (0, 16]")
    if isinstance(topk, bool) or not isinstance(topk, int) or not 1 <= topk <= 8192:
        raise LaunchError("BEAR top-k must be an integer within [1, 8192]")
    if mode not in {"current", "rescue_v1"}:
        raise LaunchError("BEAR mode must be current or rescue_v1")
    enabled = any(weight > 0.0 for weight in normalized_weights)
    if mode != "current" and not enabled:
        raise LaunchError("rescue BEAR mode requires at least one positive weight")
    formula = (
        "1 + weight_step * sigmoid((target_logit - kth_logit) / tau)"
        if mode == "current"
        else "1 + weight_step * sigmoid((kth_logit - target_logit) / tau)"
    )
    return {
        "enabled": enabled,
        "mode": mode,
        "weights": dict(zip(("a", "b", "c"), normalized_weights)),
        "tau": tau,
        "topk": topk,
        "target_token_ranges": {
            "a": [151669, 159861],
            "b": [159861, 168053],
            "c": [168053, 176245],
        },
        "weight_formula": formula,
    }


def training_environment(
    cuda_visible_devices: str,
    inherited_path: str | None = None,
    *,
    bear: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    environment = {
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "PYTHONUNBUFFERED": "1",
    }
    # The locked LLaMA-Factory parser rejects even single-GPU training outside
    # distributed mode.  Force its CLI to re-exec through the project venv's
    # torchrun for every allowed device shape, not only the four-GPU path.
    environment["FORCE_TORCHRUN"] = "1"
    environment["NPROC_PER_NODE"] = str(len(cuda_visible_devices.split(",")))
    inherited_path = os.environ.get("PATH", "") if inherited_path is None else inherited_path
    environment["PATH"] = os.pathsep.join(
        part for part in (str(CLI_PATH.parent), inherited_path) if part
    )
    if bear is not None and bear.get("enabled") is True:
        weights = bear["weights"]
        environment.update(
            {
                "LLM_REC_BEAR_WEIGHT_A": repr(float(weights["a"])),
                "LLM_REC_BEAR_WEIGHT_B": repr(float(weights["b"])),
                "LLM_REC_BEAR_WEIGHT_C": repr(float(weights["c"])),
                "LLM_REC_BEAR_TAU": repr(float(bear["tau"])),
                "LLM_REC_BEAR_TOPK": str(int(bear["topk"])),
                "LLM_REC_BEAR_MODE": str(bear["mode"]),
            }
        )
    return environment


def training_command(config_path: Path, bear: Mapping[str, Any]) -> list[str]:
    if bear.get("enabled") is True and bear.get("mode") == "rescue_v1":
        return [str(PYTHON_PATH), str(RESCUE_BEAR_CLI), "train", str(config_path)]
    return [str(CLI_PATH), "train", str(config_path)]


def validate_output(output_dir: Path) -> dict[str, Any]:
    artifacts = {name: file_record(output_dir / name) for name in REQUIRED_MODEL_FILES}
    layout = verify_safetensors_layout(output_dir / "model.safetensors")
    try:
        trainer_state = json.loads(
            (output_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LaunchError(f"invalid final trainer_state.json: {error}") from error
    global_step = trainer_state.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 1:
        raise LaunchError("final trainer_state has no positive global_step")
    return {
        "directory": str(output_dir),
        "artifacts": artifacts,
        "safetensors": layout,
        "trainer_state": file_record(output_dir / "trainer_state.json"),
        "global_step": global_step,
        "epoch": trainer_state.get("epoch"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bear-weight-a", type=float, default=0.0)
    parser.add_argument("--bear-weight-b", type=float, default=0.0)
    parser.add_argument("--bear-weight-c", type=float, default=0.0)
    parser.add_argument("--bear-tau", type=float, default=1.0)
    parser.add_argument("--bear-topk", type=int, default=64)
    parser.add_argument(
        "--bear-mode", choices=("current", "rescue_v1"), default="current"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if RUN_ID_RE.fullmatch(args.run_id) is None:
        print("FAIL: invalid run ID", file=sys.stderr)
        return 1
    run_dir = RUN_ROOT / args.run_id
    state_path = run_dir / "state.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        print(f"FAIL: run directory is not empty: {run_dir}", file=sys.stderr)
        return 1
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        bear = validate_bear_parameters(
            (args.bear_weight_a, args.bear_weight_b, args.bear_weight_c),
            args.bear_tau,
            args.bear_topk,
            args.bear_mode,
        )
        inputs, output_dir = validate_inputs(
            args.config.resolve(), args.cuda_visible_devices
        )
        command = training_command(args.config.resolve(), bear)
        launch = {
            "schema": "llm-rec.takeover.training-launch",
            "schema_version": 1,
            "run_id": args.run_id,
            "argv": command,
            "cwd": str(REPO_ROOT),
            "environment": training_environment(
                args.cuda_visible_devices, bear=bear
            ),
            "optimization": {"bear": bear},
            "inputs": inputs,
            "output_directory": str(output_dir),
        }
        launch_bytes = canonical_json_bytes(launch)
        atomic_write(run_dir / "launch_manifest.json", launch_bytes)
        atomic_write(
            run_dir / "launch_manifest.json.sha256",
            (hashlib.sha256(launch_bytes).hexdigest() + "\n").encode("ascii"),
        )
        atomic_write(
            state_path,
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "status": "RUNNING",
                    "launch_manifest_sha256": hashlib.sha256(launch_bytes).hexdigest(),
                }
            ),
        )
        environment = os.environ.copy()
        environment.update(launch["environment"])
        console_path = run_dir / "console.log"
        with console_path.open("xb") as console:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if return_code != 0:
            raise LaunchError(f"trainer exited with status {return_code}")
        verified_input_locks = verify_file_locks(inputs)
        if verified_input_locks < 1:
            raise LaunchError("launch manifest contains no verifiable input locks")
        completion = {
            "schema": "llm-rec.takeover.training-completion",
            "schema_version": 1,
            "run_id": args.run_id,
            "launch_manifest": file_record(run_dir / "launch_manifest.json"),
            "console": file_record(console_path),
            "verified_input_locks": verified_input_locks,
            "output": validate_output(output_dir),
            "verdict": "PASS",
        }
        completion_bytes = canonical_json_bytes(completion)
        atomic_write(run_dir / "completion_manifest.json", completion_bytes)
        completion_hash = hashlib.sha256(completion_bytes).hexdigest()
        atomic_write(
            run_dir / "completion_manifest.json.sha256",
            (completion_hash + "\n").encode("ascii"),
        )
        atomic_write(
            state_path,
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "completion_manifest_sha256": completion_hash,
                }
            ),
        )
        print(f"PASS: {args.run_id} -> {output_dir}")
        return 0
    except BaseException as error:
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
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
