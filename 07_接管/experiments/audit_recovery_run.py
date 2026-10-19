#!/usr/bin/env python3
"""Validate recovery checkpoints and emit a deterministic training lineage manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "07_接管/experiments"
RUN_DIR = (
    EXPERIMENT_ROOT
    / "outputs/onereason_0.8b_takeover_replay2_protocol_v3"
)
CONFIG_PATH = EXPERIMENT_ROOT / "configs/sft_takeover_replay2_protocol.yaml"
CONFIGURATION_MANIFEST = EXPERIMENT_ROOT / "manifests/configuration.manifest.json"
DATASET_MANIFEST = EXPERIMENT_ROOT / "manifests/replay2_protocol.manifest.json"
DATASET_REGISTRY = EXPERIMENT_ROOT / "dataset_registry/dataset_info.json"
CANDIDATE_REGISTRY = EXPERIMENT_ROOT / "manifests/candidate_registry.json"
OUTPUT_MANIFEST = EXPERIMENT_ROOT / "manifests/recovery_run_lineage.json"
INIT_MODEL_DIR = (
    REPO_ROOT / "04_作战/rft_v2/output/onereason_0.8b_sft_v9g1"
)
RUNTIME_TRAINER = (
    REPO_ROOT
    / "02_数据/data/Explorer_LLM_Rec_Competition/demo/LLaMA-Factory/"
    "src/llamafactory/train/sft/trainer.py"
)
CHECKPOINT_STEPS = (20, 30, 33)
LOCK_PATHS = {
    step: REPO_ROOT
    / f"07_接管/platform/model_locks/takeover_recovery_step{step}.json"
    for step in CHECKPOINT_STEPS
}
REQUIRED_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def workspace_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"required file is missing: {path}")
    return {
        "path": workspace_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load JSON {path}: {exc}") from exc


def require_finite_json(value: Any, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            require_finite_json(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            require_finite_json(nested, f"{location}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite JSON value at {location}")


def scan_safetensors(path: Path) -> dict[str, Any]:
    """Load each tensor on CPU and fail closed on any NaN or infinity."""
    require(path.is_file(), f"safetensors file is missing: {path}")
    dtype_counts: Counter[str] = Counter()
    element_count = 0
    nonfinite_tensors: list[str] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor_names = sorted(handle.keys())
        require(tensor_names, f"safetensors has no tensors: {path}")
        for name in tensor_names:
            tensor_slice = handle.get_slice(name)
            shape = tuple(int(value) for value in tensor_slice.get_shape())
            dtype_counts[str(tensor_slice.get_dtype())] += 1
            element_count += math.prod(shape)
            tensor = handle.get_tensor(name)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                nonfinite_tensors.append(name)
            del tensor
    return {
        "tensor_count": len(tensor_names),
        "element_count": element_count,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "all_finite": not nonfinite_tensors,
        "nonfinite_tensors": nonfinite_tensors,
    }


def verify_record(actual_path: Path, expected: dict, label: str) -> dict:
    actual = file_record(actual_path)
    require(actual["bytes"] == expected.get("bytes"), f"{label} byte size drift")
    require(actual["sha256"] == expected.get("sha256"), f"{label} hash drift")
    return actual


def verify_effective_configuration() -> dict[str, Any]:
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AuditError(f"cannot load recovery YAML: {exc}") from exc
    require(isinstance(config, dict), "recovery YAML must be a mapping")

    expected_fields = {
        "model_name_or_path": str(INIT_MODEL_DIR),
        "dataset": "takeover_replay2_protocol",
        "dataset_dir": str(EXPERIMENT_ROOT / "dataset_registry"),
        "output_dir": str(RUN_DIR),
        "val_size": 0.02,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "steps",
        "save_steps": 10,
        "overwrite_output_dir": False,
        "prediction_loss_only": True,
        "learning_rate": 2.0e-6,
        "num_train_epochs": 1,
        "seed": 20260701,
    }
    for key, expected in expected_fields.items():
        require(config.get(key) == expected, f"unexpected recovery config {key}")

    configuration_manifest = load_json(CONFIGURATION_MANIFEST)
    config_entry = configuration_manifest["runnable_yaml_configs"][CONFIG_PATH.name]
    config_record = file_record(CONFIG_PATH)
    require(
        config_record["sha256"] == config_entry["sha256"],
        "recovery YAML hash does not match configuration manifest",
    )
    require(
        config_entry["dataset_sha256"]
        == configuration_manifest["private_dataset_registry"]["entries"]
        ["takeover_replay2_protocol"]["sha256"],
        "configuration manifest disagrees about dataset hash",
    )
    runtime_entry = configuration_manifest["runtime_sources"][
        "llamafactory_sft_trainer"
    ]
    runtime_record = file_record(RUNTIME_TRAINER)
    require(
        runtime_record["sha256"] == runtime_entry["sha256"],
        "runtime trainer source hash drift",
    )
    return {
        "yaml": config_record,
        "configuration_manifest": file_record(CONFIGURATION_MANIFEST),
        "effective_fields": expected_fields,
        "runtime_trainer": runtime_record,
    }


def verify_dataset_lineage() -> dict[str, Any]:
    dataset_manifest = load_json(DATASET_MANIFEST)
    dataset_registry = load_json(DATASET_REGISTRY)
    output = dataset_manifest["output"]
    dataset_path = REPO_ROOT / output["path"]
    dataset_record = verify_record(dataset_path, output, "recovery dataset")
    registry_entry = dataset_registry["takeover_replay2_protocol"]
    require(
        Path(registry_entry["file_name"]).resolve() == dataset_path.resolve(),
        "dataset registry path does not match dataset manifest",
    )
    require(
        dataset_manifest.get("single_variable_claim") is False,
        "sequential recovery must not claim to be single-variable",
    )
    return {
        "dataset": dataset_record,
        "dataset_manifest": file_record(DATASET_MANIFEST),
        "dataset_registry": file_record(DATASET_REGISTRY),
        "hypothesis_class": dataset_manifest["hypothesis_class"],
        "single_variable_claim": dataset_manifest["single_variable_claim"],
        "root_sources": dataset_manifest["root_sources"],
    }


def verify_initial_model() -> dict[str, Any]:
    registry = load_json(CANDIDATE_REGISTRY)
    entry = registry["training_initializers"]["v9g1_final"]
    expected_artifacts = entry["artifacts"]["artifacts"]
    artifacts = {
        filename: verify_record(
            INIT_MODEL_DIR / filename,
            expected_artifacts[filename],
            f"initial model {filename}",
        )
        for filename in REQUIRED_MODEL_FILES
    }
    require(
        entry["formal_r0_hits"] == {"hits": 7, "denominator": 574},
        "v9g1 formal R0 evidence drift",
    )
    return {
        "candidate_registry": file_record(CANDIDATE_REGISTRY),
        "initializer_id": "v9g1_final",
        "global_step": entry["global_step"],
        "formal_r0_hits": entry["formal_r0_hits"],
        "artifacts": artifacts,
    }


def verify_checkpoint(step: int) -> dict[str, Any]:
    checkpoint_dir = RUN_DIR / f"checkpoint-{step}"
    lock_path = LOCK_PATHS[step]
    lock = load_json(lock_path)
    require(
        lock.get("schema") == "llm-rec.takeover.model-upload-lock",
        f"checkpoint {step} has wrong lock schema",
    )
    require(
        Path(lock.get("model_dir", "")).resolve() == checkpoint_dir.resolve(),
        f"checkpoint {step} lock points to another directory",
    )
    require(
        lock.get("operations_performed")
        == {"formal_eval": False, "network": False, "upload": False},
        f"checkpoint {step} operation flags changed",
    )
    artifacts = {
        filename: verify_record(
            checkpoint_dir / filename,
            lock["artifacts"][filename],
            f"checkpoint {step} {filename}",
        )
        for filename in REQUIRED_MODEL_FILES
    }
    finite_scan = scan_safetensors(checkpoint_dir / "model.safetensors")
    require(finite_scan["all_finite"], f"checkpoint {step} contains NaN/Inf")
    require(
        finite_scan["tensor_count"]
        == lock["integrity"]["safetensors"]["tensor_count"],
        f"checkpoint {step} tensor count disagrees with upload lock",
    )

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = load_json(trainer_state_path)
    require_finite_json(trainer_state, f"checkpoint-{step}.trainer_state")
    require(trainer_state.get("global_step") == step, f"checkpoint {step} state drift")
    require(trainer_state.get("max_steps") == 33, f"checkpoint {step} max_steps drift")
    matching_eval = [
        item
        for item in trainer_state.get("log_history", [])
        if item.get("step") == step and "eval_loss" in item
    ]
    require(len(matching_eval) == 1, f"checkpoint {step} eval record missing/ambiguous")
    rank_states = sorted(checkpoint_dir.glob("rng_state_*.pth"))
    require(len(rank_states) == 4, f"checkpoint {step} does not have four rank RNG states")
    return {
        "step": step,
        "epoch": trainer_state["epoch"],
        "eval_loss": matching_eval[0]["eval_loss"],
        "rank_rng_state_count": len(rank_states),
        "artifacts": artifacts,
        "trainer_state": file_record(trainer_state_path),
        "upload_lock": file_record(lock_path),
        "finite_scan": finite_scan,
    }


def verify_run_outputs(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    results = {}
    for filename in ("all_results.json", "train_results.json", "eval_results.json"):
        path = RUN_DIR / filename
        payload = load_json(path)
        require_finite_json(payload, filename)
        results[filename] = {"artifact": file_record(path), "values": payload}

    final_model = file_record(RUN_DIR / "model.safetensors")
    step33_model = checkpoints[-1]["artifacts"]["model.safetensors"]
    require(
        final_model["sha256"] == step33_model["sha256"]
        and final_model["bytes"] == step33_model["bytes"],
        "final model is not byte-identical to checkpoint-33",
    )
    final_state = load_json(RUN_DIR / "trainer_state.json")
    require_finite_json(final_state, "final trainer_state")
    require(final_state.get("global_step") == 33, "final trainer state is incomplete")
    return {
        "final_model": final_model,
        "final_model_identical_to_checkpoint_33": True,
        "trainer_state": file_record(RUN_DIR / "trainer_state.json"),
        "training_args": file_record(RUN_DIR / "training_args.bin"),
        "trainer_log": file_record(RUN_DIR / "trainer_log.jsonl"),
        "model_card": file_record(RUN_DIR / "README.md"),
        "results": results,
    }


def atomic_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp.{os.getpid()}")
    try:
        sidecar_temporary.write_text(
            f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
        )
        sidecar_temporary.replace(sidecar)
    finally:
        sidecar_temporary.unlink(missing_ok=True)


def build_manifest() -> dict[str, Any]:
    configuration = verify_effective_configuration()
    dataset = verify_dataset_lineage()
    initial_model = verify_initial_model()
    checkpoints = [verify_checkpoint(step) for step in CHECKPOINT_STEPS]
    run_outputs = verify_run_outputs(checkpoints)
    return {
        "schema": "llm-rec.takeover.training-run-lineage",
        "schema_version": 1,
        "run_id": "onereason_0.8b_takeover_replay2_protocol_v3",
        "run_directory": workspace_path(RUN_DIR),
        "hypothesis_class": "sequential-recovery",
        "configuration": configuration,
        "dataset_lineage": dataset,
        "initial_model": initial_model,
        "checkpoints": checkpoints,
        "run_outputs": run_outputs,
        "validation": {
            "all_checkpoint_weights_finite": True,
            "all_checkpoint_locks_match": True,
            "all_json_metrics_finite": True,
            "config_dataset_runtime_hashes_match": True,
            "final_model_identical_to_checkpoint_33": True,
            "four_rank_rng_states_present": True,
        },
        "lineage_completeness": "PARTIAL",
        "traceability_gaps": [
            {
                "field": "launch_process_argv_and_environment",
                "status": "not_captured_at_launch",
                "impact": (
                    "Effective trainer arguments and artifacts are preserved, but the exact "
                    "shell argv/environment cannot be reconstructed without guessing."
                ),
            },
            {
                "field": "complete_console_log",
                "status": "not_captured_at_launch",
                "impact": (
                    "Trainer JSON logs and states are present; pre-trainer launcher output is not."
                ),
            },
        ],
        "artifact_integrity_verdict": "PASS",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    manifest = build_manifest()
    atomic_write_manifest(OUTPUT_MANIFEST, manifest)
    print(f"ARTIFACT_INTEGRITY={manifest['artifact_integrity_verdict']}")
    print(f"LINEAGE_COMPLETENESS={manifest['lineage_completeness']}")
    print(f"OUTPUT={OUTPUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
