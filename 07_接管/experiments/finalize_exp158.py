#!/usr/bin/env python3
"""Capture and BF16-merge EXP-157's receipt-bound one-epoch checkpoint as EXP-158."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import audit_recovery_run
import run_training


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
RUN = ROOT / "runs/exp158_exp157_one_epoch_duration_v1"
DESIGN = ROOT / "preregistrations/exp158_exp157_one_epoch_duration_design.json"
PARENT_FULL = ROOT / "preregistrations/exp157_answer_only_r0_continuation_full_launch.json"
PARENT_AUDIT = ROOT / "runs/exp157_answer_only_r0_continuation_v100_v1/audit_manifest.json"
PARENT_UPLOAD = REPO / "07_接管/platform/upload_receipts/takeover_exp157_answer_only_r0_continuation_20260725.json"
PARENT_OUTPUT = ROOT / "outputs/onereason_0.8b_exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16"
ADAPTER = ROOT / "outputs/onereason_0.8b_exp158_exp157_one_epoch_r0_continuation"
MERGED = Path(str(ADAPTER) + "_merged_bf16")
CAPTURE = RUN / "capture_manifest.json"
COMPLETION = RUN / "completion_manifest.json"
AUDIT = RUN / "audit_manifest.json"
TARGET = REPO / "07_接管/autopilot/TARGET_1_1_ACHIEVED.json"
SNAPSHOT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "trainer_state.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
MERGED_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class Exp158Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Exp158Error(message)


def load(path: Path) -> Any:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_locked(path: Path, value: dict[str, Any]) -> None:
    run_training.atomic_write(path, run_training.canonical_json_bytes(value))
    digest = run_training.sha256_file(path)
    run_training.atomic_write(path.with_suffix(path.suffix + ".sha256"), f"{digest}  {path.name}\n".encode("ascii"))


def verify_design() -> dict[str, Any]:
    value = load(DESIGN)
    require(value.get("experiment") == "EXP-158", "design identity drift")
    require(value.get("status") == "LOCKED_BEFORE_EXP157_TRAINING", "design is not locked")
    for section in ("parent", "immutable_inputs", "implementation"):
        records = value.get(section, {})
        for key, expected in records.items():
            if not key.endswith("_sha256"):
                continue
            path_key = key.removesuffix("_sha256") + "_path"
            path = REPO / records.get(path_key, "")
            require(path.is_file() and not path.is_symlink(), f"missing design lock {path_key}")
            require(run_training.sha256_file(path) == expected, f"design lock drift: {path_key}")
    require(not TARGET.exists(), "target marker already exists")
    return value


def validate_adapter(directory: Path, *, require_optimizer: bool) -> dict[str, Any]:
    files = {name: run_training.file_record(directory / name) for name in SNAPSHOT_FILES}
    if require_optimizer:
        require((directory / "optimizer.pt").is_file(), "checkpoint optimizer save is incomplete")
    state = load(directory / "trainer_state.json")
    step = state.get("global_step")
    max_steps = state.get("max_steps")
    epoch = float(state.get("epoch", 0.0))
    require(isinstance(step, int) and step > 0, "invalid checkpoint global_step")
    require(isinstance(max_steps, int) and max_steps > step, "invalid parent max_steps")
    require(math.isclose(epoch, 1.0, rel_tol=0.0, abs_tol=0.01), "checkpoint is not the one-epoch arm")
    require(abs(2 * step - max_steps) <= 2, "one-epoch step is not half of the two-epoch schedule")
    config = load(directory / "adapter_config.json")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM", "adapter identity mismatch")
    require(config.get("r") == 64 and config.get("lora_alpha") == 64, "adapter geometry mismatch")
    require(config.get("lora_dropout") == 0.05 and config.get("use_rslora") is False, "adapter method mismatch")
    require(config.get("use_dora") is False, "unexpected DoRA")
    finite = audit_recovery_run.scan_safetensors(directory / "adapter_model.safetensors")
    require(finite.get("all_finite") is True, "adapter contains non-finite values")
    return {
        "directory": str(directory.resolve()),
        "files": files,
        "trainer_state": {"global_step": step, "max_steps": max_steps, "epoch": epoch},
        "adapter_config": config,
        "finite_scan": finite,
    }


def select_source() -> Path:
    matches = []
    for path in sorted(PARENT_OUTPUT.glob("checkpoint-*")):
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if math.isclose(float(state.get("epoch", 0.0)), 1.0, rel_tol=0.0, abs_tol=0.01):
                matches.append(path)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    require(len(matches) == 1, f"expected exactly one one-epoch checkpoint, found {len(matches)}")
    return matches[0]


def capture() -> None:
    verify_design()
    require(not ADAPTER.exists() and not CAPTURE.exists(), "EXP-158 was already captured")
    source_path = select_source()
    source = validate_adapter(source_path, require_optimizer=True)
    temporary = ADAPTER.with_name(f".{ADAPTER.name}.capture-tmp")
    require(not temporary.exists(), "stale capture temporary directory exists")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for name in SNAPSHOT_FILES:
            shutil.copy2(source_path / name, temporary / name)
        for name in ("README.md", "training_args.bin"):
            if (source_path / name).is_file():
                shutil.copy2(source_path / name, temporary / name)
        snapshot = validate_adapter(temporary, require_optimizer=False)
        for name in SNAPSHOT_FILES:
            require(snapshot["files"][name]["sha256"] == source["files"][name]["sha256"], f"capture hash mismatch: {name}")
        temporary.rename(ADAPTER)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    manifest = {
        "schema": "llm-rec.takeover.exp158-checkpoint-capture",
        "schema_version": 1,
        "experiment": "EXP-158",
        "source_experiment": "EXP-157",
        "verdict": "PASS",
        "design": run_training.file_record(DESIGN),
        "source": source,
        "snapshot": validate_adapter(ADAPTER, require_optimizer=False),
        "claim": "The only controlled difference from EXP-157 is stopping at the retained one-epoch checkpoint.",
    }
    RUN.mkdir(parents=True, exist_ok=True)
    write_locked(CAPTURE, manifest)
    print(json.dumps({"status": "CAPTURED", "source": str(source_path), "adapter": str(ADAPTER)}))


def verify_parent() -> tuple[dict[str, Any], Path]:
    parent = load(PARENT_AUDIT)
    require(parent.get("experiment") == "EXP-157" and parent.get("verdict") == "PASS", "parent audit is not PASS")
    require(parent.get("lineage_completeness") == "FULL", "parent lineage is incomplete")
    upload = load(PARENT_UPLOAD)
    require(isinstance(upload.get("model_id"), str) and upload["model_id"].startswith("md-"), "parent upload receipt is invalid")
    launch = load(PARENT_FULL)
    require(launch.get("experiment") == "EXP-157" and launch.get("status") == "LOCKED_BEFORE_TRAINING", "parent launch drift")
    anchor = Path(launch.get("anchor", {}).get("model_dir", ""))
    model = anchor / "model.safetensors"
    require(model.is_file(), "receipt-bound parent anchor is missing")
    record = launch.get("anchor", {}).get("model", {})
    require(run_training.sha256_file(model) == record.get("sha256"), "parent anchor model SHA drift")
    require(model.stat().st_size == record.get("size_bytes"), "parent anchor model size drift")
    return launch, anchor


def merge_adapter(anchor: Path) -> None:
    require(not MERGED.exists(), "EXP-158 merged output already exists")
    MERGED.mkdir(parents=True, exist_ok=False)
    base = AutoModelForCausalLM.from_pretrained(anchor, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": "cpu"})
    peft_model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED)
    for name in ("generation_config.json", "chat_template.jinja", "special_tokens_map.json"):
        source = ADAPTER / name if (ADAPTER / name).is_file() else anchor / name
        if source.is_file() and not (MERGED / name).is_file():
            shutil.copy2(source, MERGED / name)


def finalize() -> None:
    verify_design()
    require(CAPTURE.is_file(), "capture manifest is missing")
    require(not COMPLETION.exists() and not AUDIT.exists(), "EXP-158 was already finalized")
    capture_value = load(CAPTURE)
    require(capture_value.get("verdict") == "PASS", "capture is not PASS")
    adapter = validate_adapter(ADAPTER, require_optimizer=False)
    for name, record in capture_value["snapshot"]["files"].items():
        require(adapter["files"][name]["sha256"] == record["sha256"], f"captured adapter drift: {name}")
    launch, anchor = verify_parent()
    merge_adapter(anchor)
    merged_files = {name: run_training.file_record(MERGED / name) for name in MERGED_FILES}
    finite = audit_recovery_run.scan_safetensors(MERGED / "model.safetensors")
    require(finite.get("all_finite") is True, "merged model contains non-finite values")
    require(load(MERGED / "config.json").get("model_type") == "qwen3", "merged model type mismatch")
    completion = {
        "schema": "llm-rec.takeover.exp158-completion",
        "schema_version": 1,
        "experiment": "EXP-158",
        "run_id": "exp158_exp157_one_epoch_duration_v1",
        "verdict": "PASS",
        "verdict_scope": "one-epoch checkpoint capture, parent receipt-bound lineage, and CPU BF16 merge; formal efficacy remains unknown",
        "design": run_training.file_record(DESIGN),
        "capture": run_training.file_record(CAPTURE),
        "parent_audit": run_training.file_record(PARENT_AUDIT),
        "parent_upload": run_training.file_record(PARENT_UPLOAD),
        "anchor": launch["anchor"],
        "adapter": adapter,
        "merged": {"directory": str(MERGED), "files": merged_files, "finite_scan": finite, "dtype": "bfloat16"},
    }
    write_locked(COMPLETION, completion)
    audit = {
        "schema": "llm-rec.takeover.training-run-audit",
        "schema_version": 1,
        "run_id": "exp158_exp157_one_epoch_duration_v1",
        "experiment": "EXP-158",
        "verdict": "PASS",
        "lineage_completeness": "FULL",
        "efficacy_claim": "unknown until formal platform evaluation",
        "training": {"parent_experiment": "EXP-157", "design": run_training.file_record(DESIGN), "capture": run_training.file_record(CAPTURE), "adapter": adapter, "completion_manifest": run_training.file_record(COMPLETION)},
        "finite_scan": finite,
        "output": {"directory": str(MERGED), "artifacts": merged_files},
    }
    write_locked(AUDIT, audit)
    print(json.dumps({"status": "PASS", "merged": str(MERGED), "audit": str(AUDIT)}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true")
    action.add_argument("--capture", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_design()
        print(json.dumps({"verified": True, "experiment": "EXP-158"}))
    elif args.capture:
        capture()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
