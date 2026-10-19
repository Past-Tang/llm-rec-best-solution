#!/usr/bin/env python3
"""Record, audit, and merge the EXP-135 V100 v58 reproduction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import audit_recovery_run
import run_training


REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_ROOT = REPO_ROOT / "07_接管/experiments"
RUN_ID = "exp135_public091_v58_v100_16k_ga8_v1"
EXPERIMENT = "EXP-135"
TRAINING_STARTED_AT = "2026-07-23T23:25:31+08:00"
RUN_DIR = EXP_ROOT / "runs" / RUN_ID
CONFIG = EXP_ROOT / "configs/exp135_public091_v58_loraplus_r64_a64_xformers_16k_ga8_fp16.yaml"
PREREG = EXP_ROOT / "preregistrations/exp135_public091_v58_v100_16k_ga8_v1.json"
DATASET = EXP_ROOT / "data/exp132_public091_exact_v1.jsonl"
DATASET_MANIFEST = EXP_ROOT / "manifests/exp132_public091_exact_v1.manifest.json"
REGISTRY = EXP_ROOT / "dataset_registry_exp132_public091/dataset_info.json"
EXTERNAL_RECIPE = REPO_ROOT / "07_接管/analysis/external_evidence/lldois_v58_recipe_d12f1e5c.json"
BASE = REPO_ROOT / "03_模型/models/OneReason-0.8B-pretrain-competition"
ADAPTER = EXP_ROOT / "outputs/onereason_0.8b_exp135_public091_v58_loraplus_r64_a64_xformers_16k_ga8_fp16"
MERGED = EXP_ROOT / "outputs/onereason_0.8b_exp135_public091_v58_loraplus_r64_a64_xformers_16k_ga8_fp16_merged_bf16"
TRAINING_ROOT = REPO_ROOT / "02_数据/data/Explorer_LLM_Rec_Competition/demo/LLaMA-Factory"
CLI = Path("/workspace/tmp/venvs/llmrec-train/bin/llamafactory-cli")
LAUNCH_MANIFEST = RUN_DIR / "launch_manifest.json"
COMPLETION_MANIFEST = RUN_DIR / "completion_manifest.json"
AUDIT_MANIFEST = RUN_DIR / "audit_manifest.json"
EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
EXPECTED_RANK = 64
EXPECTED_ALPHA = 64
EXPECTED_DROPOUT = 0.05
EXPECTED_RSLORA = False
EXPECTED_STEPS = 652
EXPECTED_EPOCHS = 2.0
RUNTIME_FILES = (
    TRAINING_ROOT / "pyproject.toml",
    TRAINING_ROOT / "src/llamafactory/extras/constants.py",
    TRAINING_ROOT / "src/llamafactory/data/collator.py",
    TRAINING_ROOT / "src/llamafactory/model/loader.py",
    TRAINING_ROOT / "src/llamafactory/model/model_utils/attention.py",
    TRAINING_ROOT / "src/llamafactory/model/adapter.py",
    TRAINING_ROOT / "src/llamafactory/train/sft/workflow.py",
    TRAINING_ROOT / "src/llamafactory/train/trainer_utils.py",
)
BASE_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
MERGED_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


class FinalizeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizeError(message)


def load_json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def write_locked_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = run_training.canonical_json_bytes(value)
    run_training.atomic_write(path, payload)
    digest = run_training.sha256_file(path)
    run_training.atomic_write(
        path.with_suffix(path.suffix + ".sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return run_training.file_record(path)


def capture_inputs() -> dict[str, Any]:
    return {
        "config": run_training.file_record(CONFIG),
        "preregistration": run_training.file_record(PREREG),
        "dataset": run_training.file_record(DATASET),
        "dataset_manifest": run_training.file_record(DATASET_MANIFEST),
        "dataset_registry": run_training.file_record(REGISTRY),
        "external_recipe": run_training.file_record(EXTERNAL_RECIPE),
        "base": {name: run_training.file_record(BASE / name) for name in BASE_FILES},
        "runtime": {
            str(path.relative_to(REPO_ROOT)): run_training.file_record(path) for path in RUNTIME_FILES
        },
    }


def record_launch() -> None:
    require(not LAUNCH_MANIFEST.exists(), "launch manifest already exists")
    prereg = load_json(PREREG)
    require(prereg.get("status") == "LOCKED_BEFORE_TRAINING", "preregistration is not locked")
    manifest = {
        "schema": "llm-rec.takeover.exp135-launch",
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "training_started_at": TRAINING_STARTED_AT,
        "command": [str(CLI), "train", str(CONFIG)],
        "cuda_visible_devices": "0",
        "inputs": capture_inputs(),
        "adapter_output": str(ADAPTER),
        "merged_output": str(MERGED),
        "evidence_note": "Recorded during active training; all input locks are verified again at completion.",
    }
    write_locked_json(LAUNCH_MANIFEST, manifest)
    print(json.dumps({"status": "RECORDED", "manifest": str(LAUNCH_MANIFEST)}, indent=2))


def validate_adapter() -> dict[str, Any]:
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "all_results.json",
        "train_results.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    files = {name: run_training.file_record(ADAPTER / name) for name in required}
    state = load_json(ADAPTER / "trainer_state.json")
    require(
        state.get("global_step") == EXPECTED_STEPS,
        f"final global_step is not {EXPECTED_STEPS}",
    )
    require(
        float(state.get("epoch", 0.0)) >= EXPECTED_EPOCHS,
        f"training did not expose {EXPECTED_EPOCHS} epochs",
    )
    config = load_json(ADAPTER / "adapter_config.json")
    require(config.get("peft_type") == "LORA", "adapter is not LoRA")
    require(config.get("task_type") == "CAUSAL_LM", "adapter task mismatch")
    require(config.get("r") == EXPECTED_RANK, "adapter rank mismatch")
    require(config.get("lora_alpha") == EXPECTED_ALPHA, "adapter alpha mismatch")
    require(config.get("lora_dropout") == EXPECTED_DROPOUT, "adapter dropout mismatch")
    require(config.get("use_rslora") is EXPECTED_RSLORA, "rsLoRA flag mismatch")
    require(config.get("use_dora") is False, "unexpected DoRA")
    require(set(config.get("target_modules", [])) == EXPECTED_TARGETS, "target modules mismatch")
    finite = audit_recovery_run.scan_safetensors(ADAPTER / "adapter_model.safetensors")
    require(finite.get("all_finite") is True, "adapter contains non-finite values")
    return {
        "directory": str(ADAPTER),
        "files": files,
        "trainer_state": {"global_step": state["global_step"], "epoch": state["epoch"]},
        "adapter_config": config,
        "finite_scan": finite,
        "recovery_checkpoints": sorted(path.name for path in ADAPTER.glob("checkpoint-*")),
    }


def merge_adapter() -> None:
    require(not MERGED.exists(), "merged directory already exists")
    MERGED.mkdir(parents=True, exist_ok=False)
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    peft_model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    tokenizer.save_pretrained(MERGED)
    for name in ("generation_config.json", "chat_template.jinja", "special_tokens_map.json"):
        source = ADAPTER / name
        if source.is_file() and not (MERGED / name).is_file():
            shutil.copy2(source, MERGED / name)
    if not (MERGED / "generation_config.json").is_file():
        shutil.copy2(BASE / "generation_config.json", MERGED / "generation_config.json")
    require((MERGED / "chat_template.jinja").is_file(), "merged chat template is missing")


def validate_merged() -> dict[str, Any]:
    files = {name: run_training.file_record(MERGED / name) for name in MERGED_FILES}
    layout = run_training.verify_safetensors_layout(MERGED / "model.safetensors")
    finite = audit_recovery_run.scan_safetensors(MERGED / "model.safetensors")
    require(finite.get("all_finite") is True, "merged model contains non-finite values")
    config = load_json(MERGED / "config.json")
    require(config.get("model_type") == "qwen3", "merged model type mismatch")
    require(config.get("vocab_size") == 176253, "merged vocabulary mismatch")
    require(config.get("num_hidden_layers") == 28, "merged layer count mismatch")
    require(not (MERGED / "adapter_config.json").exists(), "merged root still identifies as adapter")
    return {
        "directory": str(MERGED),
        "files": files,
        "layout": layout,
        "finite_scan": finite,
        "dtype": "bfloat16",
    }


def finalize() -> None:
    require(LAUNCH_MANIFEST.is_file(), "launch manifest is missing")
    require(not COMPLETION_MANIFEST.exists(), "completion manifest already exists")
    require(not AUDIT_MANIFEST.exists(), "audit manifest already exists")
    launch = load_json(LAUNCH_MANIFEST)
    verified_before = run_training.verify_file_locks(launch["inputs"], "launch.inputs")
    adapter = validate_adapter()
    merge_adapter()
    merged = validate_merged()
    verified_after = run_training.verify_file_locks(launch["inputs"], "completion.inputs")
    require(verified_after == verified_before, "input lock count changed")
    console = run_training.file_record(RUN_DIR / "console.log")
    completion = {
        "schema": "llm-rec.takeover.exp135-completion",
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "run_id": RUN_ID,
        "verdict": "PASS",
        "verdict_scope": "training, adapter integrity, and BF16 merge; formal efficacy remains unknown",
        "verified_input_locks": verified_after,
        "launch_manifest": run_training.file_record(LAUNCH_MANIFEST),
        "console_log": console,
        "adapter": adapter,
        "merged": merged,
    }
    write_locked_json(COMPLETION_MANIFEST, completion)
    audit = {
        "schema": "llm-rec.takeover.training-run-audit",
        "schema_version": 1,
        "run_id": RUN_ID,
        "experiment": EXPERIMENT,
        "verdict": "PASS",
        "lineage_completeness": "FULL",
        "efficacy_claim": "unknown until formal platform evaluation",
        "training": {
            "config": run_training.file_record(CONFIG),
            "preregistration": run_training.file_record(PREREG),
            "dataset": run_training.file_record(DATASET),
            "base_model": run_training.file_record(BASE / "model.safetensors"),
            "adapter": adapter,
            "completion_manifest": run_training.file_record(COMPLETION_MANIFEST),
        },
        "finite_scan": merged["finite_scan"],
        "output": {
            "directory": str(MERGED),
            "artifacts": merged["files"],
        },
    }
    write_locked_json(AUDIT_MANIFEST, audit)
    print(json.dumps({"status": "PASS", "merged": str(MERGED)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--record-launch", action="store_true")
    action.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.record_launch:
        record_launch()
    else:
        finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
