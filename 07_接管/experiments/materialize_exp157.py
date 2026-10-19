#!/usr/bin/env python3
"""Materialize receipt-bound EXP-157 smoke/full configs from the selected merged anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
GATE = REPO / "07_接管/autopilot/exp157_receipt_bound_gate.json"
DESIGN = ROOT / "preregistrations/exp157_answer_only_r0_compact_design.json"
TARGET = REPO / "07_接管/autopilot/TARGET_1_1_ACHIEVED.json"
SMOKE_CONFIG = ROOT / "configs/smoke_exp157_answer_only_r0_continuation_r64_a64_xformers_fp16.yaml"
FULL_CONFIG = ROOT / "configs/exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16.yaml"
LAUNCH = ROOT / "preregistrations/exp157_answer_only_r0_continuation_launch.json"
FULL_OUTPUT = ROOT / "outputs/onereason_0.8b_exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16"
SMOKE_OUTPUT = Path("/tmp/llmrec-exp157-answer-only-r0-smoke")
MIN_FREE = 6 * 1024**3


ANCHORS = {
    "EXP-136": {
        "audit": ROOT / "runs/exp136_public091_v66_v100_16k_ga8_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp136_v66_public091_16k_ga8_20260724.json",
    },
    "EXP-152": {
        "audit": ROOT / "runs/exp152_public091_answer_only_liger_v100_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp152_public091_answer_only_liger_20260725.json",
    },
    "EXP-153": {
        "audit": ROOT / "runs/exp153_answer_only_v66_liger_v100_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp153_answer_only_v66_liger_20260725.json",
    },
    "EXP-154": {
        "audit": ROOT / "runs/exp154_exp153_one_epoch_checkpoint300_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp154_answer_only_v66_one_epoch_20260725.json",
    },
    "EXP-155": {
        "audit": ROOT / "runs/exp155_answer_only_v66_sid_weight3_v100_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp155_answer_only_v66_sid3_20260725.json",
    },
    "EXP-156": {
        "audit": ROOT / "runs/exp156_exp155_one_epoch_checkpoint300_v1/audit_manifest.json",
        "upload": REPO / "07_接管/platform/upload_receipts/takeover_exp156_answer_only_v66_sid3_one_epoch_20260725.json",
    },
}


class MaterializeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition: raise MaterializeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_new(path: Path, payload: bytes) -> None:
    require(not path.exists() and not path.is_symlink(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path); os.unlink(temporary); temporary = ""
    finally:
        if temporary:
            try: os.unlink(temporary)
            except FileNotFoundError: pass


def render_config(model: Path, *, smoke: bool) -> str:
    dataset = "exp157_answer_only_r0_smoke_v2" if smoke else "exp157_answer_only_r0_compact_v2"
    registry = ROOT / ("dataset_registry_exp157_smoke" if smoke else "dataset_registry_exp157_answer_only_r0_compact")
    output = SMOKE_OUTPUT if smoke else FULL_OUTPUT
    schedule = "max_steps: 1\nnum_train_epochs: 1\nsave_strategy: 'no'" if smoke else "num_train_epochs: 2\nsave_strategy: epoch\nsave_total_limit: 2"
    scheduler = "constant" if smoke else "cosine"
    warmup = 0.0 if smoke else 0.03
    workers = 1 if smoke else 16
    return f"""# Generated only from the receipt-bound EXP-157 gate.
model_name_or_path: {model}
trust_remote_code: true
flash_attn: xformers
enable_liger_kernel: true
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 64
lora_dropout: 0.05
lora_target: all
loraplus_lr_ratio: 16.0
dataset: {dataset}
dataset_dir: {registry}
template: qwen3_nothink
cutoff_len: 16384
packing: true
neat_packing: true
overwrite_cache: false
preprocessing_num_workers: {workers}
dataloader_num_workers: 8
val_size: 0.0
eval_strategy: 'no'
output_dir: {output}
logging_steps: 1
{schedule}
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: none
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
lr_scheduler_type: {scheduler}
warmup_ratio: {warmup}
weight_decay: 0.0
max_grad_norm: 1.0
fp16: true
seed: 202607209
"""


def load_gate(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "receipt-bound gate is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == "llm-rec.takeover.exp157-receipt-bound-gate", "gate schema drift")
    require(value.get("gate_open") is True and value.get("target_achieved") is False, "EXP-157 gate is closed")
    require(value.get("decision") == "MATERIALIZE_EXP157_R0_CONTINUATION", "gate decision drift")
    require(not TARGET.exists(), "target marker already exists")
    return value


def verify_anchor(gate: dict[str, Any]) -> dict[str, Any]:
    anchor = gate.get("anchor", {})
    experiment = anchor.get("experiment")
    require(experiment in ANCHORS, "unsupported anchor")
    paths = ANCHORS[experiment]
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    upload = json.loads(paths["upload"].read_text(encoding="utf-8"))
    require(audit.get("experiment") == experiment and audit.get("verdict") == "PASS", "anchor audit is not PASS")
    require(audit.get("lineage_completeness") == "FULL", "anchor lineage is incomplete")
    model_id = anchor.get("model_id")
    require(isinstance(model_id, str) and model_id.startswith("md-"), "anchor formal model ID is invalid")
    require(upload.get("model_id") == model_id, "formal/upload model ID mismatch")
    receipt = Path(anchor.get("receipt_path", ""))
    require(record(receipt)["sha256"] == anchor.get("receipt_sha256"), "anchor receipt SHA drift")
    output = audit.get("output", {})
    merged = Path(output.get("directory", ""))
    artifact = output.get("artifacts", {}).get("model.safetensors", {})
    expected_model = Path(artifact.get("path", ""))
    require(expected_model == merged / "model.safetensors", "audit model path mismatch")
    if not expected_model.is_file():
        raise MaterializeError(f"RECONSTRUCT_REQUIRED:{experiment}:{merged}:{artifact.get('sha256')}")
    require(expected_model.stat().st_size == artifact.get("size_bytes"), "anchor model size drift")
    require(sha256(expected_model) == artifact.get("sha256"), "anchor model SHA drift")
    return {
        "experiment": experiment,
        "model_id": model_id,
        "model_dir": str(merged.resolve()),
        "model": record(expected_model),
        "audit": record(paths["audit"]),
        "upload": record(paths["upload"]),
        "formal_receipt": record(receipt),
        "formal_score": anchor["score"],
        "additional_r0_hits_if_non_r0_fixed": anchor["additional_r0_hits_if_non_r0_fixed"],
    }


def verify_inputs(gate: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    require(design.get("experiment") == "EXP-157", "design identity drift")
    for item in gate.get("data_bindings", {}).values():
        path = Path(item.get("path", ""))
        require(record(path)["sha256"] == item.get("sha256"), "gate data binding drift")
    contract = design.get("data_contract", {})
    require(contract.get("dataset_sha256") == "78ee84eefeadb677b8f627073a71fe100223c85fc684213a4ea77da857653ef9", "full dataset contract drift")
    require(contract.get("smoke_dataset_sha256") == "846ac973c0a76c23240b0599d58edff86a5bb737c934e02a350e7768ba8fc622", "smoke dataset contract drift")
    recipe = design.get("locked_training_design", {})
    require(recipe.get("learning_rate") == 5e-6 and recipe.get("lora_rank") == 64 and recipe.get("epochs") == 2, "recipe drift")
    return design, recipe


def materialize(gate_path: Path) -> dict[str, Any]:
    require(shutil.disk_usage("/workspace/tmp").free >= MIN_FREE, "less than 6 GiB free")
    require(not FULL_OUTPUT.exists() and not SMOKE_OUTPUT.exists(), "EXP-157 output already exists")
    gate = load_gate(gate_path)
    design, recipe = verify_inputs(gate)
    anchor = verify_anchor(gate)
    smoke = render_config(Path(anchor["model_dir"]), smoke=True).encode()
    full = render_config(Path(anchor["model_dir"]), smoke=False).encode()
    atomic_new(SMOKE_CONFIG, smoke)
    atomic_new(FULL_CONFIG, full)
    launch = {
        "schema": "llm-rec.takeover.exp157-launch-preregistration",
        "schema_version": 1,
        "experiment": "EXP-157",
        "status": "LOCKED_BEFORE_NONZERO_LR_SMOKE",
        "anchor": anchor,
        "recipe": recipe,
        "locks": {
            "gate": record(gate_path),
            "design": record(DESIGN),
            "smoke_config": record(SMOKE_CONFIG),
            "full_config": record(FULL_CONFIG),
            "full_dataset": record(REPO / design["data_contract"]["dataset_path"]),
            "smoke_dataset": record(REPO / design["data_contract"]["smoke_dataset_path"]),
            "target_mass": record(REPO / design["data_contract"]["target_mass_report"]),
        },
        "next_gate": "Run exactly one V100 optimizer step on the mixed 8+8 smoke slice; require finite loss, gradient norm, nonzero LR, finite adapter tensors, and measured ETA before full training.",
    }
    atomic_new(LAUNCH, (json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())
    return launch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=GATE)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    gate = load_gate(args.gate.resolve())
    verify_inputs(gate)
    anchor = verify_anchor(gate)
    if not args.execute:
        print(json.dumps({"mode": "DRY_RUN", "anchor": anchor, "free_bytes": shutil.disk_usage('/workspace/tmp').free}, sort_keys=True))
    else:
        launch = materialize(args.gate.resolve())
        print(json.dumps({"status": "MATERIALIZED", "anchor": launch["anchor"]["experiment"], "launch": str(LAUNCH)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
