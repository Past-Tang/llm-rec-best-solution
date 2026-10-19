#!/usr/bin/env python3
"""Build and fully audit the locked EXP-174 negative LoRA task-vector probe."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import audit_recovery_run
import run_training


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
DESIGN = ROOT / "preregistrations/exp174_exp173_negative_task_vector_probe.json"
BASE = ROOT / "outputs/onereason_0.8b_exp158_exp157_one_epoch_r0_continuation_merged_bf16"
SOURCE = ROOT / "outputs/onereason_0.8b_exp173_exp158_unambiguous_r0_r64_a64_xformers_16k_ga8_fp16"
DERIVED = ROOT / "outputs/onereason_0.8b_exp174_exp173_negative_half_adapter"
MERGED = ROOT / "outputs/onereason_0.8b_exp174_exp173_negative_half_merged_bf16"
RUN = ROOT / "runs/exp174_exp173_negative_half_task_vector_probe_v1"
AUDIT = RUN / "audit_manifest.json"
LEDGER = REPO / "07_接管/truth_ledger/generated/truth_ledger.json"
EXP158 = REPO / "07_接管/platform/receipts/official_exp158_exp157_one_epoch_r0_continuation_20260726_a_metrics.json"
EXP173 = REPO / "07_接管/platform/receipts/official_exp173_exp158_unambiguous_r0_20260728_a_metrics.json"
TARGET = REPO / "07_接管/autopilot/TARGET_1_1_ACHIEVED.json"
COEFFICIENT = -0.5


class ProbeError(RuntimeError): pass
def require(value: bool, message: str) -> None:
    if not value: raise ProbeError(message)
def record(path: Path, sha: str | None = None) -> dict:
    require(path.is_file() and not path.is_symlink(), f"missing or unsafe file: {path}")
    value = run_training.file_record(path)
    if sha is not None: require(value["sha256"] == sha, f"hash drift: {path}")
    return value


def verify_inputs() -> dict:
    require(not TARGET.exists(), "target marker already exists")
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    require(design.get("status") == "LOCKED_BEFORE_DERIVATION_MERGE_UPLOAD_OR_FORMAL", "design status drift")
    require(design.get("transform", {}).get("coefficient") == COEFFICIENT, "coefficient drift")
    base = record(BASE / "model.safetensors", "a7e3d5edb59419c2a15c992b3c5c62c2c5924cd92a91a69e6cd4e7ad9a5defaf")
    adapter = record(SOURCE / "adapter_model.safetensors", "e692f129347263afa026f77c487bed4a73732658d16ceae623a8a16db04c40e6")
    config = record(SOURCE / "adapter_config.json", "e9ecadc7dbd2314967bfff0c201324450abdfcb44f6b3c65c5eedaaea179d61b")
    ledger = record(LEDGER, "f9e44d6c64c3eb8c40d4070bb776986a633bba8d3fa19331f9206095a463c5b9")
    r158 = record(EXP158, "c24ba496b0db71ea32648a130fd1b06c55a3a4098d442600cf328d1a1c079d74")
    r173 = record(EXP173, "847daa984b464b535f6499569f9aa709e124478c2dabdfa566485c208aa62225")
    ld = json.loads(LEDGER.read_text()); f158 = json.loads(EXP158.read_text())["responseData"]; f173 = json.loads(EXP173.read_text())["responseData"]
    require(f158["taskStatus"] == f173["taskStatus"] == "SUCCEEDED" and f158["evalType"] == f173["evalType"] == "FORMAL", "formal receipt gate closed")
    require(f158["score"] == 1.0350521506259462 and f173["score"] == 0.8653390712826569, "formal score drift")
    for formal in (f158, f173):
        entries = [x for x in ld.get("evaluations", []) if x.get("eval_task_id") == formal["evalTaskId"]]
        require(len(entries) == 1 and entries[0]["score"] == formal["score"], "ledger receipt binding missing")
    finite = audit_recovery_run.scan_safetensors(SOURCE / "adapter_model.safetensors")
    require(finite.get("all_finite") is True and finite.get("tensor_count") == 392, "source adapter finite/layout drift")
    return {"design": record(DESIGN), "base_model": base, "source_adapter": adapter, "source_adapter_config": config, "truth_ledger": ledger, "exp158_receipt": r158, "exp173_receipt": r173, "source_finite_scan": finite}


def execute() -> dict:
    inputs = verify_inputs()
    require(not DERIVED.exists() and not MERGED.exists() and not AUDIT.exists(), "EXP-174 output conflict")
    require(shutil.disk_usage("/workspace/tmp").free >= 8 * 1024**3, "less than 8 GiB free before merge")
    DERIVED.mkdir(parents=True)
    MERGED.mkdir(parents=True)
    try:
        tensors = {}
        a_count = b_count = 0
        with safe_open(SOURCE / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if "lora_B" in key:
                    tensor = (tensor * COEFFICIENT).contiguous(); b_count += 1
                else:
                    require("lora_A" in key, f"unexpected adapter tensor: {key}"); tensor = tensor.contiguous(); a_count += 1
                tensors[key] = tensor
        require(a_count == b_count == 196, "LoRA A/B layout drift")
        save_file(tensors, DERIVED / "adapter_model.safetensors", metadata={"format": "pt", "llmrec_transform": "lora_B_times_negative_0.5"})
        for path in SOURCE.iterdir():
            if path.is_file() and not path.is_symlink() and path.name != "adapter_model.safetensors": shutil.copy2(path, DERIVED / path.name)
        derived_finite = audit_recovery_run.scan_safetensors(DERIVED / "adapter_model.safetensors")
        require(derived_finite.get("all_finite") is True and derived_finite.get("tensor_count") == 392, "derived adapter finite/layout drift")
        algebra = {"coefficient": COEFFICIENT, "lora_A_tensors_unchanged": 0, "lora_B_tensors_scaled": 0, "max_abs_error": 0.0}
        with safe_open(SOURCE / "adapter_model.safetensors", framework="pt", device="cpu") as src, safe_open(DERIVED / "adapter_model.safetensors", framework="pt", device="cpu") as dst:
            for key in src.keys():
                expected = src.get_tensor(key) * (COEFFICIENT if "lora_B" in key else 1.0)
                actual = dst.get_tensor(key); error = (expected - actual).abs().max().item()
                algebra["max_abs_error"] = max(algebra["max_abs_error"], error)
                require(error == 0.0, f"derived algebra drift: {key}")
                algebra["lora_B_tensors_scaled" if "lora_B" in key else "lora_A_tensors_unchanged"] += 1
        base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": "cpu"})
        merged = PeftModel.from_pretrained(base, DERIVED, is_trainable=False).merge_and_unload(safe_merge=True)
        merged.save_pretrained(MERGED, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(BASE, trust_remote_code=True).save_pretrained(MERGED)
        for name in ("generation_config.json", "chat_template.jinja", "special_tokens_map.json"):
            if (BASE / name).is_file() and not (MERGED / name).exists(): shutil.copy2(BASE / name, MERGED / name)
        merged_finite = audit_recovery_run.scan_safetensors(MERGED / "model.safetensors")
        require(merged_finite == {"tensor_count":311,"element_count":801433600,"dtype_counts":{"BF16":311},"all_finite":True,"nonfinite_tensors":[]}, "merged finite/layout drift")
        artifacts = {p.name: run_training.file_record(p) for p in sorted(MERGED.iterdir()) if p.is_file() and not p.is_symlink()}
        value = {"schema":"llm-rec.takeover.training-run-audit","schema_version":1,"experiment":"EXP-174","run_id":RUN.name,"verdict":"PASS","lineage_completeness":"FULL","audit_kind":"deterministic_task_vector_transform","inputs":inputs,"transform_algebra":algebra,"derived_adapter":{"directory":str(DERIVED),"files":{"adapter_model.safetensors":run_training.file_record(DERIVED / "adapter_model.safetensors"),"adapter_config.json":run_training.file_record(DERIVED / "adapter_config.json")},"finite_scan":derived_finite},"output":{"directory":str(MERGED),"artifacts":artifacts},"finite_scan":merged_finite,"efficacy_claim":"unknown until one formal evaluation","formal_policy":"Exactly one formal evaluation; no coefficient sweep."}
        RUN.mkdir(parents=True, exist_ok=True)
        run_training.atomic_write(AUDIT, run_training.canonical_json_bytes(value))
        run_training.atomic_write(AUDIT.with_suffix(".json.sha256"), f"{run_training.sha256_file(AUDIT)}  {AUDIT.name}\n".encode())
        return value
    except BaseException:
        if MERGED.exists() and not AUDIT.exists(): shutil.rmtree(MERGED)
        if DERIVED.exists() and not AUDIT.exists(): shutil.rmtree(DERIVED)
        raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--execute",action="store_true"); args=parser.parse_args()
    try: value=execute() if args.execute else verify_inputs()
    except (ProbeError,OSError,UnicodeError,json.JSONDecodeError,KeyError,TypeError,ValueError,RuntimeError) as error: print(f"FAIL: {error}"); return 1
    print(json.dumps({"verdict":"PASS","mode":"execute" if args.execute else "verify","audit":str(AUDIT) if args.execute else None})); return 0
if __name__ == "__main__": raise SystemExit(main())
