#!/usr/bin/env python3
"""EXP-173 profile for the shared audited LoRA merger."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import finalize_exp135 as impl
import run_training


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
PREREG = ROOT / "preregistrations/exp173_exp158_unambiguous_r0_continuation_full_launch.json"
if not PREREG.is_file():
    raise SystemExit("REFUSE: EXP-173 full preregistration is missing")
value = json.loads(PREREG.read_text(encoding="utf-8"))
if value.get("status") != "LOCKED_BEFORE_TRAINING" or value.get("experiment") != "EXP-173":
    raise SystemExit("REFUSE: EXP-173 preregistration drift")

impl.RUN_ID = "exp173_exp158_unambiguous_r0_continuation_v100_v1"
impl.EXPERIMENT = "EXP-173"
impl.TRAINING_STARTED_AT = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
impl.RUN_DIR = ROOT / "runs" / impl.RUN_ID
impl.CONFIG = ROOT / "configs/exp173_exp158_unambiguous_r0_r64_a64_xformers_16k_ga8_fp16.yaml"
impl.PREREG = PREREG
impl.DATASET = ROOT / "data/exp173_exp158_public_answer_only_plus_unambiguous_r0_v1.jsonl"
impl.DATASET_MANIFEST = ROOT / "manifests/exp173_exp158_public_answer_only_plus_unambiguous_r0_v1.manifest.json"
impl.REGISTRY = ROOT / "dataset_registry_exp173_unambiguous_r0/dataset_info.json"
impl.EXTERNAL_RECIPE = ROOT / "preregistrations/exp161_external_unambiguous_r0_prompt_design.json"
impl.BASE = ROOT / "outputs/onereason_0.8b_exp158_exp157_one_epoch_r0_continuation_merged_bf16"
impl.BASE_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
impl.ADAPTER = ROOT / "outputs/onereason_0.8b_exp173_exp158_unambiguous_r0_r64_a64_xformers_16k_ga8_fp16"
impl.MERGED = Path(str(impl.ADAPTER) + "_merged_bf16")
impl.LAUNCH_MANIFEST = impl.RUN_DIR / "launch_manifest.json"
impl.COMPLETION_MANIFEST = impl.RUN_DIR / "completion_manifest.json"
impl.AUDIT_MANIFEST = impl.RUN_DIR / "audit_manifest.json"
impl.EXPECTED_RANK = 64
impl.EXPECTED_ALPHA = 64
impl.EXPECTED_DROPOUT = 0.05
impl.EXPECTED_RSLORA = False
impl.EXPECTED_EPOCHS = 1.0
state_path = impl.ADAPTER / "trainer_state.json"
if "--finalize" in sys.argv:
    if not state_path.is_file():
        raise SystemExit("REFUSE: final trainer state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = state.get("max_steps")
    if not isinstance(steps, int) or steps <= 0:
        raise SystemExit("REFUSE: invalid dynamic max_steps")
    impl.EXPECTED_STEPS = steps


def ensure_merge_space() -> None:
    """Reclaim only EXP-173's reproducible HF cache if the 8 GiB merge gate needs it."""
    minimum = 8 * 1024**3
    if shutil.disk_usage("/workspace/tmp").free >= minimum:
        return
    cache = Path("/workspace/tmp/llmrec-hf-cache-exp173")
    receipt = impl.RUN_DIR / "hf_cache_cleanup_before_merge.json"
    if cache.resolve() != Path("/workspace/tmp/llmrec-hf-cache-exp173") or not cache.is_dir() or cache.is_symlink():
        raise SystemExit("REFUSE: merge space below 8 GiB and exact EXP-173 cache is unavailable")
    state = json.loads((impl.ADAPTER / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("global_step") != state.get("max_steps") or float(state.get("epoch", 0.0)) < 1.0:
        raise SystemExit("REFUSE: training is not complete; cache cleanup forbidden")
    launch = json.loads(PREREG.read_text(encoding="utf-8"))
    locks = (launch["dataset"], launch["full_config"], launch["initializer_model"])
    for item in locks:
        path = Path(item["path"])
        if run_training.sha256_file(path) != item["sha256"]:
            raise SystemExit("REFUSE: reconstructibility input lock drift")
    files = []
    for path in sorted(cache.rglob("*")):
        if path.is_symlink():
            raise SystemExit(f"REFUSE: cache contains symlink: {path}")
        if path.is_file():
            files.append(run_training.file_record(path))
    before = shutil.disk_usage("/workspace/tmp").free
    planned = {
        "schema": "llm-rec.takeover.exp173-reconstructible-cache-cleanup",
        "schema_version": 1,
        "experiment": "EXP-173",
        "status": "PLANNED_AFTER_COMPLETED_TRAINING_BEFORE_MERGE",
        "cache": str(cache),
        "files": files,
        "reconstruction_inputs": list(locks),
        "adapter": run_training.file_record(impl.ADAPTER / "adapter_model.safetensors"),
        "free_bytes_before": before,
    }
    run_training.atomic_write(receipt.with_suffix(".planned.json"), run_training.canonical_json_bytes(planned))
    shutil.rmtree(cache)
    after = shutil.disk_usage("/workspace/tmp").free
    planned.update({"status": "COMPLETED", "free_bytes_after": after, "bytes_reclaimed": after - before})
    run_training.atomic_write(receipt, run_training.canonical_json_bytes(planned))
    if after < minimum:
        raise SystemExit("REFUSE: still below 8 GiB after exact cache cleanup")


if __name__ == "__main__":
    if "--finalize" in sys.argv:
        ensure_merge_space()
    raise SystemExit(impl.main())
