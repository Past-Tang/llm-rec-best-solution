#!/usr/bin/env python3
"""EXP-157 profile for the shared LoRA audit and CPU BF16 merger."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import finalize_exp135 as impl


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
PREREG = ROOT / "preregistrations/exp157_answer_only_r0_continuation_full_launch.json"
if not PREREG.is_file():
    raise SystemExit("REFUSE: EXP-157 full preregistration is missing")
value = json.loads(PREREG.read_text(encoding="utf-8"))
if value.get("status") != "LOCKED_BEFORE_TRAINING" or value.get("experiment") != "EXP-157":
    raise SystemExit("REFUSE: EXP-157 preregistration drift")
base = Path(value.get("anchor", {}).get("model_dir", ""))
if not (base / "model.safetensors").is_file():
    raise SystemExit("REFUSE: receipt-bound anchor model is missing")

impl.RUN_ID = "exp157_answer_only_r0_continuation_v100_v1"
impl.EXPERIMENT = "EXP-157"
impl.TRAINING_STARTED_AT = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
impl.RUN_DIR = ROOT / "runs" / impl.RUN_ID
impl.CONFIG = ROOT / "configs/exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16.yaml"
impl.PREREG = PREREG
impl.DATASET = ROOT / "data/exp157_answer_only_r0_compact_v2.jsonl"
impl.DATASET_MANIFEST = ROOT / "manifests/exp157_answer_only_r0_compact_v2.manifest.json"
impl.REGISTRY = ROOT / "dataset_registry_exp157_answer_only_r0_compact/dataset_info.json"
impl.EXTERNAL_RECIPE = REPO / "07_接管/analysis/generated/exp157_answer_only_r0_target_mass_v2.json"
impl.BASE = base
impl.BASE_FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
impl.ADAPTER = ROOT / "outputs/onereason_0.8b_exp157_answer_only_r0_continuation_r64_a64_xformers_16k_ga8_fp16"
impl.MERGED = Path(str(impl.ADAPTER) + "_merged_bf16")
impl.LAUNCH_MANIFEST = impl.RUN_DIR / "launch_manifest.json"
impl.COMPLETION_MANIFEST = impl.RUN_DIR / "completion_manifest.json"
impl.AUDIT_MANIFEST = impl.RUN_DIR / "audit_manifest.json"
impl.EXPECTED_RANK = 64
impl.EXPECTED_ALPHA = 64
impl.EXPECTED_DROPOUT = 0.05
impl.EXPECTED_RSLORA = False
impl.EXPECTED_EPOCHS = 2.0
state_path = impl.ADAPTER / "trainer_state.json"
if "--finalize" in sys.argv:
    if not state_path.is_file(): raise SystemExit("REFUSE: final trainer state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = state.get("max_steps")
    if not isinstance(steps, int) or steps <= 0: raise SystemExit("REFUSE: invalid dynamic max_steps")
    impl.EXPECTED_STEPS = steps


if __name__ == "__main__": raise SystemExit(impl.main())
