#!/usr/bin/env python3
"""EXP-136 profile for the shared adapter audit and BF16 merge finalizer."""

from pathlib import Path

import finalize_exp135 as impl


impl.RUN_ID = "exp136_public091_v66_v100_16k_ga8_v1"
impl.EXPERIMENT = "EXP-136"
impl.TRAINING_STARTED_AT = "recorded immediately after the EXP-136 process starts"
impl.RUN_DIR = impl.EXP_ROOT / "runs" / impl.RUN_ID
impl.CONFIG = impl.EXP_ROOT / "configs/exp136_public091_v66_rslora_loraplus_r128_a16_xformers_16k_ga8_fp16.yaml"
impl.PREREG = impl.EXP_ROOT / "preregistrations/exp136_public091_v66_v100_16k_ga8_v1.json"
impl.EXTERNAL_RECIPE = (
    impl.REPO_ROOT / "07_接管/analysis/external_evidence/lldois_v66_recipe_223853ad.json"
)
impl.ADAPTER = (
    impl.EXP_ROOT
    / "outputs/onereason_0.8b_exp136_public091_v66_rslora_loraplus_r128_a16_xformers_16k_ga8_fp16"
)
impl.MERGED = Path(str(impl.ADAPTER) + "_merged_bf16")
impl.LAUNCH_MANIFEST = impl.RUN_DIR / "launch_manifest.json"
impl.COMPLETION_MANIFEST = impl.RUN_DIR / "completion_manifest.json"
impl.AUDIT_MANIFEST = impl.RUN_DIR / "audit_manifest.json"
impl.EXPECTED_RANK = 128
impl.EXPECTED_ALPHA = 16
impl.EXPECTED_RSLORA = True


if __name__ == "__main__":
    raise SystemExit(impl.main())
