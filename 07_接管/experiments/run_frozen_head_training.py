#!/usr/bin/env python3
"""Audited run_training adapter that propagates the frozen-head Liger fix."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import run_training


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "frozen_head_liger_site"
PATCH = ROOT / "liger_frozen_head_patch.py"
ADAPTER = Path(__file__).resolve()
BASE_ENV = run_training.training_environment


def environment(cuda_visible_devices: str, inherited_path: str | None = None, *, bear: Mapping | None = None) -> dict[str, str]:
    value = BASE_ENV(cuda_visible_devices, inherited_path, bear=bear)
    inherited = os.environ.get("PYTHONPATH", "")
    value["PYTHONPATH"] = os.pathsep.join(part for part in (str(SITE), str(ROOT), inherited) if part)
    value["LLMREC_FROZEN_HEAD_LIGER_FIX"] = "1"
    return value


def main() -> int:
    old_launcher = run_training.LAUNCHER_PATH
    old_runtime = run_training.RUNTIME_PATHS
    old_environment = run_training.training_environment
    try:
        run_training.LAUNCHER_PATH = ADAPTER
        run_training.RUNTIME_PATHS = tuple(dict.fromkeys((*old_runtime, ADAPTER, PATCH, SITE / "sitecustomize.py")))
        run_training.training_environment = environment
        return run_training.main()
    finally:
        run_training.LAUNCHER_PATH = old_launcher
        run_training.RUNTIME_PATHS = old_runtime
        run_training.training_environment = old_environment


if __name__ == "__main__":
    raise SystemExit(main())
