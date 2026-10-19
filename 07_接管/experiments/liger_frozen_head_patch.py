#!/usr/bin/env python3
"""Fail-closed Liger fix for a trainable lm_head over frozen hidden states."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "765747ba10bae599cd65c7c5f3dfebfd620e526bceb7696c8cef6b0cf42433ce"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def install() -> dict:
    from liger_kernel.ops import fused_linear_cross_entropy as fused

    source = Path(inspect.getsourcefile(fused) or "").resolve()
    if not source.is_file() or source.is_symlink() or sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Liger fused CE source lock drift")
    if getattr(fused, "_llmrec_frozen_head_fix", False):
        return {"verdict": "PASS", "already_installed": True, "source_sha256": EXPECTED_SOURCE_SHA256}
    original = fused.fused_linear_cross_entropy_forward

    def corrected(_input, weight, *args, **kwargs):
        # Liger 0.6.4 allocates grad_weight only when input.requires_grad.  A
        # frozen backbone makes that false even though lm_head is trainable.
        # A detached leaf requests the same local fused backward without
        # opening a gradient path through the frozen transformer.
        if weight.requires_grad and not _input.requires_grad:
            _input = _input.detach().requires_grad_(True)
        return original(_input, weight, *args, **kwargs)

    fused.fused_linear_cross_entropy_forward = corrected
    fused._llmrec_frozen_head_fix = True
    return {"verdict": "PASS", "already_installed": False, "source_sha256": EXPECTED_SOURCE_SHA256}
