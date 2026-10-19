#!/usr/bin/env python3
"""Install a fail-closed rescue-direction BEAR factor into Liger at process start."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from pathlib import Path
from typing import Any

import torch


MODE = "rescue_v1"
EXPECTED_LIGER_SOURCE_SHA256 = (
    "2f19063dee7435cbf9189fa4a377b13816e36820318d6a08f0e6381e53acadb1"
)
SID_RANGES = ((151669, 159861), (159861, 168053), (168053, 176245))
WEIGHT_ENV = (
    "LLM_REC_BEAR_WEIGHT_A",
    "LLM_REC_BEAR_WEIGHT_B",
    "LLM_REC_BEAR_WEIGHT_C",
)


class RescueBearPatchError(RuntimeError):
    """Raised when the runtime cannot prove the intended rescue patch is active."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_config() -> tuple[tuple[float, float, float], float, int]:
    if os.environ.get("LLM_REC_BEAR_MODE") != MODE:
        raise RescueBearPatchError(f"LLM_REC_BEAR_MODE must be exactly {MODE}")
    try:
        weights = tuple(float(os.environ.get(name, "0.0")) for name in WEIGHT_ENV)
        tau = float(os.environ.get("LLM_REC_BEAR_TAU", "1.0"))
        topk = int(os.environ.get("LLM_REC_BEAR_TOPK", "64"))
    except ValueError as error:
        raise RescueBearPatchError("rescue BEAR environment is not numeric") from error
    if len(weights) != 3 or any(
        not math.isfinite(value) or not 0.0 <= value <= 16.0
        for value in weights
    ):
        raise RescueBearPatchError("rescue BEAR weights must be finite in [0, 16]")
    if not any(value > 0.0 for value in weights):
        raise RescueBearPatchError("rescue BEAR requires at least one positive weight")
    if not math.isfinite(tau) or not 0.0 < tau <= 16.0:
        raise RescueBearPatchError("rescue BEAR tau must be finite in (0, 16]")
    if not 1 <= topk <= 8192:
        raise RescueBearPatchError("rescue BEAR top-k must be in [1, 8192]")
    return (weights[0], weights[1], weights[2]), tau, topk


def rescue_bear_scaling_factors(
    logits_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor | None:
    """Weight low-margin SID targets more strongly, with detached FP32 factors."""

    step_weights, tau, topk = runtime_config()
    if logits_chunk.ndim != 2 or target_chunk.ndim != 1:
        raise RescueBearPatchError("rescue BEAR received tensors with invalid ranks")
    if logits_chunk.shape[0] != target_chunk.shape[0] or logits_chunk.shape[1] < 1:
        raise RescueBearPatchError("rescue BEAR received incompatible logits and targets")
    valid_sid = torch.zeros_like(target_chunk, dtype=torch.bool)
    for (start, end), step_weight in zip(SID_RANGES, step_weights):
        if step_weight > 0.0:
            valid_sid |= (target_chunk >= start) & (target_chunk < end)
    valid_sid &= target_chunk != ignore_index
    if not bool(valid_sid.any()):
        return None
    scaling = torch.ones(
        target_chunk.shape, dtype=torch.float32, device=target_chunk.device
    )
    valid_indices = valid_sid.nonzero(as_tuple=False).squeeze(-1)
    valid_targets = target_chunk.index_select(0, valid_indices)
    valid_logits = logits_chunk.index_select(0, valid_indices).detach().float()
    kth_logits = torch.topk(
        valid_logits, k=min(topk, valid_logits.shape[-1]), dim=-1
    ).values[:, -1]
    target_logits = valid_logits.gather(1, valid_targets[:, None]).squeeze(1)
    factors = torch.ones_like(target_logits, dtype=torch.float32)
    rescue_margin = (kth_logits - target_logits) / tau
    for (start, end), step_weight in zip(SID_RANGES, step_weights):
        mask = (valid_targets >= start) & (valid_targets < end)
        if step_weight > 0.0 and bool(mask.any()):
            factors[mask] = 1.0 + step_weight * torch.sigmoid(rescue_margin[mask])
    scaling[valid_indices] = factors
    return scaling


def install_rescue_bear_patch() -> dict[str, Any]:
    """Verify the exact Liger source and replace only its BEAR helper."""

    runtime_config()
    from liger_kernel.ops import fused_linear_cross_entropy as fused

    source_path = Path(inspect.getsourcefile(fused) or "").resolve()
    if not source_path.is_file() or source_path.is_symlink():
        raise RescueBearPatchError("Liger fused CE source is not a regular file")
    actual_sha = sha256_file(source_path)
    if actual_sha != EXPECTED_LIGER_SOURCE_SHA256:
        raise RescueBearPatchError(
            "Liger fused CE source SHA drifted; refusing an unverified monkeypatch"
        )
    if getattr(fused, "_llm_rec_bear_patch_mode", None) == MODE:
        if fused._bear_scaling_factors is not rescue_bear_scaling_factors:
            raise RescueBearPatchError("rescue BEAR mode marker and function disagree")
    else:
        if not callable(getattr(fused, "_bear_scaling_factors", None)):
            raise RescueBearPatchError("Liger exposes no BEAR helper to replace")
        fused._bear_scaling_factors = rescue_bear_scaling_factors
        fused._llm_rec_bear_patch_mode = MODE
    return {
        "verdict": "PASS",
        "mode": MODE,
        "formula": "1 + weight * sigmoid((kth_logit - target_logit) / tau)",
        "liger_source": str(source_path),
        "liger_source_sha256": actual_sha,
        "patch_source": str(Path(__file__).resolve()),
        "patch_source_sha256": sha256_file(Path(__file__).resolve()),
    }
