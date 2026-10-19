#!/usr/bin/env python3
"""Create an immutable finite-tensor record for an EXP-157 epoch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


EXPECTED_STEPS = {543, 1086}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_new(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise SystemExit(f"REFUSE: output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise SystemExit("REFUSE: invalid checkpoint name") from error
    if step not in EXPECTED_STEPS:
        raise SystemExit("REFUSE: step is outside the locked EXP-157 epoch set")
    adapter = checkpoint / "adapter_model.safetensors"
    state_path = checkpoint / "trainer_state.json"
    optimizer = checkpoint / "optimizer.pt"
    for path in (adapter, state_path, optimizer):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"REFUSE: checkpoint is incomplete or unsafe: {path.name}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("global_step") != step or state.get("max_steps") != 1086:
        raise SystemExit("REFUSE: trainer-state geometry mismatch")

    tensors = elements = 0
    max_abs = 0.0
    dtypes: dict[str, int] = {}
    with safe_open(adapter, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if not bool(torch.isfinite(tensor).all()):
                raise SystemExit(f"FAIL: non-finite tensor {key}")
            tensors += 1
            elements += tensor.numel()
            dtypes[str(tensor.dtype)] = dtypes.get(str(tensor.dtype), 0) + 1
            if tensor.numel():
                max_abs = max(max_abs, float(tensor.detach().abs().max().item()))
    if tensors == 0 or not math.isfinite(max_abs):
        raise SystemExit("FAIL: empty or invalid adapter")
    matching = [row for row in state.get("log_history", []) if row.get("step") == step and "loss" in row]
    if len(matching) != 1:
        raise SystemExit("REFUSE: exact checkpoint metric is absent or ambiguous")
    metric = matching[0]
    for key in ("loss", "grad_norm", "learning_rate", "epoch"):
        if not math.isfinite(float(metric[key])):
            raise SystemExit(f"FAIL: non-finite metric {key}")
    record = {
        "schema": "llm-rec.takeover.exp157-checkpoint-trajectory",
        "schema_version": 1,
        "experiment": "EXP-157",
        "verdict": "PASS",
        "global_step": step,
        "checkpoint": str(checkpoint),
        "adapter": {
            "size_bytes": adapter.stat().st_size,
            "sha256": sha256(adapter),
            "tensors": tensors,
            "elements": elements,
            "dtypes": dtypes,
            "max_abs": max_abs,
            "all_finite": True,
        },
        "optimizer": {"size_bytes": optimizer.stat().st_size, "sha256": sha256(optimizer)},
        "trainer_state_sha256": sha256(state_path),
        "logged_metrics": {key: metric[key] for key in ("loss", "grad_norm", "learning_rate", "epoch", "step")},
    }
    atomic_new(args.output.resolve(), record)
    print(json.dumps({"verdict": "PASS", "step": step, "sha256": record["adapter"]["sha256"], "max_abs": max_abs}, sort_keys=True))


if __name__ == "__main__":
    main()
