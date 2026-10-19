#!/usr/bin/env python3
"""Rebuild the EXP-136 anchor and EXP-158 merged BF16 model locally."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_SHA = "28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90"
EXP136_ADAPTER_SHA = "8b989fdf1b0e85dfd27a37a09213dc991f1baefe36697df7bbc79ca62eca25d4"
EXP136_MERGED_SHA = "2c6322aeb3e0645bd0cb2234580b5e7ed4007456d9931857ce472f67f15d36a5"
EXP158_ADAPTER_SHA = "701fd34c1995037d499a84812d4dfe3280340131926957999ba58dbf8ebcf8df"
EXP158_MERGED_SHA = "a7e3d5edb59419c2a15c992b3c5c62c2c5924cd92a91a69e6cd4e7ad9a5defaf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, expected: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or unsafe file: {path}")
    observed = sha256(path)
    if observed != expected:
        raise SystemExit(f"SHA-256 drift: {path}\nexpected {expected}\nobserved {observed}")


def merge(base: Path, adapter: Path, output: Path, expected: str) -> None:
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise SystemExit(f"refusing stale temporary output: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": "cpu"},
        )
        merged = PeftModel.from_pretrained(model, adapter, is_trainable=False).merge_and_unload(
            safe_merge=True
        )
        merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(adapter, trust_remote_code=True).save_pretrained(temporary)
        for name in ("generation_config.json", "chat_template.jinja", "special_tokens_map.json"):
            source = adapter / name if (adapter / name).is_file() else base / name
            if source.is_file() and not (temporary / name).exists():
                shutil.copy2(source, temporary / name)
        require_file(temporary / "model.safetensors", expected)
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    base = args.base_dir.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output root must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    exp136_adapter = repo / (
        "07_接管/experiments/outputs/"
        "onereason_0.8b_exp136_public091_v66_rslora_loraplus_r128_a16_xformers_16k_ga8_fp16"
    )
    exp158_adapter = repo / (
        "07_接管/experiments/outputs/"
        "onereason_0.8b_exp158_exp157_one_epoch_r0_continuation"
    )
    require_file(base / "model.safetensors", BASE_SHA)
    require_file(exp136_adapter / "adapter_model.safetensors", EXP136_ADAPTER_SHA)
    require_file(exp158_adapter / "adapter_model.safetensors", EXP158_ADAPTER_SHA)

    anchor = output_root / "exp136_merged_bf16"
    final = output_root / "exp158_merged_bf16"
    merge(base, exp136_adapter, anchor, EXP136_MERGED_SHA)
    merge(anchor, exp158_adapter, final, EXP158_MERGED_SHA)
    print(f"PASS\nanchor={anchor}\nfinal={final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
