#!/usr/bin/env python3
"""Verify the materialized EXP-173 mix and its immutable prior isolation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
DESIGN = ROOT / "preregistrations/exp173_exp158_unambiguous_r0_continuation_design.json"
MANIFEST = ROOT / "manifests/exp173_exp158_public_answer_only_plus_unambiguous_r0_v1.manifest.json"
KEYS = ("instruction", "input", "output", "system", "history")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> int:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(design.get("experiment") == "EXP-173", "design drift")
    require(sha256(MANIFEST) == design["data"]["manifest"]["sha256"], "manifest drift")
    require(manifest.get("experiment") == "EXP-173", "manifest identity drift")
    for key in ("full", "smoke"):
        item = design["data"][key]
        path = Path(item["path"])
        require(path.is_file() and not path.is_symlink(), f"missing {key} dataset")
        require(sha256(path) == item["sha256"], f"{key} dataset drift")
        rows = 0
        with path.open("rb") as f:
            for rows, raw in enumerate(f, 1):
                require(raw.endswith(b"\n"), f"newline drift at {key}:{rows}")
                value = json.loads(raw)
                require(tuple(value) == KEYS, f"schema drift at {key}:{rows}")
                require(isinstance(value["history"], list), f"history drift at {key}:{rows}")
                require(all(isinstance(value[name], str) for name in KEYS[:-1]), f"text drift at {key}:{rows}")
        require(rows == item["rows"], f"row-count drift for {key}")
    source = design["data"]["source_audit"]
    source_path = Path(source["path"])
    require(source_path.is_file() and sha256(source_path) == source["sha256"], "prior novelty/isolation audit drift")
    novel = Path(manifest["sources"]["novel_unambiguous_r0"]["path"])
    require(sha256(novel) == manifest["sources"]["novel_unambiguous_r0"]["sha256"], "novel source drift")
    print(json.dumps({"verdict": "PASS", "full_rows": design["data"]["full"]["rows"], "novel_rows": design["data"]["novel_rows"], "isolation_evidence_bound": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
