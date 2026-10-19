#!/usr/bin/env python3
"""Materialize the locked EXP-173 public replay plus novel unambiguous R0 mix."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "07_接管/experiments"
PUBLIC = ROOT / "data/exp142_public091_answer_only_c48c42a_v1.jsonl"
NOVEL = ROOT / "data/exp161_external_unambiguous_r0_v1.jsonl"
OUTPUT = ROOT / "data/exp173_exp158_public_answer_only_plus_unambiguous_r0_v1.jsonl"
SMOKE = ROOT / "data/exp173_exp158_unambiguous_r0_smoke_v1.jsonl"
MANIFEST = ROOT / "manifests/exp173_exp158_public_answer_only_plus_unambiguous_r0_v1.manifest.json"
LOCKS = {
    PUBLIC: "ad4f2fe884e190b255c9270d94401e8f5760cc7e57bac6781bb3d539ff2ebe5a",
    NOVEL: "f3f8c1e3915d8da770aafc84d52471b327598ba64d10bcf8d65b38299563df47",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


KEYS = ("instruction", "input", "output", "system", "history")


def count_and_validate(path: Path, *, source: bool = False) -> int:
    rows = 0
    with path.open("rb") as f:
        for rows, raw in enumerate(f, 1):
            if not raw.endswith(b"\n"):
                raise RuntimeError(f"missing newline in {path} row {rows}")
            value = json.loads(raw)
            allowed = {KEYS, ("instruction", "input", "output", "history")} if source else {KEYS}
            if tuple(value) not in allowed:
                raise RuntimeError(f"schema drift in {path} row {rows}")
    return rows


def normalized(raw: bytes) -> bytes:
    value = json.loads(raw)
    value.setdefault("system", "")
    ordered = {key: value[key] for key in KEYS}
    return (json.dumps(ordered, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def new_stream(path: Path, sources: tuple[Path, ...]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            for source in sources:
                with source.open("rb") as inp:
                    for raw in inp:
                        out.write(normalized(raw))
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def new_smoke(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to overwrite {path}")
    rows: list[bytes] = []
    for source in (PUBLIC, NOVEL):
        with source.open("rb") as f:
            rows.extend(normalized(next(f)) for _ in range(8))
    payload = b"".join(rows)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.link(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def main() -> int:
    for path, digest in LOCKS.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"source lock failed: {path}")
    public_rows = count_and_validate(PUBLIC, source=True)
    novel_rows = count_and_validate(NOVEL, source=True)
    if (public_rows, novel_rows) != (32705, 5174):
        raise RuntimeError("source row-count drift")
    new_stream(OUTPUT, (PUBLIC, NOVEL))
    new_smoke(SMOKE)
    output_rows = count_and_validate(OUTPUT)
    smoke_rows = count_and_validate(SMOKE)
    if output_rows != 37879 or smoke_rows != 16:
        raise RuntimeError("materialized row-count drift")
    value = {
        "schema": "llm-rec.takeover.exp173-dataset-manifest",
        "schema_version": 1,
        "experiment": "EXP-173",
        "policy": "content- and order-exact public answer-only replay with only an empty system field added for schema parity, followed by locked globally unambiguous evaluation-SID-excluded R0 rows",
        "controlled_schema_transform": "Rows lacking system receive system=''; instruction, input, output and history are unchanged.",
        "sources": {
            "public_answer_only": {"path": str(PUBLIC), "rows": public_rows, "sha256": sha256(PUBLIC)},
            "novel_unambiguous_r0": {"path": str(NOVEL), "rows": novel_rows, "sha256": sha256(NOVEL)},
        },
        "output": {"path": str(OUTPUT), "rows": output_rows, "size_bytes": OUTPUT.stat().st_size, "sha256": sha256(OUTPUT)},
        "smoke": {"path": str(SMOKE), "rows": smoke_rows, "size_bytes": SMOKE.stat().st_size, "sha256": sha256(SMOKE)},
        "novel_fraction_by_rows": novel_rows / output_rows,
        "efficacy_claim": "unknown until formal evaluation",
    }
    atomic_json(MANIFEST, value)
    print(json.dumps({"verdict": "PASS", "rows": output_rows, "sha256": value["output"]["sha256"], "smoke_sha256": value["smoke"]["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
