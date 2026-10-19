#!/usr/bin/env python3
"""LLaMA-Factory entrypoint that installs the locked rescue BEAR patch first."""

from __future__ import annotations

import json
import sys

from liger_rescue_bear_patch import install_rescue_bear_patch


def main() -> int:
    patch_record = install_rescue_bear_patch()
    if sys.argv[1:] == ["--verify-patch-only"]:
        print(json.dumps(patch_record, ensure_ascii=False, sort_keys=True))
        return 0
    from llamafactory.cli import main as llamafactory_main

    result = llamafactory_main()
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
