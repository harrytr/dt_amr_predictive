#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

IMPORT_TRAIN = Path("import_high_train_pt_flat")
ENDOG_TRAIN = Path("endog_high_train_pt_flat")
OUT = Path("synthetic_amr_graphs_train")


def main() -> int:
    for src in [IMPORT_TRAIN, ENDOG_TRAIN]:
        if not src.exists():
            raise SystemExit(f"Missing source folder: {src}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    n = 0
    for src_dir, prefix in [(IMPORT_TRAIN, "IMPORT"), (ENDOG_TRAIN, "ENDOG")]:
        for p in sorted(src_dir.glob("*.pt")):
            dest = OUT / f"{prefix}__{p.name}"
            shutil.copy2(p, dest)
            n += 1

    print(f"COMBINED_DONE files={n} out={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
