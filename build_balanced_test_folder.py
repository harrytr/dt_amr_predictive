#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

IMPORT_TEST_DIR = Path("import_high_test_pt_flat")
ENDOG_TEST_DIR = Path("endog_high_test_pt_flat")
OUT = Path("synthetic_amr_graphs_test")

N_EACH = 150


def copy_with_prefix(src_dir: Path, prefix: str, n: int) -> int:
    pts = sorted(src_dir.glob("*.pt"))[:n]
    c = 0
    for p in pts:
        dest = OUT / f"{prefix}__{p.name}"
        shutil.copy2(p, dest)
        c += 1
    return c


def main() -> int:
    for src in [IMPORT_TEST_DIR, ENDOG_TEST_DIR]:
        if not src.exists():
            raise SystemExit(f"Missing source folder: {src}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    n_import = copy_with_prefix(IMPORT_TEST_DIR, "IMPORT", N_EACH)
    n_endog = copy_with_prefix(ENDOG_TEST_DIR, "ENDOG", N_EACH)

    total = len(list(OUT.glob("*.pt")))
    print(
        f"OK test_folder={OUT} copied_import={n_import} copied_endog={n_endog} total_files={total}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
