#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path
import torch


SRC = Path("synthetic_endog_import_step7c_sweep_pt_flat")
OUT = Path("synthetic_amr_graphs_test")
LABEL_ATTR = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")

N0 = 150
N1 = 150


def main() -> int:
    pts = sorted(SRC.glob("*.pt"))
    if not pts:
        raise SystemExit("No pt files found")

    zeros = []
    ones = []
    for p in pts:
        d = torch.load(p, weights_only=False)
        if not hasattr(d, LABEL_ATTR):
            raise SystemExit(f"Missing {LABEL_ATTR} in {p.name}")
        y = int(getattr(d, LABEL_ATTR).item())
        if y == 0:
            zeros.append(p)
        elif y == 1:
            ones.append(p)
        else:
            raise SystemExit(f"Invalid {LABEL_ATTR}={y} in {p.name}")

    if len(zeros) < N0 or len(ones) < N1:
        raise SystemExit(f"Not enough samples for balance: zeros={len(zeros)} ones={len(ones)}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    for p in zeros[:N0]:
        shutil.copy2(p, OUT / f"Z__{p.name}")
    for p in ones[:N1]:
        shutil.copy2(p, OUT / f"O__{p.name}")

    print(f"SWEEP_TEST_DONE label_attr={LABEL_ATTR} zeros={N0} ones={N1} total={N0+N1} out={OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
