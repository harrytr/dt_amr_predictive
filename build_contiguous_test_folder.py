#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
import torch


SRC = Path("synthetic_endog_import_step7c_sweep_pt_flat")  # sweep dataset
OUT = Path("synthetic_amr_graphs_test")
T = 7
LABEL_ATTR = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")

# how many simulations to include (more sims -> more windows)
N_SIMS = 10


def parse_sim_prefix(fname: str) -> str | None:
    # expects: sim_XXX__<prefix>_t<day>.pt
    m = re.match(r"^(sim_\d+__.+?)_t\d+\.pt$", fname)
    return m.group(1) if m else None


def parse_day(fname: str) -> int | None:
    m = re.match(r"^.+_t(\d+)\.pt$", fname)
    return int(m.group(1)) if m else None


def main() -> int:
    pts = sorted(SRC.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No pt files in {SRC}")

    groups = defaultdict(list)
    for p in pts:
        pref = parse_sim_prefix(p.name)
        day = parse_day(p.name)
        if pref is None or day is None:
            continue
        groups[pref].append((day, p))

    good = []
    for pref, items in groups.items():
        items.sort(key=lambda x: x[0])
        days = [d for d, _ in items]
        ok = False
        for i in range(0, len(days) - T + 1):
            d0 = days[i]
            if all(days[i + k] == d0 + k for k in range(T)):
                ok = True
                break
        if ok:
            good.append((pref, items))

    if len(good) == 0:
        raise SystemExit("No groups with contiguous windows found.")

    chosen = good[:N_SIMS]

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    n_files = 0
    for pref, items in chosen:
        for _, p in items:
            shutil.copy2(p, OUT / p.name)
            n_files += 1

    y = []
    for p in OUT.glob("*.pt"):
        d = torch.load(p, weights_only=False)
        if not hasattr(d, LABEL_ATTR):
            raise SystemExit(f"Missing {LABEL_ATTR} in {p.name}")
        y.append(int(getattr(d, LABEL_ATTR).item()))
    print(f"OK built {n_files} files from {len(chosen)} sims into {OUT}")
    print(f"label_attr={LABEL_ATTR} label0={sum(v==0 for v in y)} label1={sum(v==1 for v in y)} total={len(y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
