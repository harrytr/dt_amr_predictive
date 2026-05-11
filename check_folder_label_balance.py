#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
from pathlib import Path
import torch

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
label_attr = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")
pts = sorted(folder.glob("*.pt"))
y = []
for p in pts:
    d = torch.load(p, weights_only=False)
    if not hasattr(d, label_attr):
        raise SystemExit(f"Missing {label_attr} in {p.name}")
    y.append(int(getattr(d, label_attr).item()))
print("folder=", folder)
print("label_attr=", label_attr)
print("n_files=", len(pts))
print("label0=", sum(v==0 for v in y))
print("label1=", sum(v==1 for v in y))
