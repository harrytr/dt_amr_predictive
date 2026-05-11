#!/usr/bin/env python3
from __future__ import annotations
import os
import torch
from pathlib import Path

folder = Path("synthetic_amr_graphs_test")
label_attr = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")
pts = sorted(folder.glob("*.pt"))
y = []
for p in pts:
    d = torch.load(p, weights_only=False)
    if hasattr(d, label_attr):
        y.append(int(getattr(d, label_attr).item()))
print("label_attr=", label_attr)
print("n_files=", len(pts))
print("label0=", sum(v==0 for v in y))
print("label1=", sum(v==1 for v in y))
