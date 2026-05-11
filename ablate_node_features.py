#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import torch


SRC = Path("synthetic_amr_graphs_train")
DST = Path("step5_ablation") / "core_node_features_only"


def main() -> int:
    if not SRC.exists() or not SRC.is_dir():
        raise SystemExit(f"Source folder not found: {SRC.resolve()}")

    pts = sorted(SRC.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt files found in {SRC.resolve()}")

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in pts:
        d = torch.load(p, map_location="cpu", weights_only=False)

        x = getattr(d, "x", None)
        if x is not None and hasattr(x, "numel") and x.numel() > 0:
            if x.dim() == 2 and x.size(1) >= 6:
                x2 = x.clone()
                x2[:, 2:6] = 0.0
                d.x = x2

        out_path = DST / p.name
        torch.save(d, out_path)
        n += 1

    print(f"ABLATION_DONE files={n} dst={DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
