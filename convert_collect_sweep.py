#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


BASE = Path("synthetic_endog_import_step7c_sweep")
OUT = Path("synthetic_endog_import_step7c_sweep_pt_flat")


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _keep_graphml_enabled() -> bool:
    return str(os.environ.get("DT_KEEP_GRAPHML", "0")).strip() in {"1", "true", "True", "YES", "yes"}


def _workers_from_convert_env() -> int | None:
    args = _extra_args_from_env("DT_CONVERT_EXTRA_ARGS")
    for i, tok in enumerate(args):
        if tok == "--workers" and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
            except Exception:
                raise ValueError(f"Invalid --workers value in DT_CONVERT_EXTRA_ARGS: {args[i + 1]!r}")
            if workers < 0:
                raise ValueError(f"--workers must be >= 0, got {workers}")
            return workers
        if tok.startswith("--workers="):
            raw = tok.split("=", 1)[1]
            try:
                workers = int(raw)
            except Exception:
                raise ValueError(f"Invalid --workers value in DT_CONVERT_EXTRA_ARGS: {raw!r}")
            if workers < 0:
                raise ValueError(f"--workers must be >= 0, got {workers}")
            return workers
    return None


def run_convert(sim_dir: Path) -> None:
    cmd: List[str] = [
        sys.executable,
        "convert_to_pt.py",
        "--graphml_dir",
        str(sim_dir),
        "--label_csv_dir",
        str(sim_dir / "labels"),
    ]

    cmd += _extra_args_from_env("DT_CONVERT_EXTRA_ARGS")
    if _keep_graphml_enabled() and "--keep_graphml" not in cmd:
        cmd.append("--keep_graphml")

    print("CONVERT:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"convert_to_pt.py failed for {sim_dir} rc={p.returncode}")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    sims = sorted([p for p in BASE.iterdir() if p.is_dir() and p.name.startswith("sim_")])
    total = 0
    workers = _workers_from_convert_env()
    print(f"DT_CONVERT_WORKERS={workers if workers is not None else 'default'}", flush=True)
    for sd in sims:
        run_convert(sd)
        for pt in sd.glob("*.pt"):
            shutil.copy2(pt, OUT / f"{sd.name}__{pt.name}")
            total += 1

    print(f"STEP7D_SWEEP_DONE total_pt={total} out={OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())