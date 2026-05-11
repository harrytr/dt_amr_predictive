#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


TRAJECTORIES = [
    "endog_high_train",
    "import_high_train",
    "endog_high_test",
    "import_high_test",
]


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _keep_graphml_enabled() -> bool:
    return str(os.environ.get("DT_KEEP_GRAPHML", "0")).strip() in {"1", "true", "True", "YES", "yes"}


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
        raise RuntimeError(f"convert_to_pt.py failed for {sim_dir} (rc={p.returncode})")


def collect_pt(sim_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = sorted(sim_dir.glob("*.pt"))
    n = 0
    for pt in pts:
        new_name = f"{sim_dir.parent.name}__{sim_dir.name}__{pt.name}"
        shutil.copy2(pt, out_dir / new_name)
        n += 1
    return n


def process_trajectory(trajectory_name: str) -> None:
    traj_dir = Path(trajectory_name)
    if not traj_dir.exists():
        raise RuntimeError(f"Missing trajectory dir: {traj_dir}")

    out_dir = Path(f"{trajectory_name}_pt_flat")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted([p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")])
    if not sim_dirs:
        raise RuntimeError(f"No sim_* folders found in {traj_dir}")

    total_pt = 0
    for sd in sim_dirs:
        run_convert(sd)
        total_pt += collect_pt(sd, out_dir)

    print(f"TRAJECTORY_DONE {trajectory_name}: sims={len(sim_dirs)} pt_files_copied={total_pt} out={out_dir}", flush=True)


def main() -> int:
    for trajectory_name in TRAJECTORIES:
        process_trajectory(trajectory_name)
    print("STEP2_DONE: flat PT folders created for canonical trajectories", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
