#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union


STEP1_TRAJECTORIES: Dict[str, Dict[str, Union[float, int]]] = {
    "endog_high_train": {
        "seed_base": 4100,
        "p_cs": 0.005,
        "p_cr": 0.005,
        "discharge_frac": 0.02,
        "discharge_min": 0,
    },
    "import_high_train": {
        "seed_base": 5100,
        "p_cs": 0.6,
        "p_cr": 0.6,
        "discharge_frac": 0.25,
        "discharge_min": 1,
    },
    "endog_high_test": {
        "seed_base": 6100,
        "p_cs": 0.005,
        "p_cr": 0.005,
        "discharge_frac": 0.02,
        "discharge_min": 0,
    },
    "import_high_test": {
        "seed_base": 7100,
        "p_cs": 0.6,
        "p_cr": 0.6,
        "discharge_frac": 0.25,
        "discharge_min": 1,
    },
}


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def run_one(
    out_dir: Path,
    seed: int,
    num_days: int,
    p_cs: float,
    p_cr: float,
    discharge_frac: float,
    discharge_min: int,
) -> int:
    cmd: List[str] = [
        sys.executable,
        "generate_amr_data.py",
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--num_days",
        str(num_days),
        "--daily_discharge_frac",
        str(discharge_frac),
        "--daily_discharge_min_per_ward",
        str(discharge_min),
        "--p_admit_import_cs",
        str(p_cs),
        "--p_admit_import_cr",
        str(p_cr),
        "--export_yaml",
    ]
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")
    print("\nRUN:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True)
    return int(p.returncode)


def main() -> int:
    base = Path(".")
    num_reps = 10
    num_days = 30

    for traj_name, spec in STEP1_TRAJECTORIES.items():
        for r in range(num_reps):
            out = base / traj_name / f"sim_{r:03d}"
            out.mkdir(parents=True, exist_ok=True)
            rc = run_one(
                out,
                seed=int(spec["seed_base"]) + r,
                num_days=num_days,
                p_cs=float(spec["p_cs"]),
                p_cr=float(spec["p_cr"]),
                discharge_frac=float(spec["discharge_frac"]),
                discharge_min=int(spec["discharge_min"]),
            )
            if rc != 0:
                print(f"FAILED trajectory {traj_name} rep {r} (rc={rc})", flush=True)
                return rc

    print("\nSTEP1_DONE: all canonical trajectories completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
