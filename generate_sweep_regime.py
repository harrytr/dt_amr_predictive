#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


# 20 sims total, grouped into 5 bands to cross the boundary.
# Values are admission-importation weights for CS/CR/IS/IR.  IS/IR mirror
# CS/CR by default so the sweep changes importation intensity rather than
# silently reverting to a CS/CR-only admission process.
BANDS = [
    # (discharge_frac, p_admit_import_cs, p_admit_import_cr, p_admit_import_is, p_admit_import_ir)
    (0.05, 0.05, 0.05, 0.05, 0.05),  # likely endogenous-ish
    (0.08, 0.08, 0.08, 0.08, 0.08),
    (0.10, 0.12, 0.12, 0.12, 0.12),
    (0.12, 0.18, 0.18, 0.18, 0.18),
    (0.15, 0.25, 0.25, 0.25, 0.25),  # likely import-ish
]

DEFAULT_NUM_DAYS = 60

CANONICAL_SURVEILLANCE_ARGS = [
    "--screen_every_k_days", "7",
    "--screen_result_delay_days", "2",
    "--screen_on_admission", "1",
    "--persist_observations", "1",
]


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def run_one(
    out_dir: Path,
    seed: int,
    discharge_frac: float,
    p_cs: float,
    p_cr: float,
    p_is: float,
    p_ir: float,
) -> int:
    cmd: List[str] = [
        sys.executable,
        "generate_amr_data.py",
        "--output_dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--num_days",
        str(int(DEFAULT_NUM_DAYS)),
        "--daily_discharge_frac",
        str(discharge_frac),
        "--daily_discharge_min_per_ward",
        "1",
        "--p_admit_import_cs",
        str(p_cs),
        "--p_admit_import_cr",
        str(p_cr),
        "--p_admit_import_is",
        str(p_is),
        "--p_admit_import_ir",
        str(p_ir),
        "--export_yaml",
    ]

    cmd += list(CANONICAL_SURVEILLANCE_ARGS)
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")

    print("RUN:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, text=True).returncode)


def main() -> int:
    base = Path("synthetic_endog_import_step7c_sweep")
    base.mkdir(parents=True, exist_ok=True)

    sim_id = 0
    for disc, p_cs, p_cr, p_is, p_ir in BANDS:
        for _ in range(4):  # 5 bands * 4 sims = 20 sims
            out = base / f"sim_{sim_id:03d}"
            out.mkdir(parents=True, exist_ok=True)
            rc = run_one(
                out,
                seed=8000 + sim_id,
                discharge_frac=disc,
                p_cs=p_cs,
                p_cr=p_cr,
                p_is=p_is,
                p_ir=p_ir,
            )
            if rc != 0:
                print(f"FAILED sim_{sim_id:03d} rc={rc}", flush=True)
                return rc
            sim_id += 1

    print("STEP7C_SWEEP_DONE: generated sweep regime sims", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())