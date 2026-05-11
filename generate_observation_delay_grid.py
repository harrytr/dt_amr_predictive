#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


BASE_OUTPUT_DIR = Path("synthetic_endog_import_step6_delay_v2")

# Step 6 is an observation-delay stress test for the predictive task.  The
# default profile now mirrors the canonical Step 1 train split so that Step 6
# changes the observation-delay intervention without also weakening the
# import/endogenous regimes or shortening the trajectories.  The old Step 6
# shifted profile remains available through DT_STEP6_PROFILE=legacy_shifted.
DEFAULT_DELAY_VALUES = [0, 2, 5]
DEFAULT_N_SIMS_PER_REGIME = 10
DEFAULT_STEP6_NUM_DAYS = 60
DEFAULT_STEP6_SCREEN_EVERY_K_DAYS = 7
DEFAULT_STEP6_SCREEN_ON_ADMISSION = 1
DEFAULT_STEP6_PERSIST_OBSERVATIONS = 1
DEFAULT_STEP6_CLEAN = 1

REGIME_PROFILES: Dict[str, Dict[str, Dict[str, float | int]]] = {
    "canonical_baseline": {
        "endog": {
            "daily_discharge_frac": 0.02,
            "daily_discharge_min_per_ward": 0,
            "p_admit_import_cs": 0.005,
            "p_admit_import_cr": 0.005,
            "p_admit_import_is": 0.005,
            "p_admit_import_ir": 0.005,
            "seed_offset": 0,
        },
        "import": {
            "daily_discharge_frac": 0.25,
            "daily_discharge_min_per_ward": 1,
            "p_admit_import_cs": 0.60,
            "p_admit_import_cr": 0.60,
            "p_admit_import_is": 0.60,
            "p_admit_import_ir": 0.60,
            "seed_offset": 50000,
        },
    },
    "legacy_shifted": {
        "endog": {
            "daily_discharge_frac": 0.04,
            "daily_discharge_min_per_ward": 0,
            "p_admit_import_cs": 0.01,
            "p_admit_import_cr": 0.01,
            "p_admit_import_is": 0.01,
            "p_admit_import_ir": 0.01,
            "seed_offset": 0,
        },
        "import": {
            "daily_discharge_frac": 0.12,
            "daily_discharge_min_per_ward": 1,
            "p_admit_import_cs": 0.18,
            "p_admit_import_cr": 0.18,
            "p_admit_import_is": 0.18,
            "p_admit_import_ir": 0.18,
            "seed_offset": 50000,
        },
    },
}


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _env_int(name: str, default: int, *, min_value: int | None = None) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        value = int(default)
    else:
        try:
            value = int(raw)
        except Exception as exc:
            raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc
    if min_value is not None and value < int(min_value):
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_bool_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        return 1 if int(default) else 0
    if raw.lower() in {"1", "true", "yes", "y", "on"}:
        return 1
    if raw.lower() in {"0", "false", "no", "n", "off"}:
        return 0
    raise ValueError(f"Invalid boolean/integer for {name}: {raw!r}")


def _parse_int_list_env(name: str, default: List[int], *, min_value: int | None = None) -> List[int]:
    raw = str(os.environ.get(name, "")).strip()
    if raw == "":
        values = list(default)
    else:
        values = []
        for part in raw.split(","):
            token = part.strip()
            if token == "":
                continue
            try:
                value = int(token)
            except Exception as exc:
                raise ValueError(f"Invalid integer token in {name}: {token!r}") from exc
            values.append(value)
    if not values:
        raise ValueError(f"{name} resolved to an empty list")
    if min_value is not None:
        bad = [v for v in values if int(v) < int(min_value)]
        if bad:
            raise ValueError(f"{name} values must be >= {min_value}; got {bad}")
    return values


def _selected_profile_name() -> str:
    profile = str(os.environ.get("DT_STEP6_PROFILE", "canonical_baseline")).strip() or "canonical_baseline"
    if profile not in REGIME_PROFILES:
        allowed = ", ".join(sorted(REGIME_PROFILES))
        raise ValueError(f"Unknown DT_STEP6_PROFILE={profile!r}. Allowed values: {allowed}")
    return profile


def _selected_regimes() -> Dict[str, Dict[str, float | int]]:
    profile = _selected_profile_name()
    return {name: dict(spec) for name, spec in REGIME_PROFILES[profile].items()}


def _write_manifest(
    *,
    profile: str,
    delay_values: List[int],
    n_sims_per_regime: int,
    num_days: int,
    screen_every_k_days: int,
    screen_on_admission: int,
    persist_observations: int,
    regimes: Dict[str, Dict[str, float | int]],
) -> None:
    payload = {
        "step": "6.1_observation_delay",
        "profile": profile,
        "delay_values": [int(x) for x in delay_values],
        "n_sims_per_regime": int(n_sims_per_regime),
        "num_days": int(num_days),
        "screen_every_k_days": int(screen_every_k_days),
        "screen_on_admission": int(screen_on_admission),
        "persist_observations": int(persist_observations),
        "regimes": regimes,
    }
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_OUTPUT_DIR / "step6_delay_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_one(
    out_dir: Path,
    seed: int,
    delay: int,
    regime_name: str,
    regime: Dict[str, float | int],
    *,
    num_days: int,
    screen_every_k_days: int,
    screen_on_admission: int,
    persist_observations: int,
) -> int:
    cmd: List[str] = [
        sys.executable,
        "generate_amr_data.py",
        "--output_dir",
        str(out_dir),
        "--seed",
        str(int(seed)),
        "--num_days",
        str(int(num_days)),
        "--daily_discharge_frac",
        str(float(regime["daily_discharge_frac"])),
        "--daily_discharge_min_per_ward",
        str(int(regime["daily_discharge_min_per_ward"])),
        "--p_admit_import_cs",
        str(float(regime["p_admit_import_cs"])),
        "--p_admit_import_cr",
        str(float(regime["p_admit_import_cr"])),
        "--p_admit_import_is",
        str(float(regime.get("p_admit_import_is", regime["p_admit_import_cs"]))),
        "--p_admit_import_ir",
        str(float(regime.get("p_admit_import_ir", regime["p_admit_import_cr"]))),
        "--screen_every_k_days",
        str(int(screen_every_k_days)),
        "--screen_on_admission",
        str(int(screen_on_admission)),
        "--screen_result_delay_days",
        str(int(delay)),
        "--persist_observations",
        str(int(persist_observations)),
        "--export_yaml",
    ]

    # Keep global simulator overrides available. These are appended last so an
    # explicit DT_SIM_EXTRA_ARGS value can still override the Step 6 defaults.
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")

    print(
        f"RUN_DELAY regime={regime_name} delay={delay} seed={seed} num_days={num_days}:",
        " ".join(cmd),
        flush=True,
    )
    return int(subprocess.run(cmd, text=True).returncode)


def main() -> int:
    delay_values = _parse_int_list_env("DT_STEP6_DELAY_VALUES", DEFAULT_DELAY_VALUES, min_value=0)
    n_sims_per_regime = _env_int("DT_STEP6_N_SIMS_PER_REGIME", DEFAULT_N_SIMS_PER_REGIME, min_value=1)
    num_days = _env_int("DT_STEP6_NUM_DAYS", DEFAULT_STEP6_NUM_DAYS, min_value=1)
    screen_every_k_days = _env_int("DT_STEP6_SCREEN_EVERY_K_DAYS", DEFAULT_STEP6_SCREEN_EVERY_K_DAYS, min_value=1)
    screen_on_admission = _env_bool_int("DT_STEP6_SCREEN_ON_ADMISSION", DEFAULT_STEP6_SCREEN_ON_ADMISSION)
    persist_observations = _env_bool_int("DT_STEP6_PERSIST_OBSERVATIONS", DEFAULT_STEP6_PERSIST_OBSERVATIONS)
    clean_first = _env_bool_int("DT_STEP6_CLEAN", DEFAULT_STEP6_CLEAN)
    profile = _selected_profile_name()
    regimes = _selected_regimes()

    if clean_first and BASE_OUTPUT_DIR.exists():
        shutil.rmtree(BASE_OUTPUT_DIR)
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_manifest(
        profile=profile,
        delay_values=delay_values,
        n_sims_per_regime=n_sims_per_regime,
        num_days=num_days,
        screen_every_k_days=screen_every_k_days,
        screen_on_admission=screen_on_admission,
        persist_observations=persist_observations,
        regimes=regimes,
    )

    print(
        "STEP6_DELAY_CONFIG "
        f"profile={profile} delays={delay_values} n_sims_per_regime={n_sims_per_regime} "
        f"num_days={num_days} screen_every_k_days={screen_every_k_days} "
        f"screen_on_admission={screen_on_admission} persist_observations={persist_observations} "
        f"clean_first={clean_first}",
        flush=True,
    )

    for delay in delay_values:
        ddir = BASE_OUTPUT_DIR / f"delay_{delay}"
        ddir.mkdir(parents=True, exist_ok=True)

        for regime_name, regime in regimes.items():
            regime_dir = ddir / regime_name
            regime_dir.mkdir(parents=True, exist_ok=True)
            seed_offset = int(regime.get("seed_offset", 0))

            for r in range(n_sims_per_regime):
                out = regime_dir / f"sim_{r:03d}"
                out.mkdir(parents=True, exist_ok=True)
                seed = 9000 + delay * 100 + seed_offset + r
                rc = run_one(
                    out,
                    seed=seed,
                    delay=delay,
                    regime_name=regime_name,
                    regime=regime,
                    num_days=num_days,
                    screen_every_k_days=screen_every_k_days,
                    screen_on_admission=screen_on_admission,
                    persist_observations=persist_observations,
                )
                if rc != 0:
                    print(
                        f"FAILED delay={delay} regime={regime_name} sim={r} rc={rc}",
                        flush=True,
                    )
                    return rc

    print("STEP6A_DONE: generated paired delay grid", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
