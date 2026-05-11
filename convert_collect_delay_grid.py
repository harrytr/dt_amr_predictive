#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE = Path("synthetic_endog_import_step6_delay_v2")
OUT_BASE = Path("synthetic_endog_import_step6_delay_v2_pt_flat")
PREFERRED_LABEL_ATTR = "y_h7_endog_majority"
DEFAULT_MAX_MAJORITY_FRAC = 0.70
DEFAULT_T_LIST = [7]
DEFAULT_WINDOW_BALANCE_CHECK = 1


def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _keep_graphml_enabled() -> bool:
    return str(os.environ.get("DT_KEEP_GRAPHML", "0")).strip() in {"1", "true", "True", "YES", "yes"}


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


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    tag = re.sub(r"_+", "_", tag).strip("_")
    return tag or "item"


def _max_majority_frac_from_env() -> float:
    raw = str(os.environ.get("DT_STEP6_MAX_MAJORITY_FRAC", "")).strip()
    if raw == "":
        return float(DEFAULT_MAX_MAJORITY_FRAC)
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"Invalid DT_STEP6_MAX_MAJORITY_FRAC={raw!r}") from exc
    if not 0.5 <= value <= 1.0:
        raise ValueError("DT_STEP6_MAX_MAJORITY_FRAC must be in [0.5, 1.0]")
    return value


def _t_values_from_env() -> List[int]:
    # The trainer uses labels from the last graph in each temporal window.  The
    # day-level label balance gate can look acceptable even when the actual
    # window targets are collapsed, so Step 6 also checks target labels for T.
    return _parse_int_list_env("DT_STEP6_T_LIST", DEFAULT_T_LIST, min_value=1)


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


def _relative_sim_tag(condition_dir: Path, sim_dir: Path) -> str:
    try:
        rel = sim_dir.relative_to(condition_dir)
    except Exception:
        rel = Path(sim_dir.name)
    return _safe_tag("__".join(rel.parts))


def collect(sim_dir: Path, out_dir: Path, condition_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    sim_tag = _relative_sim_tag(condition_dir, sim_dir)
    for pt in sorted(sim_dir.glob("*.pt")):
        shutil.copy2(pt, out_dir / f"{sim_tag}__{pt.name}")
        n += 1
    return n


def _to_int(x: Any) -> Optional[int]:
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return None


def _read_pt_fields(pt_path: Path, preferred_label_attr: str = PREFERRED_LABEL_ATTR) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    label: Optional[int] = None
    if hasattr(obj, preferred_label_attr):
        value = _to_int(getattr(obj, preferred_label_attr))
        if value in (0, 1):
            label = int(value)

    sim_id_raw = getattr(obj, "sim_id", None)
    sim_id = str(sim_id_raw).strip() if sim_id_raw is not None and str(sim_id_raw).strip() else None
    day = _to_int(getattr(obj, "day", None)) if hasattr(obj, "day") else None
    return label, sim_id, day


def _read_label_from_pt(pt_path: Path, preferred_label_attr: str = PREFERRED_LABEL_ATTR) -> Optional[int]:
    label, _, _ = _read_pt_fields(pt_path, preferred_label_attr=preferred_label_attr)
    return label


def label_stats_for_pt_folder(folder: Path) -> Dict[str, int]:
    n0 = 0
    n1 = 0
    unknown = 0
    for pt_path in sorted(folder.glob("*.pt")):
        label = _read_label_from_pt(pt_path)
        if label == 0:
            n0 += 1
        elif label == 1:
            n1 += 1
        else:
            unknown += 1
    return {"n_total": n0 + n1 + unknown, "n0": n0, "n1": n1, "unknown": unknown}


def window_label_stats_for_pt_folder(folder: Path, *, T: int) -> Dict[str, int]:
    records: Dict[str, List[Tuple[int, int]]] = {}
    unknown = 0
    missing_meta = 0

    for pt_path in sorted(folder.glob("*.pt")):
        label, sim_id, day = _read_pt_fields(pt_path)
        if label not in (0, 1):
            unknown += 1
            continue
        if sim_id is None or day is None:
            missing_meta += 1
            continue
        records.setdefault(str(sim_id), []).append((int(day), int(label)))

    n0 = 0
    n1 = 0
    noncontiguous = 0
    n_sims = 0
    for sim_id, rows in records.items():
        if not rows:
            continue
        n_sims += 1
        rows.sort(key=lambda z: z[0])
        days = [int(d) for d, _ in rows]
        labels = [int(y) for _, y in rows]
        for start in range(0, len(rows) - int(T) + 1):
            d0 = days[start]
            expected_last = d0 + int(T) - 1
            if days[start + int(T) - 1] != expected_last or days[start:start + int(T)] != list(range(d0, expected_last + 1)):
                noncontiguous += 1
                continue
            target = labels[start + int(T) - 1]
            if target == 0:
                n0 += 1
            elif target == 1:
                n1 += 1
            else:
                unknown += 1

    return {
        "n_total": n0 + n1,
        "n0": n0,
        "n1": n1,
        "unknown": unknown,
        "missing_meta": missing_meta,
        "noncontiguous": noncontiguous,
        "n_sims": n_sims,
    }


def _assert_binary_balance(stats: Dict[str, int], condition_name: str, max_majority_frac: float, *, scope: str) -> None:
    n0 = int(stats["n0"])
    n1 = int(stats["n1"])
    known_total = n0 + n1

    if int(stats.get("unknown", 0)) > 0:
        raise RuntimeError(
            f"{condition_name} {scope}: {stats.get('unknown', 0)} files/windows lack a valid {PREFERRED_LABEL_ATTR} label."
        )
    if int(stats.get("missing_meta", 0)) > 0:
        raise RuntimeError(
            f"{condition_name} {scope}: {stats.get('missing_meta', 0)} PT files lack sim_id/day metadata."
        )
    if int(stats.get("noncontiguous", 0)) > 0:
        raise RuntimeError(
            f"{condition_name} {scope}: {stats.get('noncontiguous', 0)} non-contiguous temporal windows were detected."
        )
    if known_total <= 0:
        raise RuntimeError(f"{condition_name} {scope}: no known binary labels found.")
    if n0 <= 0 or n1 <= 0:
        raise RuntimeError(
            f"{condition_name} {scope}: single-label Step 6 dataset: n0={n0}, n1={n1}."
        )

    majority_frac = max(n0, n1) / float(known_total)
    print(
        f"STEP6_LABEL_BALANCE {condition_name} {scope}: "
        f"majority_frac={majority_frac:.4f} max_allowed={max_majority_frac:.4f}",
        flush=True,
    )

    if majority_frac > max_majority_frac:
        raise RuntimeError(
            f"{condition_name} {scope}: Step 6 label imbalance too extreme for {PREFERRED_LABEL_ATTR}: "
            f"n0={n0}, n1={n1}, majority_frac={majority_frac:.4f}, "
            f"max_allowed={max_majority_frac:.4f}."
        )


def assert_step6_label_balance(folder: Path, condition_name: str, max_majority_frac: float, t_values: List[int]) -> Dict[str, int]:
    stats = label_stats_for_pt_folder(folder)
    print(
        f"STEP6_LABEL_STATS {condition_name} day_level: "
        f"n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['unknown']}",
        flush=True,
    )
    _assert_binary_balance(stats, condition_name, max_majority_frac, scope="day_level")

    if _env_bool_int("DT_STEP6_WINDOW_BALANCE_CHECK", DEFAULT_WINDOW_BALANCE_CHECK):
        for t_val in t_values:
            wstats = window_label_stats_for_pt_folder(folder, T=int(t_val))
            print(
                f"STEP6_LABEL_STATS {condition_name} window_T{int(t_val)}: "
                f"n_total={wstats['n_total']} n0={wstats['n0']} n1={wstats['n1']} "
                f"unknown={wstats['unknown']} missing_meta={wstats['missing_meta']} "
                f"noncontiguous={wstats['noncontiguous']} n_sims={wstats['n_sims']}",
                flush=True,
            )
            _assert_binary_balance(wstats, condition_name, max_majority_frac, scope=f"window_T{int(t_val)}")

    return stats


def _sim_dirs_for_condition(condition_dir: Path) -> List[Path]:
    # Supports both the old layout delay_X/sim_000 and the paired layout
    # delay_X/{endog,import}/sim_000.
    return sorted(
        [p for p in condition_dir.rglob("sim_*") if p.is_dir()],
        key=lambda p: p.relative_to(condition_dir).as_posix(),
    )


def main() -> int:
    if OUT_BASE.exists():
        shutil.rmtree(OUT_BASE)
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    condition_dirs = sorted([p for p in BASE.iterdir() if p.is_dir() and p.name.startswith("delay_")])
    total_all = 0
    workers = _workers_from_convert_env()
    max_majority_frac = _max_majority_frac_from_env()
    t_values = _t_values_from_env()
    print(f"DT_CONVERT_WORKERS={workers if workers is not None else 'default'}", flush=True)
    print(f"DT_STEP6_MAX_MAJORITY_FRAC={max_majority_frac:.4f}", flush=True)
    print(f"DT_STEP6_T_LIST={','.join(str(x) for x in t_values)}", flush=True)
    print(f"DT_STEP6_WINDOW_BALANCE_CHECK={_env_bool_int('DT_STEP6_WINDOW_BALANCE_CHECK', DEFAULT_WINDOW_BALANCE_CHECK)}", flush=True)

    for condition_dir in condition_dirs:
        out_dir = OUT_BASE / condition_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        sims = _sim_dirs_for_condition(condition_dir)
        if not sims:
            raise RuntimeError(f"No sim_* directories found under {condition_dir}")

        total = 0
        for sd in sims:
            run_convert(sd)
            total += collect(sd, out_dir, condition_dir=condition_dir)

        assert_step6_label_balance(
            out_dir,
            condition_name=condition_dir.name,
            max_majority_frac=max_majority_frac,
            t_values=t_values,
        )
        total_all += total
        print(f"DELAY_DONE {condition_dir.name}: sims={len(sims)} pt_files={total} out={out_dir}", flush=True)

    print(f"STEP6_1_DONE total_pt={total_all} out_base={OUT_BASE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
