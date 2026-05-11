#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _to_int_maybe(x) -> Optional[int]:
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return None


def _extract_sim_id_and_day(pt_path: Path, obj) -> Tuple[str, Optional[int]]:
    """
    Best-effort extraction of (sim_id, day) from a PyG Data object and/or filename.
    """
    sim_id: Optional[str] = None
    for key in ["sim_id", "simulation_id", "trajectory_id", "traj_id", "sim", "trajectory", "run_id"]:
        if hasattr(obj, key):
            v = getattr(obj, key)
            if isinstance(v, str) and v.strip():
                sim_id = v.strip()
                break

    day: Optional[int] = None
    for key in ["day", "t", "time", "time_idx", "day_idx", "step", "step_idx"]:
        if hasattr(obj, key):
            v = _to_int_maybe(getattr(obj, key))
            if v is not None:
                day = int(v)
                break

    name = pt_path.name

    # Filename fallback for sim_id
    if sim_id is None:
        # Try to capture the common sim prefix before a day token
        m = re.search(
            r"^(sim[^_]*?__[^_]+__[^.]+?)(?:__|_|-)(?:day|d)\d{1,4}\.pt$",
            name,
            flags=re.IGNORECASE,
        )
        if m:
            sim_id = m.group(1)
        else:
            m2 = re.search(r"^(sim[^.]+?)(?:__amr|__|\.pt$)", name, flags=re.IGNORECASE)
            if m2:
                sim_id = m2.group(1)

    if sim_id is None:
        sim_id = f"{pt_path.parent.name}::{pt_path.stem}"

    # Filename fallback for day
    if day is None:
        for pat in [
            r"(?:^|[_-]|__)(?:day)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
            r"(?:^|[_-]|__)(?:d)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
            r"(?:^|[_-]|__)(?:t)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
        ]:
            mm = re.search(pat, name, flags=re.IGNORECASE)
            if mm:
                day = int(mm.group(1))
                break

    return str(sim_id), day


def _read_label_sim_day(pt_path: Path) -> Tuple[int, str, int]:
    """
    Loads the .pt once and returns (label, sim_id, day).
    The label attribute is controlled by DT_LABEL_ATTR and defaults to y_h7_trans_majority.
    """
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    label_attr = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")
    if not hasattr(obj, label_attr):
        raise SystemExit(f"Missing {label_attr} in: {pt_path.name}")
    y = int(getattr(obj, label_attr).item())
    if y not in (0, 1):
        raise SystemExit(f"Invalid label {label_attr}={y} in: {pt_path.name}")

    sim_id, day = _extract_sim_id_and_day(pt_path, obj)
    if day is None:
        raise SystemExit(f"Could not extract day index for: {pt_path.name} (sim_id='{sim_id}')")
    if int(day) <= 0:
        raise SystemExit(f"Invalid day={day} for: {pt_path.name} (sim_id='{sim_id}')")

    return y, sim_id, int(day)


def _copy_clean(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def _build_balanced_noncontiguous(
    *,
    src: Path,
    out: Path,
    cap: Optional[int],
) -> Tuple[int, int]:
    """
    Original behavior: select individual files by label (NOT contiguous-safe).
    """
    pts = sorted(src.glob("*.pt"))
    zeros: List[Path] = []
    ones: List[Path] = []

    for p in pts:
        obj = torch.load(p, map_location="cpu", weights_only=False)
        label_attr = os.environ.get("DT_LABEL_ATTR", "y_h7_trans_majority")
        if not hasattr(obj, label_attr):
            raise SystemExit(f"Missing {label_attr} in: {p.name}")
        y = int(getattr(obj, label_attr).item())
        if y == 0:
            zeros.append(p)
        elif y == 1:
            ones.append(p)
        else:
            raise SystemExit(f"Invalid {label_attr}={y} in: {p.name}")

    n_each = min(len(zeros), len(ones))
    if n_each == 0:
        raise SystemExit(f"Cannot balance: zeros={len(zeros)} ones={len(ones)} in {src}")

    if cap is not None:
        n_each = min(n_each, int(cap))

    _copy_clean(out)

    for p in zeros[:n_each]:
        shutil.copy2(p, out / f"Z__{p.name}")
    for p in ones[:n_each]:
        shutil.copy2(p, out / f"O__{p.name}")

    return n_each, n_each


def _build_balanced_contiguous(
    *,
    src: Path,
    out: Path,
    T_needed: int,
    cap_blocks: Optional[int],
    seed: int,
) -> Tuple[int, int]:
    """
    Contiguous-safe behavior:
      - Candidate = contiguous block of length T_needed within a single sim_id
      - All days in block must share same label
      - Global sim_id uniqueness across labels (no sim_id used in both labels)
      - Copy k blocks per label => (k*T_needed) files per label
    """
    if T_needed <= 1:
        raise SystemExit("--T_needed must be >= 2 for contiguous mode")

    rng = torch.Generator()
    rng.manual_seed(int(seed))

    pts = sorted(src.glob("*.pt"))
    if not pts:
        raise SystemExit(f"No .pt files found in {src}")

    # sim_id -> day -> (path, label)
    sim_map: Dict[str, Dict[int, Tuple[Path, int]]] = {}
    for p in pts:
        y, sim_id, day = _read_label_sim_day(p)
        sim_map.setdefault(sim_id, {})
        if day in sim_map[sim_id]:
            continue
        sim_map[sim_id][day] = (p, y)

    # candidates: (sim_id, start_day, [paths...]) per label
    cand0: List[Tuple[str, int, List[Path]]] = []
    cand1: List[Tuple[str, int, List[Path]]] = []

    for sim_id, day_dict in sim_map.items():
        days = sorted(day_dict.keys())
        if len(days) < T_needed:
            continue
        day_set = set(days)
        for start in days:
            want = list(range(int(start), int(start) + int(T_needed)))
            if any(d not in day_set for d in want):
                continue
            trip = [day_dict[d] for d in want]  # (path, label)
            labs = {int(t[1]) for t in trip}
            if len(labs) != 1:
                continue
            label = int(next(iter(labs)))
            paths = [t[0] for t in trip]
            if label == 0:
                cand0.append((sim_id, int(start), paths))
            else:
                cand1.append((sim_id, int(start), paths))

    if not cand0 or not cand1:
        raise SystemExit(
            f"Cannot build contiguous balanced test: T_needed={T_needed} cand0={len(cand0)} cand1={len(cand1)} in {src}"
        )

    # Shuffle deterministically using torch
    idx0 = torch.randperm(len(cand0), generator=rng).tolist()
    idx1 = torch.randperm(len(cand1), generator=rng).tolist()
    cand0 = [cand0[i] for i in idx0]
    cand1 = [cand1[i] for i in idx1]

    uniq0 = len({x[0] for x in cand0})
    uniq1 = len({x[0] for x in cand1})
    k_max = min(uniq0, uniq1)

    if cap_blocks is not None:
        k_max = min(k_max, int(cap_blocks))

    if k_max <= 0:
        raise SystemExit(
            f"Cannot build contiguous balanced test: no feasible blocks under constraints (uniq0={uniq0}, uniq1={uniq1})"
        )

    # pick label1 first (scarce), then label0 excluding used sims (global uniqueness)
    used_sims = set()
    pick1: List[Tuple[str, int, List[Path]]] = []
    for sim_id, start, paths in cand1:
        if sim_id in used_sims:
            continue
        used_sims.add(sim_id)
        pick1.append((sim_id, start, paths))
        if len(pick1) >= k_max:
            break

    pick0: List[Tuple[str, int, List[Path]]] = []
    for sim_id, start, paths in cand0:
        if sim_id in used_sims:
            continue
        used_sims.add(sim_id)
        pick0.append((sim_id, start, paths))
        if len(pick0) >= k_max:
            break

    if len(pick0) < k_max or len(pick1) < k_max:
        raise SystemExit(
            f"Cannot build contiguous balanced test: k_max={k_max} but got pick0={len(pick0)} pick1={len(pick1)}"
        )

    _copy_clean(out)

    # Copy blocks
    for sim_id, start, paths in pick0:
        for p in paths:
            shutil.copy2(p, out / f"Z__{p.name}")
    for sim_id, start, paths in pick1:
        for p in paths:
            shutil.copy2(p, out / f"O__{p.name}")

    n0 = k_max * T_needed
    n1 = k_max * T_needed
    return n0, n1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=str, help="Source folder containing condition .pt files (delay_k or freq_k).")
    ap.add_argument("--out", type=str, default="synthetic_amr_graphs_test", help="Output test folder path.")
    ap.add_argument(
        "--cap",
        type=int,
        default=1000,
        help="Non-contiguous mode: max files per class. Set <=0 to disable.",
    )
    ap.add_argument(
        "--contiguous",
        action="store_true",
        help="Enable contiguous-by-sim_id balanced construction using blocks of length T_needed.",
    )
    ap.add_argument(
        "--T_needed",
        type=int,
        default=0,
        help="Required block length in contiguous mode (>=2). If 0, defaults to 7.",
    )
    ap.add_argument(
        "--cap_blocks",
        type=int,
        default=0,
        help="Contiguous mode: max blocks per class. Set <=0 to disable.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic shuffle seed for contiguous mode.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.exists() or not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    if args.contiguous:
        T_needed = int(args.T_needed) if int(args.T_needed) > 0 else 7
        cap_blocks = int(args.cap_blocks) if int(args.cap_blocks) > 0 else None
        n0, n1 = _build_balanced_contiguous(
            src=src,
            out=out,
            T_needed=T_needed,
            cap_blocks=cap_blocks,
            seed=int(args.seed),
        )
        print(
            f"OK contiguous_src={src} test_out={out} T_needed={T_needed} "
            f"zeros={n0} ones={n1} total={n0 + n1}"
        )
        return 0

    cap = int(args.cap) if int(args.cap) > 0 else None
    n0, n1 = _build_balanced_noncontiguous(src=src, out=out, cap=cap)
    print(f"OK delay_src={src} test_out={out} zeros={n0} ones={n1} total={n0 + n1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
