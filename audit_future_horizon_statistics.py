#!/usr/bin/env python3
"""
audit_future_horizon_statistics.py
==================================

Read-only audit of the predictive AMR test datasets.

Purpose
-------
For the canonical and shifted experiments, quantify exactly how many model
windows:
  1. have a complete H-day future horizon;
  2. have a truncated future horizon;
  3. are event-free for the mechanism endpoint
     (endogenous events == 0 and imported resistant events == 0);
  4. receive positive/negative endogenous-majority labels;
  5. have stored mechanism labels inconsistent with convert_to_pt.py.

The script reproduces the model's temporal-window construction without using
TemporalGraphDataset, so it does not create or modify node-vocabulary files.
It only reads existing .pt files and writes audit CSV/JSON outputs.

Default datasets
----------------
Canonical test: synthetic_amr_graphs_test_frozen
Shifted test:   synthetic_step8_shift_test

The folders may be directly under --root, under TRACK_ground_truth/work,
under experiments_results/TRACK_ground_truth/work, or anywhere uniquely
beneath --root. Explicit paths can also be supplied.

Example
-------
python audit_future_horizon_statistics.py --root .

or

python audit_future_horizon_statistics.py \
  --canonical-folder /path/to/synthetic_amr_graphs_test_frozen \
  --shifted-folder /path/to/synthetic_step8_shift_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


DEFAULT_CANONICAL_NAME = "synthetic_amr_graphs_test_frozen"
DEFAULT_SHIFTED_NAME = "synthetic_step8_shift_test"
DEFAULT_HISTORY = 7
DEFAULT_HORIZON = 7


@dataclass(frozen=True)
class DayRecord:
    """Metadata and H-day mechanism labels stored on one daily graph."""

    path: Path
    sim_id: str
    day: int
    endog_res: float
    import_res: float
    trans_res: float
    select_res: float
    endog_share: float
    endog_majority: int


@dataclass(frozen=True)
class WindowRecord:
    """One model evaluation window ending on a particular decision day."""

    experiment: str
    sim_id: str
    history_start_day: int
    decision_day: int
    history_days: int
    future_days_available: int
    complete_future: bool
    truncated_future: bool
    endog_res: float
    import_res: float
    trans_res: float
    select_res: float
    event_free: bool
    stored_endog_share: float
    recomputed_endog_share: float
    stored_label: int
    recomputed_label: int
    share_mismatch: bool
    label_mismatch: bool
    endog_component_mismatch: bool
    source_pt: str


def _safe_scalar(value, *, name: str, path: Path) -> float:
    """Convert a scalar-like Python or torch value to float."""
    if value is None:
        raise AttributeError(f"Missing required attribute '{name}' in {path}")

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel"):
        if int(value.numel()) != 1:
            raise ValueError(
                f"Attribute '{name}' is not scalar in {path}: shape={tuple(value.shape)}"
            )
        value = value.reshape(-1)[0].item()

    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert '{name}' to float in {path}: {value!r}") from exc

    if not math.isfinite(out):
        raise ValueError(f"Non-finite value for '{name}' in {path}: {out}")
    return out


def _safe_int(value, *, name: str, path: Path) -> int:
    """Convert a scalar-like Python or torch value to int."""
    x = _safe_scalar(value, name=name, path=path)
    rounded = int(round(x))
    if abs(x - rounded) > 1e-9:
        raise ValueError(f"Expected integer-valued '{name}' in {path}, found {x}")
    return rounded


def _torch_load(path: Path):
    """Load one PyG .pt graph on CPU with compatibility across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _label_attr(horizon: int, suffix: str) -> str:
    return f"y_h{int(horizon)}_{suffix}"


def _read_day_record(path: Path, horizon: int) -> DayRecord:
    data = _torch_load(path)

    sim_id = getattr(data, "sim_id", None)
    day = getattr(data, "day", None)
    if sim_id is None or str(sim_id).strip() == "":
        raise AttributeError(f"Missing required Data.sim_id in {path}")
    if day is None:
        raise AttributeError(f"Missing required Data.day in {path}")

    endog_attr = _label_attr(horizon, "endog_res")
    import_attr = _label_attr(horizon, "import_res")
    trans_attr = _label_attr(horizon, "trans_res")
    select_attr = _label_attr(horizon, "select_res")
    share_attr = _label_attr(horizon, "endog_share")
    majority_attr = _label_attr(horizon, "endog_majority")

    return DayRecord(
        path=path,
        sim_id=str(sim_id),
        day=_safe_int(day, name="day", path=path),
        endog_res=_safe_scalar(getattr(data, endog_attr, None), name=endog_attr, path=path),
        import_res=_safe_scalar(getattr(data, import_attr, None), name=import_attr, path=path),
        trans_res=_safe_scalar(getattr(data, trans_attr, None), name=trans_attr, path=path),
        select_res=_safe_scalar(getattr(data, select_attr, None), name=select_attr, path=path),
        endog_share=_safe_scalar(getattr(data, share_attr, None), name=share_attr, path=path),
        endog_majority=_safe_int(
            getattr(data, majority_attr, None), name=majority_attr, path=path
        ),
    )


def _find_named_dirs(root: Path, name: str) -> List[Path]:
    """Find directories named exactly `name`, avoiding common large/cache trees."""
    matches: List[Path] = []
    skip_parts = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }

    for path in root.rglob(name):
        if not path.is_dir():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        matches.append(path.resolve())
    return sorted(set(matches))


def _resolve_dataset_folder(
    *,
    root: Path,
    explicit: Optional[str],
    default_name: str,
    experiment: str,
) -> Path:
    """Resolve one dataset folder from an explicit path or known repository layouts."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{experiment}: folder not found: {path}")
        return path

    candidates = [
        root / default_name,
        root / "TRACK_ground_truth" / "work" / default_name,
        root / "experiments_results" / "TRACK_ground_truth" / "work" / default_name,
    ]
    existing = [p.resolve() for p in candidates if p.is_dir()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        unique = sorted(set(existing))
        if len(unique) == 1:
            return unique[0]
        raise RuntimeError(
            f"{experiment}: more than one standard-location folder was found:\n  "
            + "\n  ".join(str(p) for p in unique)
            + "\nSupply the intended folder explicitly."
        )

    matches = _find_named_dirs(root, default_name)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"{experiment}: could not find '{default_name}' under {root}. "
            f"Supply --{experiment.replace('_', '-')}-folder explicitly."
        )

    raise RuntimeError(
        f"{experiment}: found multiple '{default_name}' folders under {root}:\n  "
        + "\n  ".join(str(p) for p in matches)
        + "\nSupply the intended folder explicitly."
    )


def _load_dataset(folder: Path, horizon: int) -> Dict[str, List[DayRecord]]:
    """Load required metadata/labels once and group daily graphs by trajectory."""
    pt_files = sorted(folder.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in {folder}")

    groups: Dict[str, List[DayRecord]] = defaultdict(list)
    seen_keys: set[Tuple[str, int]] = set()

    for index, path in enumerate(pt_files, start=1):
        record = _read_day_record(path, horizon)
        key = (record.sim_id, record.day)
        if key in seen_keys:
            raise RuntimeError(
                f"Duplicate trajectory/day in {folder}: sim_id={record.sim_id!r}, day={record.day}"
            )
        seen_keys.add(key)
        groups[record.sim_id].append(record)

        if index % 500 == 0:
            print(f"  loaded {index}/{len(pt_files)} PT files from {folder.name}", flush=True)

    for sim_id in groups:
        groups[sim_id].sort(key=lambda r: r.day)

    return dict(groups)


def _count_consecutive_future_days(
    *,
    day_to_record: Dict[int, DayRecord],
    decision_day: int,
    horizon: int,
) -> int:
    """Match convert_to_pt.py: stop at the first missing next day or after H days."""
    available = 0
    for k in range(1, horizon + 1):
        if decision_day + k not in day_to_record:
            break
        available += 1
    return available


def _build_model_windows(
    *,
    experiment: str,
    groups: Dict[str, List[DayRecord]],
    history: int,
    horizon: int,
    sliding_step: int,
    tolerance: float,
) -> List[WindowRecord]:
    """Reproduce the repository's contiguous T-day sliding model windows."""
    if history <= 0:
        raise ValueError("history must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if sliding_step <= 0:
        raise ValueError("sliding_step must be positive")

    windows: List[WindowRecord] = []

    for sim_id, seq in sorted(groups.items()):
        day_to_record = {record.day: record for record in seq}

        for start in range(0, len(seq) - history + 1, sliding_step):
            history_seq = seq[start : start + history]
            d0 = history_seq[0].day
            expected_days = list(range(d0, d0 + history))
            actual_days = [record.day for record in history_seq]

            # TemporalGraphDataset with fail_on_noncontiguous=True would fail here.
            if actual_days != expected_days:
                raise RuntimeError(
                    f"Non-contiguous history window in experiment={experiment}, "
                    f"sim_id={sim_id!r}: expected={expected_days}, got={actual_days}"
                )

            last = history_seq[-1]
            future_days = _count_consecutive_future_days(
                day_to_record=day_to_record,
                decision_day=last.day,
                horizon=horizon,
            )
            complete = future_days == horizon

            denom = last.endog_res + last.import_res
            recomputed_share = 0.0 if denom <= 0.0 else last.endog_res / denom
            recomputed_label = 1 if recomputed_share >= 0.5 else 0

            event_free = abs(last.endog_res) <= tolerance and abs(last.import_res) <= tolerance
            share_mismatch = abs(last.endog_share - recomputed_share) > tolerance
            label_mismatch = int(last.endog_majority) != int(recomputed_label)
            component_mismatch = abs(last.endog_res - (last.trans_res + last.select_res)) > tolerance

            windows.append(
                WindowRecord(
                    experiment=experiment,
                    sim_id=sim_id,
                    history_start_day=history_seq[0].day,
                    decision_day=last.day,
                    history_days=history,
                    future_days_available=future_days,
                    complete_future=complete,
                    truncated_future=not complete,
                    endog_res=last.endog_res,
                    import_res=last.import_res,
                    trans_res=last.trans_res,
                    select_res=last.select_res,
                    event_free=event_free,
                    stored_endog_share=last.endog_share,
                    recomputed_endog_share=recomputed_share,
                    stored_label=int(last.endog_majority),
                    recomputed_label=int(recomputed_label),
                    share_mismatch=share_mismatch,
                    label_mismatch=label_mismatch,
                    endog_component_mismatch=component_mismatch,
                    source_pt=last.path.name,
                )
            )

    return windows


def _pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else 0.0


def _summarize(
    *,
    experiment: str,
    folder: Path,
    groups: Dict[str, List[DayRecord]],
    windows: Sequence[WindowRecord],
    history: int,
    horizon: int,
) -> Dict[str, object]:
    total = len(windows)
    complete = sum(w.complete_future for w in windows)
    truncated = total - complete

    event_free_all = sum(w.event_free for w in windows)
    event_free_complete = sum(w.event_free and w.complete_future for w in windows)
    event_free_truncated = sum(w.event_free and w.truncated_future for w in windows)

    positive_all = sum(w.stored_label == 1 for w in windows)
    positive_complete = sum(w.stored_label == 1 and w.complete_future for w in windows)
    positive_truncated = sum(w.stored_label == 1 and w.truncated_future for w in windows)

    negative_all = total - positive_all
    negative_complete = complete - positive_complete
    negative_truncated = truncated - positive_truncated

    share_mismatches = sum(w.share_mismatch for w in windows)
    label_mismatches = sum(w.label_mismatch for w in windows)
    component_mismatches = sum(w.endog_component_mismatch for w in windows)

    return {
        "experiment": experiment,
        "folder": str(folder),
        "trajectories": len(groups),
        "daily_pt_graphs": sum(len(seq) for seq in groups.values()),
        "history_T_days": history,
        "forecast_H_days": horizon,
        "model_windows_total": total,
        "complete_future_windows": complete,
        "complete_future_pct": _pct(complete, total),
        "truncated_future_windows": truncated,
        "truncated_future_pct": _pct(truncated, total),
        "event_free_all": event_free_all,
        "event_free_all_pct": _pct(event_free_all, total),
        "event_free_complete": event_free_complete,
        "event_free_complete_pct": _pct(event_free_complete, complete),
        "event_free_truncated": event_free_truncated,
        "event_free_truncated_pct": _pct(event_free_truncated, truncated),
        "positive_endogenous_label_all": positive_all,
        "negative_importation_label_all": negative_all,
        "positive_endogenous_label_complete": positive_complete,
        "negative_importation_label_complete": negative_complete,
        "positive_endogenous_label_truncated": positive_truncated,
        "negative_importation_label_truncated": negative_truncated,
        "stored_share_mismatches": share_mismatches,
        "stored_label_mismatches": label_mismatches,
        "endog_equals_trans_plus_select_mismatches": component_mismatches,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _window_to_row(window: WindowRecord) -> Dict[str, object]:
    return {
        "experiment": window.experiment,
        "sim_id": window.sim_id,
        "history_start_day": window.history_start_day,
        "decision_day": window.decision_day,
        "history_days": window.history_days,
        "future_days_available": window.future_days_available,
        "complete_future": int(window.complete_future),
        "truncated_future": int(window.truncated_future),
        "endog_res": window.endog_res,
        "import_res": window.import_res,
        "trans_res": window.trans_res,
        "select_res": window.select_res,
        "event_free": int(window.event_free),
        "stored_endog_share": window.stored_endog_share,
        "recomputed_endog_share": window.recomputed_endog_share,
        "stored_label": window.stored_label,
        "recomputed_label": window.recomputed_label,
        "share_mismatch": int(window.share_mismatch),
        "label_mismatch": int(window.label_mismatch),
        "endog_component_mismatch": int(window.endog_component_mismatch),
        "source_pt": window.source_pt,
    }


def _distribution_rows(
    *,
    experiment: str,
    windows: Sequence[WindowRecord],
    horizon: int,
) -> List[Dict[str, object]]:
    counts = Counter(w.future_days_available for w in windows)
    event_free_counts = Counter(
        w.future_days_available for w in windows if w.event_free
    )
    positive_counts = Counter(
        w.future_days_available for w in windows if w.stored_label == 1
    )

    rows: List[Dict[str, object]] = []
    for future_days in range(0, horizon + 1):
        n = counts.get(future_days, 0)
        ef = event_free_counts.get(future_days, 0)
        pos = positive_counts.get(future_days, 0)
        rows.append(
            {
                "experiment": experiment,
                "future_days_available": future_days,
                "windows": n,
                "windows_pct": _pct(n, len(windows)),
                "event_free_windows": ef,
                "event_free_pct_within_future_length": _pct(ef, n),
                "positive_endogenous_labels": pos,
                "negative_importation_labels": n - pos,
            }
        )
    return rows


def _print_summary(summary: Dict[str, object]) -> None:
    print(f"\n=== {summary['experiment']} ===")
    print(f"folder: {summary['folder']}")
    print(
        "model windows: "
        f"{summary['model_windows_total']} | "
        f"complete H={summary['forecast_H_days']}: {summary['complete_future_windows']} "
        f"({summary['complete_future_pct']:.2f}%) | "
        f"truncated: {summary['truncated_future_windows']} "
        f"({summary['truncated_future_pct']:.2f}%)"
    )
    print(
        "event-free: "
        f"all={summary['event_free_all']} ({summary['event_free_all_pct']:.2f}%) | "
        f"complete={summary['event_free_complete']} "
        f"({summary['event_free_complete_pct']:.2f}% of complete) | "
        f"truncated={summary['event_free_truncated']} "
        f"({summary['event_free_truncated_pct']:.2f}% of truncated)"
    )
    print(
        "label consistency mismatches: "
        f"share={summary['stored_share_mismatches']} | "
        f"majority={summary['stored_label_mismatches']} | "
        f"endog=(trans+select)={summary['endog_equals_trans_plus_select_mismatches']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit complete/truncated H-day futures and event-free mechanism labels "
            "for the canonical and shifted predictive test datasets."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root under which the standard test folders can be found (default: current directory).",
    )
    parser.add_argument(
        "--canonical-folder",
        type=str,
        default=None,
        help=f"Explicit canonical test folder (default name: {DEFAULT_CANONICAL_NAME}).",
    )
    parser.add_argument(
        "--shifted-folder",
        type=str,
        default=None,
        help=f"Explicit shifted test folder (default name: {DEFAULT_SHIFTED_NAME}).",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=DEFAULT_HISTORY,
        help=f"Temporal history length T used by the model (default: {DEFAULT_HISTORY}).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"Future mechanism horizon H (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--sliding-step",
        type=int,
        default=1,
        help="Sliding-window step, matching model evaluation (default: 1).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="future_horizon_audit",
        help="Directory for CSV/JSON outputs (default: future_horizon_audit).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Floating-point comparison tolerance (default: 1e-6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    history = int(args.history)
    horizon = int(args.horizon)
    sliding_step = int(args.sliding_step)
    tolerance = float(args.tolerance)

    if history <= 0 or horizon <= 0 or sliding_step <= 0:
        raise ValueError("--history, --horizon and --sliding-step must all be positive integers")
    if tolerance < 0.0:
        raise ValueError("--tolerance must be non-negative")

    canonical_folder = _resolve_dataset_folder(
        root=root,
        explicit=args.canonical_folder,
        default_name=DEFAULT_CANONICAL_NAME,
        experiment="canonical",
    )
    shifted_folder = _resolve_dataset_folder(
        root=root,
        explicit=args.shifted_folder,
        default_name=DEFAULT_SHIFTED_NAME,
        experiment="shifted",
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, object]] = []
    all_windows: List[WindowRecord] = []
    all_distribution_rows: List[Dict[str, object]] = []

    for experiment, folder in (
        ("canonical_test", canonical_folder),
        ("shifted_test", shifted_folder),
    ):
        print(f"\nLoading {experiment}: {folder}", flush=True)
        groups = _load_dataset(folder, horizon)
        windows = _build_model_windows(
            experiment=experiment,
            groups=groups,
            history=history,
            horizon=horizon,
            sliding_step=sliding_step,
            tolerance=tolerance,
        )
        if not windows:
            raise RuntimeError(
                f"{experiment}: no model windows were produced with T={history}, "
                f"sliding_step={sliding_step}"
            )

        summary = _summarize(
            experiment=experiment,
            folder=folder,
            groups=groups,
            windows=windows,
            history=history,
            horizon=horizon,
        )
        all_summaries.append(summary)
        all_windows.extend(windows)
        all_distribution_rows.extend(
            _distribution_rows(experiment=experiment, windows=windows, horizon=horizon)
        )
        _print_summary(summary)

    summary_fields = list(all_summaries[0].keys())
    _write_csv(out_dir / "future_horizon_summary.csv", all_summaries, summary_fields)

    window_rows = [_window_to_row(window) for window in all_windows]
    window_fields = list(window_rows[0].keys())
    _write_csv(out_dir / "future_horizon_windows.csv", window_rows, window_fields)

    distribution_fields = list(all_distribution_rows[0].keys())
    _write_csv(
        out_dir / "future_horizon_distribution.csv",
        all_distribution_rows,
        distribution_fields,
    )

    json_payload = {
        "audit_definition": {
            "history_T_days": history,
            "forecast_H_days": horizon,
            "sliding_step": sliding_step,
            "event_free_definition": "stored H-day endogenous events == 0 and imported resistant events == 0",
            "complete_future_definition": "H consecutive future days exist after the final day of the T-day model history window",
            "truncated_future_definition": "fewer than H consecutive future days exist after the final day of the T-day model history window",
            "label_convention": "endogenous_share=0 when endogenous+imported events=0; majority=1 when share>=0.5, else 0",
            "read_only": True,
        },
        "summaries": all_summaries,
    }
    with (out_dir / "future_horizon_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2, sort_keys=True)

    mismatch_total = sum(
        int(summary["stored_share_mismatches"])
        + int(summary["stored_label_mismatches"])
        + int(summary["endog_equals_trans_plus_select_mismatches"])
        for summary in all_summaries
    )

    print(f"\nWrote audit outputs to: {out_dir}")
    print("  future_horizon_summary.csv")
    print("  future_horizon_distribution.csv")
    print("  future_horizon_windows.csv")
    print("  future_horizon_summary.json")

    if mismatch_total > 0:
        raise SystemExit(
            "Audit completed, but one or more stored mechanism labels were internally inconsistent. "
            "See the exported CSV files."
        )

    print("Audit completed with no stored-label consistency mismatches.")


if __name__ == "__main__":
    main()
