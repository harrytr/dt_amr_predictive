#!/usr/bin/env python3
"""Report original Steps 4--9 with conditional CIs for ALL five metrics.

Read-only inputs: original metrics/predictions, optional archived graph metadata,
and original XGBoost results. No trainer, simulator, model inference, data
regeneration, label alteration, or manuscript editing is invoked.

Run from addons/, not the repository's top-level Python inventory.
Version 1.1.0 adds metadata-only recovery for deleted Step 5--7 flat folders.
Use a NEW output directory because the runner/source identity has changed.
See README_original_predictive_ci.txt for commands and recovery rules.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Never create bytecode caches in the supplied repository or result directories.
sys.dont_write_bytecode = True
VERSION = "1.1.0"
TASK = "endogenous_importation_majority_h7"
METRICS = ("auroc", "accuracy", "macro_f1", "sensitivity", "specificity")
TRACKS = ("ground_truth", "partial_observation")
SPLITS = ("validation", "test")
OWNER = "original_predictive_ci_manifest.json"
HELPERS = ("experiments_pbc7.py", "predictive_repeats.py", "summarize_predictive_repeats.py")
STEP_LABELS = {
    "step4_baseline": "Step 4: canonical",
    "step5_core_node_features_only": "Step 5: core node features",
    "step5_no_edge_weights": "Step 5: no edge weights",
    "step6_delay_0": "Step 6: delay 0 days",
    "step6_delay_2": "Step 6: delay 2 days",
    "step6_delay_5": "Step 6: delay 5 days",
    "step6_freq_3": "Step 6: screen every 3 days",
    "step6_freq_7": "Step 6: screen every 7 days",
    "step6_freq_14": "Step 6: screen every 14 days",
    "step7_sweep": "Step 7: regime sweep",
    "step8_distribution_shift": "Step 8: shifted train/test",
    "step9_baseline_to_shift_test": "Step 9: zero-shot",
}
MODEL_LABELS = {"GNN": "GNN + Transformer", "XGB_all": "XGB: all summaries",
                "XGB_no_counts": "XGB: without counts"}
XGB_ROWS = (
    ("step4_baseline", "XGB_all", "step4_baseline"),
    ("step4_baseline", "XGB_no_counts", "step5_no_admission_turnover"),
    ("step5_no_edge_weights", "XGB_all", "step5_no_edge_weights"),
    ("step8_distribution_shift", "XGB_all", "step8_distribution_shift"),
    ("step8_distribution_shift", "XGB_no_counts", "step8_no_admission_turnover"),
    ("step9_baseline_to_shift_test", "XGB_all", "step9_baseline_to_shift_test"),
    ("step9_baseline_to_shift_test", "XGB_no_counts", "step9_no_admission_turnover"),
)

# Recovery is enabled only for the reviewed copying/naming contracts. These
# fingerprints ignore whitespace/comments and unrelated helper functions. No
# collector, ablation, simulator or trainer main() is executed during recovery.
RECOVERY_CONTRACTS = {
    'convert_collect_delay_grid.py': (
        ('BASE', 'OUT_BASE', '_safe_tag', '_relative_sim_tag', 'collect', '_sim_dirs_for_condition'),
        'b795375ef896200e0be9f2a84653ad1bae1da4f746385224c5856a17f4e9a34f'),
    'convert_collect_freq_grid.py': (
        ('BASE', 'OUT_BASE', '_safe_tag', '_relative_sim_tag', 'collect', '_sim_dirs_for_condition'),
        'b0cedd090903a48f4ce78332a3fbc247e04e48e7aa68aa78546d6cdab4398852'),
    'convert_collect_sweep.py': (
        ('BASE', 'OUT', 'main'),
        '9ccf4ef89f85af1d39f44cb17428d15f60ba69c6591b12a03aabd9f6f5f2d812'),
    'ablate_edge_weights.py': (
        ('SRC', 'DST', 'main'),
        'fcbff306b8b7c9ba54b7e3abfa83b1392cd268cb535633b77241e7df077ff503'),
    'ablate_node_features.py': (
        ('SRC', 'DST', 'main'),
        '88d979639578efe1166d5be026fe310d6d6b73989f6880e52ad9448537dfa192'),
}


def recovery_ast_signature(value: Any) -> Any:
    """Portable AST signature, independent of line numbers and empty type_params."""
    if isinstance(value, ast.AST):
        return [type(value).__name__, [[key, recovery_ast_signature(item)]
                for key, item in ast.iter_fields(value) if key != "type_params"]]
    if isinstance(value, list):
        return [recovery_ast_signature(item) for item in value]
    return value


class EvidenceError(RuntimeError):
    """Saved evidence is missing, ambiguous, incompatible, or inconsistent."""


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True,
                             allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def natural_key(value: Any) -> List[Any]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", str(value))]


def canonical_tag(tag: str) -> Optional[str]:
    if tag in STEP_LABELS:
        return tag
    # The driver appends the generated condition-directory name to these tags.
    for prefix in ("step6_delay_", "step6_freq_"):
        if tag.startswith(prefix) and "tuning" not in tag:
            m = re.search(r"\d+", tag[len(prefix):])
            if m:
                return prefix + str(int(m.group()))
    return None


def import_source(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"Cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers(repo: Path) -> Tuple[Any, Any]:
    for name in HELPERS:
        if not (repo / name).is_file():
            raise EvidenceError(f"Required repository helper is missing: {repo / name}")
    # Only imports reporting/configuration helpers. No main() function is called.
    sys.path.insert(0, str(repo))
    driver = import_source(repo / "experiments_pbc7.py", "_original_ci_driver")
    import_source(repo / "predictive_repeats.py", "predictive_repeats")
    stats = import_source(repo / "summarize_predictive_repeats.py", "_original_ci_stats")
    if tuple(stats.METRICS) != METRICS:
        raise EvidenceError("The repository's metric list has changed; do not silently change the table.")
    if not all(callable(getattr(stats, k, None)) for k in ("binary_metrics", "trajectory_bootstrap")):
        raise EvidenceError("Required statistical functions are unavailable.")
    return driver, stats


def protect_inputs(repo: Path, inputs: Sequence[Path], output: Path) -> None:
    """Block Python filesystem mutations of inputs; not an OS security sandbox."""
    inputs = [p.resolve() for p in inputs]

    def forbidden(raw: Any) -> bool:
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return False  # Standard-output and other already-open descriptors.
        path = Path(os.fsdecode(raw))
        return any(inside(path, r) for r in inputs) or (
            inside(path, repo) and not inside(path, output))

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        paths: Sequence[Any] = ()
        if event == "open":
            mode, flags = args[1], args[2]
            writes = isinstance(mode, str) and any(x in mode for x in "wax+")
            writes = writes or (isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            paths = args[:1] if writes else ()
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "shutil.rmtree"):
            paths = args[:1]
        elif event in ("os.rename", "os.replace"):
            paths = args[:2]
        elif event in ("os.symlink", "os.link"):
            paths = args[1:2]
        if any(forbidden(p) for p in paths):
            raise PermissionError(f"Read-only input protection: {event}: {paths}")
    sys.addaudithook(audit)


def validate_paths(args: argparse.Namespace) -> None:
    repo, data, out = args.repo, args.data_root, args.out_root
    if not repo.is_dir() or not data.is_dir():
        raise EvidenceError("--repo and --data-root must be existing directories.")
    if (data / "run_manifest.json").exists() or any(
        re.fullmatch(r"repeat_\d+", p.name) or p.name == "predictive_repeat_results"
        for p in (data, *data.parents)
    ):
        raise EvidenceError("Use ORIGINAL experiments_results, not a five-repeat results directory.")
    if args.xgboost_root is not None and not args.xgboost_root.is_dir():
        raise EvidenceError(f"XGBoost output folder is missing: {args.xgboost_root}")
    inputs = [data] + ([args.xgboost_root] if args.xgboost_root is not None else [])
    if out == repo or inside(repo, out) or any(inside(out, p) or inside(p, out) for p in inputs):
        raise EvidenceError("Choose a separate output tree, not an input tree or its ancestor.")
    if not out.parent.is_dir():
        raise EvidenceError(f"Output parent does not exist: {out.parent}")
    if out.is_symlink():
        raise EvidenceError("Output directory must not be a symbolic link.")
    if out.exists():
        if not out.is_dir():
            raise EvidenceError("Output path is not a directory.")
        if any(out.iterdir()) and not (out / OWNER).is_file():
            raise EvidenceError("Refusing a nonempty, unowned output folder. Choose a new --out-root.")
        for path in out.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
                raise EvidenceError(f"Refusing linked output artifact: {path}")


@contextlib.contextmanager
def output_lock(root: Path):
    import fcntl
    root.mkdir(exist_ok=True)
    with (root / ".original_ci.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise EvidenceError(f"Another reporting process is using {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def relevant_summary(path: Path) -> bool:
    if any((re.fullmatch(r"T\d+", p) and p != "T7") or
           (re.fullmatch(r"h\d+", p) and p != "h7") for p in path.parts):
        return False
    value = read_json(path)
    cfg = value.get("config", {})
    return value.get("task") == TASK and int(cfg.get("T", -1)) == 7


def discover_tasks(args: argparse.Namespace, overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
    tasks = []
    for track in args.tracks:
        archive = args.data_root / f"TRACK_{track}" / "work" / "repro_artifacts_steps_1_9"
        found: Dict[str, List[Path]] = {}
        if archive.is_dir():
            for path in sorted(archive.rglob("metrics_summary.json")):
                rel = path.relative_to(archive)
                if not rel.parts:
                    continue
                tag = canonical_tag(rel.parts[0])
                if tag is not None and relevant_summary(path):
                    found.setdefault(tag, []).append(path)
        tags = list(STEP_LABELS) + sorted(set(found) - set(STEP_LABELS), key=natural_key)
        for tag in tags:
            key = f"{track}/{tag}"
            chosen = overrides.get(key, {}).get("summary")
            candidates = [Path(chosen).expanduser().resolve()] if chosen else found.get(tag, [])
            error = None
            if not candidates:
                error = f"No original T=7/H=7 metrics_summary.json for {key} under {archive}"
            elif len(candidates) > 1:
                # Exact duplicates are harmless; different fits must not be averaged or cherry-picked.
                hashes = {file_hash(p) for p in candidates}
                if len(hashes) != 1:
                    error = f"Multiple DIFFERENT fitted-model summaries for {key}: {[str(p) for p in candidates]}. Select one using --source-map."
            path = candidates[0] if candidates else None
            if path is not None and (not path.is_file() or not inside(path, args.data_root)):
                error = f"Selected summary is absent or outside --data-root: {path}"
            tasks.append({"track": track, "step": tag, "model": "GNN", "key": key + "/GNN",
                          "summary": str(path) if path else None, "discovery_error": error})
    if args.xgboost_root is not None and "ground_truth" in args.tracks:
        for step, model, tag in XGB_ROWS:
            folder = args.xgboost_root / "03_experiments" / "ground_truth" / tag
            path = folder / "metrics.json"
            alias = None
            # Edge-weight removal leaves every XGBoost input unchanged. Reuse only this documented alias.
            if not path.is_file() and step == "step5_no_edge_weights":
                path = args.xgboost_root / "03_experiments" / "ground_truth" / "step4_baseline" / "metrics.json"
                alias = "step4_baseline (unchanged XGBoost inputs)"
            tasks.append({"track": "ground_truth", "step": step, "model": model,
                          "key": f"ground_truth/{step}/{model}", "summary": str(path),
                          "xgb_tag": tag, "alias": alias,
                          "discovery_error": None if path.is_file() else f"Missing original XGBoost output: {path}"})
    return sorted(tasks, key=lambda t: (TRACKS.index(t["track"]), step_order(t["step"]),
                   ("GNN", "XGB_all", "XGB_no_counts").index(t["model"])))


def step_order(tag: str) -> Tuple[float, int, str]:
    if tag in STEP_LABELS:
        return float(list(STEP_LABELS).index(tag)), 0, tag
    return 99.0, 0, tag


def normalize_rows(raw: Sequence[Dict[str, Any]], probabilities: bool = True) -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for item in raw:
        try:
            sid = str(item["sim_id"])
            day, start, y = int(item["day"]), int(item["start_day"]), int(item["y"])
        except (KeyError, ValueError, TypeError) as exc:
            raise EvidenceError("Prediction/metadata rows need sim_id, day, start_day and binary y.") from exc
        if not sid.strip() or y not in (0, 1) or day - start != 6 or (sid, day) in seen:
            raise EvidenceError(f"Invalid or duplicate seven-day window: {sid}/{day}")
        seen.add((sid, day))
        row = {"sim_id": sid, "day": day, "start_day": start, "y": y}
        if item.get("last_file"):
            row["last_file"] = Path(str(item["last_file"])).name
        if probabilities:
            try:
                p = float(item.get("probability", item.get("probability_class_1")))
                pred = int(item["prediction"])
            except (KeyError, ValueError, TypeError) as exc:
                raise EvidenceError("Each prediction row needs a probability and a saved prediction.") from exc
            if not math.isfinite(p) or not 0 <= p <= 1 or pred != int(p > 0.5):
                raise EvidenceError("Saved predictions violate the fixed >0.5 rule; no threshold is changed.")
            row.update(probability=p, prediction=pred)
        rows.append(row)
    if not rows:
        raise EvidenceError("Empty prediction/metadata set.")
    return rows


def verify_points(point: Dict[str, Any], block: Dict[str, Any], label: str, gnn: bool) -> None:
    import numpy as np
    saved = block.get("metrics", {}) if gnn else block
    aliases = {"auroc": "roc_auc", "macro_f1": "f1_macro"} if gnn else {}
    cm = block.get("confusion_matrix") if gnn else None
    if cm is not None:
        expected = [[point["tn"], point["fp"]], [point["fn"], point["tp"]]]
        if not np.array_equal(np.asarray(cm), np.asarray(expected)):
            raise EvidenceError(f"{label}: saved confusion counts disagree with the predictions.")
    for metric in METRICS:
        key = aliases.get(metric, metric)
        value = saved.get(key)
        if key not in saved:
            if metric in ("sensitivity", "specificity") and cm is not None:
                # The confusion matrix has just been checked; historical JSONs lacked these keys.
                continue
            if metric == "auroc" and point[metric] is None:
                continue
            raise EvidenceError(f"{label}: no recorded {key} to verify against.")
        a, b = value, point[metric]
        if a is None or (isinstance(a, float) and not math.isfinite(a)):
            if b is not None:
                raise EvidenceError(f"{label}: {metric} was undefined but recalculated as {b}.")
        elif b is None or not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9):
            raise EvidenceError(f"{label}: {metric} changed: saved={a}, recalculated={b}.")
    for key, other in (("n_samples", "n_windows"), ("n", "n_windows"), ("positive", "positive"),
                       ("tn", "tn"), ("fp", "fp"), ("fn", "fn"), ("tp", "tp")):
        if key in block and int(block[key]) != point[other]:
            raise EvidenceError(f"{label}: saved {key} disagrees with the predictions.")


def verify_arrays(rows: List[Dict[str, Any]], block: Dict[str, Any], label: str) -> None:
    """Verify original CSV order against stored JSON; labels alone are insufficient."""
    import numpy as np
    if "y_true" not in block:
        return
    y, pred = [r["y"] for r in rows], [r["prediction"] for r in rows]
    if y != block["y_true"] or pred != block.get("y_pred"):
        raise EvidenceError(f"{label}: CSV and archived JSON labels/predictions/order disagree.")
    probs = np.asarray(block.get("probabilities"), dtype=float)
    if probs.shape != (len(rows), 2) or not np.isfinite(probs).all():
        raise EvidenceError(f"{label}: missing/invalid saved two-class probabilities.")
    if not np.allclose(probs[:, 1], [r["probability"] for r in rows], rtol=1e-7, atol=1e-9):
        raise EvidenceError(f"{label}: CSV and JSON probabilities/order disagree.")
    if not np.array_equal(probs.argmax(axis=1), np.asarray(pred)):
        raise EvidenceError(f"{label}: original probability argmax disagrees with its saved decisions.")


class Reader:
    """Resolve only saved evidence; cache metadata once per dataset and file inventory."""

    def __init__(self, args: argparse.Namespace, driver: Any, stats: Any, overrides: Dict[str, Any]):
        self.args, self.driver, self.stats, self.overrides = args, driver, stats, overrides
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        self.evidence: Dict[str, Dict[str, Any]] = {}
        self.cache_memory: Dict[str, Any] = {}
        self.recovery_contracts: Dict[str, Dict[str, Any]] = {}
        self.recovery_notes: Dict[str, Dict[str, Any]] = {}

    def record(self, path: Path) -> None:
        stat = path.stat()
        token = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        prior = self.evidence.get(str(path))
        if prior and any(prior[k] != token[k] for k in token):
            raise EvidenceError(f"Source file changed while reading: {path}")
        if prior is None:
            self.evidence[str(path)] = {"path": str(path), **token, "sha256": file_hash(path)}

    def json(self, path: Path) -> Any:
        self.record(path)
        return read_json(path)

    def csv(self, path: Path) -> List[Dict[str, str]]:
        self.record(path)
        return read_csv(path)

    def rebase(self, raw: Any, track: str) -> Optional[Path]:
        if not raw:
            return None
        path = Path(str(raw)).expanduser()
        if path.is_absolute() and inside(path, self.args.data_root):
            return path.resolve()
        anchor = f"TRACK_{track}"
        if anchor in path.parts:
            return self.args.data_root.joinpath(*path.parts[path.parts.index(anchor):]).resolve()
        if not path.is_absolute():
            return (self.args.data_root / anchor / "work" / path).resolve()
        return None

    def dataset_candidate(self, task: Dict[str, Any], cfg: Dict[str, Any], kind: str) -> Tuple[Path, bool]:
        track, step = task["track"], task["step"]
        work = self.args.data_root / f"TRACK_{track}" / "work"
        archive = work / "repro_artifacts_steps_1_9"
        summary = Path(task["summary"])
        override = self.overrides.get(f"{track}/{step}", {})
        original = self.rebase(cfg.get("data_folder" if kind == "train" else "test_folder"), track)
        candidates: List[Path] = []
        explicit = override.get(f"{kind}_dataset")
        if explicit:
            candidates = [Path(explicit).expanduser().resolve()]
        else:
            candidates.append(summary.parent / f"{kind}_folder")
            if original is not None:
                candidates.append(original)
            try:
                raw_step = summary.relative_to(archive).parts[0]
            except ValueError:
                raw_step = step
            candidates += sorted((archive / raw_step).rglob(f"{kind}_folder"))
            if kind == "test":
                # These are the saved fixed tests the ORIGINAL driver used.
                if step.startswith(("step8_", "step9_")):
                    candidates.append(work / self.driver.STEP8_TEST_REL)
                    fallback = ("step8_distribution_shift", "step9_baseline_to_shift_test")
                else:
                    candidates += [work / self.driver.BASELINE_TEST_REL, work / self.driver.LIVE_TEST_REL]
                    fallback = ("step4_baseline",)
                for tag in fallback:
                    candidates += sorted((archive / tag).rglob("test_folder"))
            elif step in ("step4_baseline", "step9_baseline_to_shift_test"):
                candidates.append(work / self.driver.BASELINE_TRAIN_REL)
                candidates += sorted((archive / "step4_baseline").rglob("train_folder"))
            elif step == "step8_distribution_shift":
                candidates.append(work / self.driver.STEP8_TRAIN_REL)
            elif step.startswith("step5_"):
                candidates.append(work / self.driver.STEP5_ABLATION_REL / step[len("step5_"):])
        seen = set()
        for folder in candidates:
            folder = folder.resolve()
            if folder in seen:
                continue
            seen.add(folder)
            if not inside(folder, self.args.data_root):
                raise EvidenceError(f"Dataset override is outside --data-root: {folder}")
            if any((re.fullmatch(r"T\d+", p) and p != "T7") or
                   (re.fullmatch(r"h\d+", p) and p != "h7") for p in folder.parts):
                continue
            if folder.is_dir() and next(folder.glob("*.pt"), None) is not None:
                # Archive filename shortening must not change historical evaluation order.
                restore_names = folder != original and (folder / "manifest.csv").is_file()
                return folder, restore_names
        raise EvidenceError(f"Missing {kind} window identifiers for {task['key']}. No CSV and no saved graph dataset. "
                            f"Searched: {[str(p) for p in seen]}. No data will be regenerated.")

    def dataset_rows(self, folder: Path, restore_names: bool,
                     virtual_entries: Optional[Sequence[Tuple[Path, str]]] = None,
                     recovery_contract: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        # A virtual entry maps a surviving nested PT file to the exact filename it
        # had in the removed flat training folder. No copies or links are created.
        def physical_key(path: Path) -> str:
            return path.relative_to(folder).as_posix() if virtual_entries is not None else path.name

        if virtual_entries is None:
            physical = sorted(folder.glob("*.pt"), key=lambda p: natural_key(p.name))
            names = {p.name: p.name for p in physical}
        else:
            if restore_names:
                raise EvidenceError("A virtual collector view cannot also use an archive-name map.")
            physical = sorted([p for p, _ in virtual_entries], key=lambda p: natural_key(physical_key(p)))
            names = {physical_key(p): name for p, name in virtual_entries}
            if len(names) != len(physical) or len(set(names.values())) != len(names):
                raise EvidenceError(f"Duplicate physical or pooled filenames in the recovery view: {folder}")
            for path in physical:
                if not path.is_file() or not inside(path, self.args.data_root):
                    raise EvidenceError(f"Recovery tensor is missing or outside --data-root: {path}")
            for name in names.values():
                if Path(name).name != name or not name.endswith(".pt"):
                    raise EvidenceError(f"Invalid reconstructed pooled filename: {name}")
        if not physical:
            raise EvidenceError(f"No saved graph tensors found: {folder}")
        inventory = [{"name": physical_key(p), "bytes": p.stat().st_size,
                      "mtime_ns": p.stat().st_mtime_ns} for p in physical]
        if restore_names:
            manifest = self.csv(folder / "manifest.csv")
            mapping = {r["archived_name"]: Path(r["relative_source_path"]).name for r in manifest
                       if r.get("archived_name", "").endswith(".pt")}
            if set(names) != set(mapping) or len(set(mapping.values())) != len(mapping):
                raise EvidenceError(f"Ambiguous archive filename map: {folder / 'manifest.csv'}")
            names = mapping
        order = sorted(physical, key=lambda p: natural_key(names[physical_key(p)]))
        payload = {"folder": str(folder), "inventory": inventory, "names": names,
                   "dataset_source": file_hash(self.args.repo / "temporal_graph_dataset.py")}
        if recovery_contract is not None:
            payload["recovery_contract"] = recovery_contract
        fingerprint = digest(payload)
        if fingerprint in self.metadata:
            return self.metadata[fingerprint]
        cache = self.args.out_root / "metadata_cache" / (fingerprint + ".json")
        if cache.is_file() and not self.args.verify_data_hashes:
            saved = read_json(cache)
            if digest(saved.get("rows")) != saved.get("rows_sha256"):
                raise EvidenceError(f"Cached metadata changed: {cache}")
            rows = normalize_rows(saved["rows"], probabilities=False)
            self.metadata[fingerprint] = rows
            return rows
        # All graph files are read once per unique inventory, without running the dataset
        # class (whose optional vocabulary cache can write into original directories).
        try:
            import torch
        except ImportError as exc:
            raise EvidenceError("Graph metadata recovery needs the original PyTorch/PyG environment; "
                                "alternatively supply original prediction CSVs.") from exc
        groups: Dict[str, List[Tuple[int, str, int]]] = {}
        source_trajectories: Dict[str, str] = {}
        trajectory_sources: Dict[str, str] = {}
        hashes = []
        print(f"  Reading original metadata: {folder} ({len(order)} graphs)", flush=True)
        for i, path in enumerate(order, 1):
            try:
                raw = path.read_bytes()
                graph = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
            except Exception as exc:
                raise EvidenceError(f"Cannot read saved graph metadata {path}: {exc}") from exc
            sid, day = getattr(graph, "sim_id", None), getattr(graph, "day", None)
            target = getattr(graph, "y_h7_endog_majority", None)
            if sid is None or day is None or target is None:
                raise EvidenceError(f"Missing original sim_id/day/y_h7_endog_majority: {path}")
            y = int(target.item()) if hasattr(target, "item") else int(target)
            sid, day = str(sid), int(day)
            if virtual_entries is not None:
                origin = path.parent.relative_to(folder).as_posix()
                if source_trajectories.setdefault(origin, sid) != sid:
                    raise EvidenceError(f"A source simulation directory has multiple trajectory IDs: {path.parent}")
                if trajectory_sources.setdefault(sid, origin) != origin:
                    raise EvidenceError(f"The same trajectory ID occurs in different source simulation directories: {sid}")
            groups.setdefault(sid, []).append((day, names[physical_key(path)], y))
            hashes.append({"name": physical_key(path), "sha256": hashlib.sha256(raw).hexdigest()})
            del graph, raw
            if i % 1000 == 0:
                print(f"    indexed {i}/{len(order)}", flush=True)
        rows = []
        for sid, days in groups.items():
            days.sort(key=lambda d: d[0])
            if len({d[0] for d in days}) != len(days):
                raise EvidenceError(f"Duplicate days in original trajectory {sid} in {folder}")
            for start in range(len(days) - 6):
                window = days[start:start + 7]
                if [d[0] for d in window] != list(range(window[0][0], window[0][0] + 7)):
                    raise EvidenceError(f"Noncontiguous original history in {folder}/{sid}; cannot infer eligible rows.")
                rows.append({"sim_id": sid, "day": window[-1][0], "start_day": window[0][0],
                             "last_file": window[-1][1], "y": window[-1][2]})
        rows = normalize_rows(rows, probabilities=False)
        after = [{"name": physical_key(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in physical]
        if inventory != after:
            raise EvidenceError(f"Dataset changed while indexing: {folder}")
        metadata = {"folder": str(folder), "inventory": inventory, "original_names_restored": restore_names,
                    "graph_sha256": hashes, "rows": rows, "rows_sha256": digest(rows)}
        if recovery_contract is not None:
            metadata["recovery_contract"] = recovery_contract
            metadata["virtual_filename_map"] = names
        if not self.args.preflight:
            atomic_json(cache, metadata)
        self.metadata[fingerprint] = rows
        return rows

    def recovery_contract(self, filename: str) -> Dict[str, Any]:
        """Verify the reviewed metadata-preserving copier, without executing it."""
        if filename in self.recovery_contracts:
            return self.recovery_contracts[filename]
        path = self.args.repo / filename
        if not path.is_file():
            raise EvidenceError(f"Metadata recovery needs its original source contract: {path}")
        self.record(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            raise EvidenceError(f"Cannot inspect the recovery source: {path}") from exc
        symbols, expected = RECOVERY_CONTRACTS[filename]
        found: Dict[str, Any] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in symbols:
                found[node.name] = node
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in symbols:
                        found[target.id] = node
        if set(found) != set(symbols) or digest(
                [recovery_ast_signature(found[key]) for key in symbols]) != expected:
            raise EvidenceError(f"The copying/naming contract changed in {path}. "
                                "Recovery is withheld rather than guessing historical window order.")
        result = {"script": filename, "source_sha256": self.evidence[str(path)]["sha256"],
                  "contract_sha256": expected, "executed": False}
        self.recovery_contracts[filename] = result
        return result

    def recovered_training_rows(self, task: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
        """Recover deleted Step 5--7 flat-folder metadata from surviving originals.

        This function is used only when ordinary saved-CSV/flat-folder discovery
        fails. It never searches for a label-matching cohort or changes an ID.
        """
        track, step = task["track"], task["step"]
        work = self.args.data_root / f"TRACK_{track}" / "work"
        archive = work / "repro_artifacts_steps_1_9"
        original = self.rebase(cfg.get("data_folder"), track)
        override = self.overrides.get(f"{track}/{step}", {}).get("train_dataset")
        explicit = Path(override).expanduser().resolve() if override else None
        if explicit is not None and not inside(explicit, self.args.data_root):
            raise EvidenceError(f"Recovery override is outside --data-root: {explicit}")
        note: Dict[str, Any] = {"removed_training_folder": str(original) if original else None,
                                "recovery_only": "metadata; no graph conversion, model inference or training"}
        if step in ("step5_core_node_features_only", "step5_no_edge_weights"):
            # The reviewed Step 5 scripts read canonical PT files, alter only x
            # or edge_attr, and save with the same name, sim_id/day and labels.
            if explicit is not None:
                raise EvidenceError(f"The explicit Step 5 training dataset has no direct PT files: {explicit}")
            helper = "ablate_node_features.py" if step.endswith("core_node_features_only") else "ablate_edge_weights.py"
            contract = self.recovery_contract(helper)
            expected_suffix = (str(self.driver.STEP5_ABLATION_REL), step[len("step5_"):])
            if original is None or original.parts[-2:] != expected_suffix:
                raise EvidenceError(f"Step 5 recorded data_folder is not the reviewed ablation output: {original}")
            canonical = work / self.driver.BASELINE_TRAIN_REL
            candidates = [canonical] + sorted((archive / "step4_baseline").rglob("train_folder"))
            for folder in candidates:
                if not inside(folder, self.args.data_root):
                    raise EvidenceError(f"Canonical metadata fallback is outside --data-root: {folder}")
                if any((re.fullmatch(r"T\d+", p) and p != "T7") or
                       (re.fullmatch(r"h\d+", p) and p != "h7") for p in folder.parts):
                    continue
                if not folder.is_dir() or next(folder.glob("*.pt"), None) is None:
                    continue
                restore = folder != canonical and (folder / "manifest.csv").is_file()
                # A restored canonical folder can itself carry an archive map.
                if folder == canonical and (folder / "manifest.csv").is_file():
                    manifest = self.csv(folder / "manifest.csv")
                    restore = bool(manifest and "archived_name" in manifest[0])
                rows = self.dataset_rows(folder, restore)
                note.update(source_folder=str(folder), method="metadata-preserving Step 5 canonical fallback",
                            source_contract=contract, original_names_restored=restore)
                self.recovery_notes[task["key"]] = note
                return rows, f"Step 5 identifiers/labels from preserved canonical training PT metadata; {folder}"
            raise EvidenceError(f"Canonical training metadata is also unavailable for {task['key']}.")

        match = re.fullmatch(r"step6_(delay|freq)_(\d+)", step)
        if match:
            arm, number = match.groups()
            condition = f"{arm}_{int(number)}"
            helper = "convert_collect_delay_grid.py" if arm == "delay" else "convert_collect_freq_grid.py"
            contract = self.recovery_contract(helper)
            if explicit is not None:
                source = explicit
            else:
                if original is None or original.name != condition or not original.parent.name.endswith("_pt_flat"):
                    raise EvidenceError(f"Cannot derive the original Step 6 source from {original}. "
                                        "Use --source-map with the exact surviving condition folder.")
                source = original.parent.with_name(original.parent.name[:-len("_pt_flat")]) / condition
            if source.name != condition:
                raise EvidenceError(f"Step 6 recovery must use only condition {condition}, not the complete grid: {source}")
            if not source.is_dir() or not inside(source, self.args.data_root):
                raise EvidenceError(f"Surviving Step 6 source folder not found within --data-root: {source}")
            sim_dirs = sorted([p for p in source.rglob("sim_*") if p.is_dir()],
                              key=lambda p: p.relative_to(source).as_posix())
            entries: List[Tuple[Path, str]] = []
            for sim_dir in sim_dirs:
                # Exact _relative_sim_tag/_safe_tag rule of the verified collector.
                rel = sim_dir.relative_to(source)
                tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", "__".join(rel.parts))
                tag = re.sub(r"_+", "_", tag).strip("_") or "item"
                pts = sorted(sim_dir.glob("*.pt"))
                if not pts:
                    raise EvidenceError(f"A Step 6 simulation directory has no surviving tensors: {sim_dir}")
                entries.extend((p, f"{tag}__{p.name}") for p in pts)
            method = "original Step 6 collector filenames reconstructed in memory"
        elif step == "step7_sweep":
            contract = self.recovery_contract("convert_collect_sweep.py")
            if explicit is not None:
                source = explicit
            else:
                if original is None or not original.name.endswith("_pt_flat"):
                    raise EvidenceError(f"Cannot derive the original Step 7 source from {original}. "
                                        "Use --source-map with the exact surviving sweep folder.")
                source = original.with_name(original.name[:-len("_pt_flat")])
            if not source.is_dir() or not inside(source, self.args.data_root):
                raise EvidenceError(f"Surviving Step 7 source folder not found within --data-root: {source}")
            sim_dirs = sorted([p for p in source.iterdir() if p.is_dir() and p.name.startswith("sim_")])
            entries = []
            for sim_dir in sim_dirs:
                pts = sorted(sim_dir.glob("*.pt"))
                if not pts:
                    raise EvidenceError(f"A Step 7 simulation directory has no surviving tensors: {sim_dir}")
                entries.extend((p, f"{sim_dir.name}__{p.name}") for p in pts)
            method = "original Step 7 collector filenames reconstructed in memory"
        else:
            raise EvidenceError(f"No metadata-only recovery rule for {task['key']}.")
        if not sim_dirs or not entries:
            raise EvidenceError(f"No original sim_*/*.pt tensors survived in {source}.")
        expected_paths = {str(p.resolve()) for p, _ in entries}
        actual_paths = {str(p.resolve()) for p in source.rglob("*.pt")}
        if len(expected_paths) != len(entries) or expected_paths != actual_paths:
            raise EvidenceError(f"Ambiguous, duplicate or uncollected PT files in {source}; recovery is withheld.")
        note.update(source_folder=str(source), method=method, source_contract=contract,
                    n_daily_graphs=len(entries), n_simulation_directories=len(sim_dirs))
        rows = self.dataset_rows(source, False, virtual_entries=entries, recovery_contract=contract)
        # The original collector preserves Data.sim_id; a source directory must
        # not silently supply multiple independent trajectory IDs or vice versa.
        if len({row["sim_id"] for row in rows}) != len(sim_dirs):
            raise EvidenceError(f"Recovered trajectory count differs from original simulation directories: {source}")
        self.recovery_notes[task["key"]] = note
        return rows, f"{method}; unchanged saved sim_id/day/labels; {source}"

    def recover_metadata(self, task: Dict[str, Any], summary: Dict[str, Any], split: str) -> Tuple[List[Dict[str, Any]], str]:
        import numpy as np
        cfg = summary["config"]
        kind = "test" if split == "test" else "train"
        recovered = False
        try:
            folder, restore = self.dataset_candidate(task, cfg, kind)
        except EvidenceError as missing:
            if kind != "train" or not task["step"].startswith(("step5_", "step6_", "step7_")):
                raise
            try:
                rows, provenance = self.recovered_training_rows(task, cfg)
                recovered = True
            except (EvidenceError, FileNotFoundError, KeyError, ValueError) as exc:
                raise EvidenceError(f"{missing} Metadata-only recovery: {exc}") from exc
        else:
            rows = self.dataset_rows(folder, restore)
            provenance = f"saved PT metadata; historical natural-file/group/day order; {folder}"
        if split != "test":
            if "split_seed" not in cfg:
                raise EvidenceError(f"Original split_seed not recorded for {task['key']}; it is not guessed.")
            ids = sorted({r["sim_id"] for r in rows})
            if len(ids) < 2:
                raise EvidenceError("Cannot reconstruct the original trajectory-level validation split.")
            np.random.RandomState(int(cfg["split_seed"])).shuffle(ids)
            nval = max(1, min(len(ids) - 1, int(round(len(ids) * 0.2))))
            val = set(ids[:nval])
            rows = [r for r in rows if (r["sim_id"] in val) == (split == "validation")]
        if recovered:
            note = self.recovery_notes[task["key"]]
            note.update(split_seed=int(cfg["split_seed"]),
                        n_recovered_training_trajectories=len(ids),
                        validation_trajectory_ids=sorted(val))
            note.setdefault("recovered_splits", {})[split] = {
                "n_windows": len(rows), "rows_sha256": digest(rows)}
        return rows, provenance

    def gnn(self, task: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        import numpy as np
        path = Path(task["summary"])
        value = self.json(path)
        cfg = value.get("config", {})
        if value.get("task") != TASK or int(cfg.get("T", -1)) != 7 or int(cfg.get("sliding_step", -1)) != 1:
            raise EvidenceError(f"Wrong task/history/stride in {path}")
        if cfg.get("train_model") is True and task["step"] == "step9_baseline_to_shift_test":
            raise EvidenceError(f"Step 9 is recorded as trained, not zero-shot: {path}")
        by_split, notes = {}, {}
        override = self.overrides.get(f"{task['track']}/{task['step']}", {})
        for split in self.args.splits:
            block = value.get(split)
            if not isinstance(block, dict):
                notes[split] = {"error": f"No saved {split} output in {path}. Validation is never relabelled as train."}
                continue
            try:
                alternate = override.get("predictions", {}).get(split)
                csv_path = Path(alternate).expanduser().resolve() if alternate else path.parent / f"{split}_predictions.csv"
                if alternate and not inside(csv_path, self.args.data_root):
                    raise EvidenceError("Prediction CSV overrides must be within --data-root.")
                if csv_path.is_file():
                    rows = normalize_rows(self.csv(csv_path))
                    provenance = f"original prediction CSV: {csv_path}"
                else:
                    meta, provenance = self.recover_metadata(task, value, split)
                    if [r["y"] for r in meta] != block.get("y_true"):
                        raise EvidenceError(f"{path}/{split}: graph-derived label/window order differs from saved predictions. "
                                            "Refusing to attach clusters from counts alone.")
                    probabilities = np.asarray(block.get("probabilities"), dtype=float)
                    preds = block.get("y_pred")
                    if probabilities.shape != (len(meta), 2) or not isinstance(preds, list) or len(preds) != len(meta):
                        raise EvidenceError(f"{path}/{split}: probabilities/decisions are unavailable.")
                    rows = normalize_rows([{**r, "probability": float(p[1]), "prediction": pred}
                                           for r, p, pred in zip(meta, probabilities, preds)])
                verify_arrays(rows, block, f"{task['key']}/{split}")
                point = self.stats.binary_metrics([r["y"] for r in rows], [r["probability"] for r in rows])
                verify_points(point, block, f"{task['key']}/{split}", gnn=True)
                by_split[split] = rows
                notes[split] = {"provenance": provenance, "metrics_verified": True,
                                "saved_model_seed": cfg.get("model_seed"), "saved_eval_seed": cfg.get("eval_seed")}
                if (split != "test" and task["key"] in self.recovery_notes and
                        split in self.recovery_notes[task["key"]].get("recovered_splits", {})):
                    notes[split]["metadata_recovery"] = clean_json(self.recovery_notes[task["key"]])
                    notes[split]["metadata_recovery"]["saved_label_order_verified"] = True
            except (EvidenceError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
                notes[split] = {"error": str(exc)}
        self.check_disjoint(by_split, task["key"])
        return by_split, notes

    def xgb(self, task: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        path = Path(task["summary"])
        value = self.json(path)
        if value.get("mode") != "ground_truth":
            raise EvidenceError(f"Not an original ground-truth XGBoost result: {path}")
        expected_tag = "step4_baseline" if task.get("alias") else task["xgb_tag"]
        if value.get("step") != expected_tag:
            raise EvidenceError(f"XGBoost step mismatch in {path}")
        expected_width = 28 if task["model"] == "XGB_all" else 14
        if int(value.get("feature_width", -1)) != expected_width:
            raise EvidenceError(f"XGBoost input width mismatch in {path}")
        if task["step"].startswith("step9_") and value.get("fit_status") != "zero_shot_baseline_model":
            raise EvidenceError(f"XGBoost Step 9 lacks its recorded zero-shot status: {path}")
        marker = path.parent / "_task.json"
        if marker.is_file():
            saved = self.json(marker)
            if saved.get("status") != "complete":
                raise EvidenceError(f"XGBoost task not complete: {marker}")
            for split in self.args.splits:
                name = f"{split}_predictions.csv"
                expected_hash = saved.get("artifacts", {}).get(name)
                if expected_hash and (not (path.parent / name).is_file() or file_hash(path.parent / name) != expected_hash):
                    raise EvidenceError(f"XGBoost saved artifact changed: {path.parent / name}")
        by_split, notes = {}, {}
        for split in self.args.splits:
            try:
                block = value.get(split)
                if not isinstance(block, dict):
                    raise EvidenceError(f"No original XGBoost {split} metrics in {path}")
                csv_path = path.parent / f"{split}_predictions.csv"
                rows = normalize_rows(self.csv(csv_path))
                point = self.stats.binary_metrics([r["y"] for r in rows], [r["probability"] for r in rows])
                verify_points(point, block, f"{task['key']}/{split}", gnn=False)
                by_split[split], notes[split] = rows, {"provenance": str(csv_path), "metrics_verified": True,
                                                      "alias": task.get("alias")}
            except (EvidenceError, FileNotFoundError, KeyError, ValueError) as exc:
                notes[split] = {"error": str(exc)}
        self.check_disjoint(by_split, task["key"])
        return by_split, notes

    @staticmethod
    def check_disjoint(by_split: Dict[str, List[Dict[str, Any]]], label: str) -> None:
        keys = list(by_split)
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if {r["sim_id"] for r in by_split[a]} & {r["sim_id"] for r in by_split[b]}:
                    raise EvidenceError(f"{label}: {a}/{b} trajectories overlap.")


def prepare(args: argparse.Namespace, reader: Reader, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] Check {task['key']}", flush=True)
        notes, data = {}, {}
        error = task.get("discovery_error")
        if not error:
            try:
                data, notes = reader.gnn(task) if task["model"] == "GNN" else reader.xgb(task)
            except (EvidenceError, FileNotFoundError, ValueError, KeyError, ImportError) as exc:
                error = str(exc)
        for split in args.splits:
            note = notes.get(split, {})
            issue = error or note.get("error")
            records.append({**task, "split": split, "status": "pending" if issue else "ready",
                            "error": issue, "provenance": note, "rows": data.get(split) if not issue else None})
    # Verify the shared original TEST cohort by exact window IDs and labels, not counts alone.
    cohort_lookup: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        if record["status"] != "ready" or record["split"] != "test":
            continue
        group = "shift" if record["step"].startswith(("step8_", "step9_")) else "canonical"
        cohort_lookup.setdefault((record["track"], group), []).append(record)
    for key, members in cohort_lookup.items():
        def fingerprint(record: Dict[str, Any]) -> str:
            return digest(sorted((r["sim_id"], r["day"], r["start_day"], r["y"]) for r in record["rows"]))
        if len({fingerprint(m) for m in members}) > 1:
            # All members are withheld: do not choose whichever dataset happens to be encountered first.
            for record in members:
                record.update(status="pending", error=f"Original {key} TEST windows/labels differ across archived models. "
                              "Choose the correct original summaries via --source-map; no mismatched rows are combined.", rows=None)
    return records


def bootstrap_rows_and_strata(record: Dict[str, Any], reader: Reader) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Keep cluster units unchanged; never mistake a legacy ID for its own stratum.

    The repeated-run helper expects family__sim_ID. Canonical/shift IDs already
    have that form. Old Step 6 IDs may be sim_ID__hash; their saved collector
    filenames encode endog/import. Otherwise use a single cohort stratum and
    explicitly report an unstratified whole-trajectory bootstrap. Parameter
    bands are never guessed from labels or a current generator's default grid.
    """
    rows = record["rows"]
    ids = sorted({r["sim_id"] for r in rows})
    override = reader.overrides.get(f"{record['track']}/{record['step']}", {}).get("family_by_sim_id", {})
    filenames: Dict[str, set] = {}
    for row in rows:
        filenames.setdefault(row["sim_id"], set()).add(row.get("last_file", ""))
    families, origins = {}, {}
    for sid in ids:
        if override:
            if sid not in override or not str(override[sid]).strip():
                raise EvidenceError(f"Explicit family_by_sim_id lacks a family for {sid}.")
            family, origin = str(override[sid]), "explicit source-map"
        elif "__sim_" in sid and sid.split("__sim_", 1)[0]:
            family, origin = sid.split("__sim_", 1)[0], "recorded trajectory ID"
        else:
            labels = set()
            for name in filenames[sid]:
                # Exact filename pattern of convert_collect_delay_grid/freq_grid.
                match = re.match(r"^(endog|import)_sim_\d+__", name)
                if match:
                    labels.add(match.group(1))
            if len(labels) > 1:
                raise EvidenceError(f"Contradictory original family filenames for {sid}.")
            family = next(iter(labels)) if labels else None
            origin = "original Step 6 collector filename" if labels else "not recorded"
        families[sid], origins[sid] = family, origin
    unstratified = any(family is None for family in families.values())
    if unstratified:
        # A mixture of known and unknown families is NOT partially stratified.
        families = {sid: "original_cohort" for sid in ids}
    if not override and not unstratified and all(origins[sid] == "recorded trajectory ID" for sid in ids):
        transformed = rows  # Exact helper inputs used in the previous Step 4/9 comparison.
    else:
        # Encoding affects only helper stratification; original exported IDs never change.
        labels = sorted(set(families.values()))
        label_index = {v: i for i, v in enumerate(labels)}
        remap = {sid: f"stratum_{label_index[families[sid]]:06d}__sim_{i:09d}" for i, sid in enumerate(ids)}
        transformed = [{**r, "sim_id": remap[r["sim_id"]]} for r in rows]
    return transformed, {
        "scheme": "unstratified whole-trajectory bootstrap" if unstratified else "whole-trajectory bootstrap within simulator family",
        "reason": "Original family is unavailable for at least one trajectory; no family is inferred from outcomes." if unstratified else "Families taken from saved identifiers/collector filenames or explicit provenance mapping.",
        "family_by_original_sim_id": families,
        "family_source_by_original_sim_id": origins,
    }


def bootstrap_record(record: Dict[str, Any], reader: Reader, args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    rows = record["rows"]
    y, p = np.asarray([r["y"] for r in rows]), np.asarray([r["probability"] for r in rows])
    ci_rows, stratification = bootstrap_rows_and_strata(record, reader)
    point = reader.stats.binary_metrics(y, p)
    key = digest({"rows": [{k: r[k] for k in ("sim_id", "day", "start_day", "y", "probability", "prediction")} for r in rows],
                  "stratification": stratification, "ci_cluster_ids": [r["sim_id"] for r in ci_rows],
                  "count": args.bootstrap, "seed": args.bootstrap_seed,
                  "statistical_source": file_hash(args.repo / "summarize_predictive_repeats.py")})
    cache = args.out_root / "bootstrap_cache" / (key + ".json")
    if key in reader.cache_memory:
        result = reader.cache_memory[key]
    elif cache.is_file():
        wrapped = read_json(cache)
        result = wrapped.get("result")
        if digest(result) != wrapped.get("result_sha256"):
            raise EvidenceError(f"Bootstrap cache changed: {cache}")
    else:
        result = {"point": point, "intervals": reader.stats.trajectory_bootstrap(ci_rows, y, p, args.bootstrap, args.bootstrap_seed)}
        atomic_json(cache, {"result": result, "result_sha256": digest(result)})
    reader.cache_memory[key] = result
    for entry in result["intervals"].values():
        entry["method"] = "95% percentile " + stratification["scheme"] + "; conditional on fitted model"
    groups: Dict[str, set] = {}
    for r in ci_rows:
        groups.setdefault(r["sim_id"].split("__sim_", 1)[0], set()).add(r["sim_id"])
    warnings = []
    if stratification["scheme"].startswith("unstratified"):
        warnings.append(stratification["reason"] + " Used an explicitly reported unstratified trajectory bootstrap.")
    if len({r["sim_id"] for r in rows}) < 2:
        warnings.append("Fewer than two trajectories: confidence intervals are undefined.")
    if all(len(ids) == 1 for ids in groups.values()):
        warnings.append("One trajectory per stratum: stratified resampling cannot estimate between-trajectory variation.")
    for metric in METRICS:
        count = result["intervals"][metric]["valid_bootstraps"]
        if count != args.bootstrap:
            warnings.append(f"{metric}: only {count}/{args.bootstrap} resamples have a defined metric.")
    return {**result, "n_trajectories": len({r["sim_id"] for r in rows}),
            "family_sizes": {k: len(v) for k, v in groups.items()}, "warnings": warnings,
            "prediction_sha256": key, "stratification": stratification}


def tex_escape(value: Any) -> str:
    table = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(table.get(c, c) for c in str(value))


def tex_cell(record: Optional[Dict[str, Any]], metric: str, digits: int) -> str:
    if record is None or record["status"] != "complete":
        return r"\textit{Pending}"
    point = record["result"]["point"][metric]
    ci = record["result"]["intervals"][metric]
    if point is None:
        return r"\textit{Undefined}"
    val = f"{point:.{digits}f}"
    if ci["low"] is None or ci["high"] is None:
        return r"\shortstack{" + val + r"\\(CI unavailable)}"
    return r"\shortstack{" + val + rf"\\({ci['low']:.{digits}f}, {ci['high']:.{digits}f})" + "}"


def export_tables(args: argparse.Namespace, records: List[Dict[str, Any]], complete: bool) -> None:
    lookup = {(r["track"], r["step"], r["model"], r["split"]): r for r in records}
    combinations = list(dict.fromkeys((r["track"], r["step"], r["model"]) for r in records))
    notes = (
        r"Entries are the original fitted-model estimates with 95\% conditional percentile confidence intervals. "
        + f"All five metrics use {args.bootstrap:,} whole-trajectory bootstrap resamples, stratified by recorded simulator family where identifiable "
        + f"(seed {args.bootstrap_seed}); otherwise the cohort is resampled without stratification, as documented in the audit. Overlapping windows remain together. "
        + r"No fitting, model inference or simulation was performed for this report. "
        r"Validation is the held-out portion of each experiment's training trajectories, not the fitted training set. "
        r"Steps 5--7 use their original experiment-specific training conditions and the fixed original canonical test; "
        r"Step 8 uses its original shifted training/test datasets; Step 9 retains the canonical model and "
        r"its canonical validation partition and evaluates the shifted test without refitting. "
        r"XGBoost is ground-truth only; both count variants are shown where evaluated. The no-edge-weight "
        r"XGBoost row reuses the canonical prediction when its original alias output is absent, because "
        r"edge weights do not enter these summaries. The threshold remains 0.5, with exact ties assigned to class 0. "
        r"Undefined metrics/intervals are not imputed. No inference across training seeds, parameter draws or "
        r"independent model fits is implied, and no paired significance test is computed here. "
        r"The five-repeat results belong in a separate robustness table."
    )
    status = "" if complete else r"\textbf{Incomplete evidence report: Pending entries are not results.} "
    common = [r"% Requires: booktabs,longtable,array,pdflscape", r"\begin{landscape}",
              r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{1.5pt}", r"\renewcommand{\arraystretch}{1.16}"]
    columns = r"@{}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.11\linewidth}*{10}{>{\centering\arraybackslash}p{0.066\linewidth}}@{}"
    if args.digits == 4:
        common[3] = r"\setlength{\tabcolsep}{0.5pt}"
        columns = r"@{}>{\raggedright\arraybackslash}p{0.11\linewidth}>{\raggedright\arraybackslash}p{0.10\linewidth}*{10}{>{\centering\arraybackslash}p{0.075\linewidth}}@{}"
    title = status + r"Original predictive experiments: validation and test performance, with confidence intervals for every metric."
    header = [r"\toprule", r"Experiment & Model & \multicolumn{5}{c}{Validation} & \multicolumn{5}{c}{Test} \\",
              r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
              r" & & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\", r"\midrule"]
    lines = common + [r"\begin{longtable}{" + columns + "}", r"\caption{" + title + "}",
                      r"\label{tab:gnn-xgboost-all-metrics}", r"\label{tab:track-ground-truth-metrics-t7-h7}",
                      r"\label{tab:track-partial-observation-metrics-t7-h7}\\"] + header + [r"\endfirsthead"] + header + [r"\endhead", r"\bottomrule\endfoot"]
    current = None
    for track, step, model in combinations:
        if track != current:
            lines.append(r"\multicolumn{12}{@{}l}{\textbf{" + ("A. Ground truth" if track == "ground_truth" else "B. Partial observation") + r"}}\\")
            current = track
        cells = [tex_escape(STEP_LABELS.get(step, step)), tex_escape(MODEL_LABELS[model])]
        cells += [tex_cell(lookup.get((track, step, model, split)), m, args.digits) for split in SPLITS for m in METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\noindent " + notes, r"\endgroup", r"\end{landscape}"]
    (args.out_root / "table4_all_metrics_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Narrow test-only companion retains the same evidence and more legible metric cells.
    test_complete = all(lookup.get((track, step, model, "test"), {}).get("status") == "complete"
                        for track, step, model in combinations)
    test_status = "" if test_complete else r"\textbf{Incomplete test evidence: Pending entries are not results.} "
    lines = [r"% Requires: booktabs,longtable,array", r"\begingroup\small", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.18}", r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.3cm}>{\raggedright\arraybackslash}p{2cm}ccccc@{}}",
             r"\caption{" + test_status + r"Original predictive test results with conditional confidence intervals.}\label{tab:original-predictive-test-ci}\\"]
    header = [r"\toprule Experiment & Model & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\", r"\midrule"]
    lines += header + [r"\endfirsthead"] + header + [r"\endhead", r"\bottomrule\endfoot"]
    current = None
    for track, step, model in combinations:
        if track != current:
            lines.append(r"\multicolumn{7}{@{}l}{\textbf{" + tex_escape(track.replace("_", " ").title()) + r"}}\\")
            current = track
        cells = [tex_escape(STEP_LABELS.get(step, step)), tex_escape(MODEL_LABELS[model])]
        cells += [tex_cell(lookup.get((track, step, model, "test")), m, args.digits) for m in METRICS]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\end{longtable}", r"\noindent " + notes, r"\endgroup"]
    (args.out_root / "original_test_all_metrics_ci.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(args: argparse.Namespace, reader: Reader, tasks: List[Dict[str, Any]], identity: Dict[str, Any]) -> int:
    start = time.monotonic()
    prepared = prepare(args, reader, tasks)
    if args.preflight:
        public = [{k: v for k, v in r.items() if k != "rows"} for r in prepared]
        print(json.dumps(clean_json({"tasks": public, "ready_splits": sum(r["status"] == "ready" for r in prepared),
              "pending_splits": sum(r["status"] != "ready" for r in prepared), "metrics": METRICS,
              "bootstrap_resamples": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
              "training_runs": 0, "inference_runs": 0, "simulations": 0, "preflight_writes": 0}), indent=2))
        return 0 if all(r["status"] == "ready" for r in prepared) else 2
    metric_rows, audit_rows, warnings = [], [], []
    for i, r in enumerate(prepared, 1):
        if r["status"] == "ready":
            print(f"[{i}/{len(prepared)}] Bootstrap all metrics: {r['key']}/{r['split']}", flush=True)
            r["result"] = bootstrap_record(r, reader, args)
            r["status"] = "complete"
            dest = args.out_root / "aligned_predictions" / r["track"] / r["step"] / r["model"] / f"{r['split']}_predictions.csv"
            write_csv(dest, r["rows"])
            result = r["result"]
            for m in METRICS:
                metric_rows.append({"track": r["track"], "step": r["step"], "model": r["model"], "split": r["split"],
                                    "metric": m, "point": result["point"][m], **result["intervals"][m],
                                    "n_windows": result["point"]["n_windows"], "seed": args.bootstrap_seed,
                                    "source": r["summary"]})
            for w in result["warnings"]:
                warnings.append({"task": r["key"], "split": r["split"], "warning": w})
        else:
            print(f"  PENDING: {r['key']}/{r['split']}: {r['error']}", flush=True)
        audit_rows.append({"track": r["track"], "step": r["step"], "model": r["model"], "split": r["split"],
                           "status": r["status"], "source": r["summary"], "error": r["error"],
                           "provenance": json.dumps(r["provenance"], sort_keys=True)})
    for info in reader.evidence.values():
        path = Path(info["path"])
        if file_hash(path) != info["sha256"]:
            raise EvidenceError(f"Input changed during reporting: {path}")
    complete = all(r["status"] == "complete" for r in prepared)
    write_csv(args.out_root / "all_metrics_ci.csv", metric_rows)
    write_csv(args.out_root / "evidence_audit.csv", audit_rows)
    write_csv(args.out_root / "warnings.csv", warnings, ["task", "split", "warning"])
    atomic_json(args.out_root / "source_evidence.json", list(reader.evidence.values()))
    atomic_json(args.out_root / "verified_results.json", [{k: v for k, v in r.items() if k != "rows"} for r in prepared])
    export_tables(args, prepared, complete)
    summary = {"complete": complete, "version": VERSION, "original_data_only": True,
               "metrics": METRICS, "splits": args.splits, "tracks": args.tracks,
               "completed_splits": sum(r["status"] == "complete" for r in prepared),
               "requested_splits": len(prepared), "pending_splits": [r for r in audit_rows if r["status"] != "complete"],
               "bootstrap_count": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
               "ci_method": "95% percentile whole-trajectory bootstrap; stratified where original families are identifiable, otherwise explicitly unstratified; fixed fitted models",
               "training_runs": 0, "inference_runs": 0, "simulations": 0, "data_regeneration": 0,
               "does_not_estimate_between_fit_variability": True, "elapsed_seconds": time.monotonic() - start,
               "identity_sha256": digest(identity)}
    atomic_json(args.out_root / "summary.json", summary)
    note = ["ORIGINAL PREDICTIVE CONFIDENCE-INTERVAL REPORT", "",
            f"Complete: {complete}; verified splits: {summary['completed_splits']}/{len(prepared)}.",
            "Original point estimates are recomputed and checked, never replaced with repeat means.",
            "All five metrics have the same conditional trajectory-bootstrap procedure.",
            "The output contains no new model fits, predictions, simulated trajectories or labels.",
            "Existing input files are read-only. Only aligned copies and new statistical reports are written.",
            "Where recorded families exist, the original XGBoost intervals are recomputed using family stratification.",
            "Legacy IDs without identifiable families use explicitly reported unstratified whole-trajectory sampling.",
            "Family groupings are never inferred from predicted or true labels; see verified_results.json.",
            "Consequently, original interval bounds can change while original point estimates remain unchanged.",
            "Validation is not training; training rows are included only when --include-train and saved train predictions exist.",
            "Intervals reflect test/validation trajectory sampling, not model-initialisation or simulator-parameter uncertainty.",
            "Step 5 may recover identifiers/labels from metadata-preserving canonical source tensors.",
            "Steps 6/7 may recover original pooled filenames from surviving nested source tensors, in memory only.",
            "Recovery uses reviewed collector contracts and verifies saved validation label/window ordering.",
            "Pending rows need saved predictions with identifiers or the original archived graph metadata; see evidence_audit.csv.",
            "Do not replace the manuscript table with an incomplete report.", "",
            "Tables require booktabs, longtable, array and pdflscape. The long table can span pages."]
    (args.out_root / "report_notes.txt").write_text("\n".join(note) + "\n", encoding="utf-8")
    # Include exact small source files used for statistics/discovery, but no graph datasets/checkpoints.
    for name in (*HELPERS, "temporal_graph_dataset.py"):
        shutil.copy2(args.repo / name, args.out_root / "sources" / name)
    for name in reader.recovery_contracts:
        shutil.copy2(args.repo / name, args.out_root / "sources" / name)
    shutil.copy2(Path(__file__), args.out_root / "sources" / Path(__file__).name)
    with zipfile.ZipFile(args.out_root / "original_predictive_ci_review_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.out_root.rglob("*")):
            if not path.is_file() or path.suffix == ".zip" or path.name.startswith(".") or ".tmp" in path.name:
                continue
            rel = path.relative_to(args.out_root)
            if rel.parts[0] in ("bootstrap_cache", "metadata_cache"):
                continue
            z.write(path, rel.as_posix())
    print(f"\n{'COMPLETE' if complete else 'INCOMPLETE; inspect evidence_audit.csv'}: {args.out_root}", flush=True)
    return 0 if complete else 2


def self_test(stats: Any) -> None:
    import numpy as np
    rows = [{"sim_id": f"family{j // 2}__sim_{j:03d}", "day": 7 + i, "start_day": 1 + i,
             "y": (i + j) % 2, "probability": (.2, .7, .5, .8)[(j + i) % 4],
             "prediction": int((.2, .7, .5, .8)[(j + i) % 4] > .5)} for j in range(4) for i in range(5)]
    rows = normalize_rows(rows)
    y = np.asarray([r["y"] for r in rows]); p = np.asarray([r["probability"] for r in rows])
    observed = stats.trajectory_bootstrap(rows, y, p, 47, 719)
    vals = {m: [] for m in METRICS}
    rng = np.random.default_rng(719)
    for _ in range(47):
        idx = []
        for family in range(2):
            selected = rng.choice([family * 2, family * 2 + 1], 2, replace=True)
            idx += [j * 5 + i for j in selected for i in range(5)]
        result = stats.binary_metrics(y[idx], p[idx])
        for m in METRICS:
            if result[m] is not None:
                vals[m].append(result[m])
    for m in METRICS:
        if observed[m]["low"] != float(np.quantile(vals[m], .025)) or observed[m]["high"] != float(np.quantile(vals[m], .975)):
            raise EvidenceError(f"Trajectory-bootstrap numerical contract failed: {m}")
    try:
        normalize_rows(rows + rows[:1])
    except EvidenceError:
        pass
    else:
        raise EvidenceError("Duplicate-window safety test failed.")
    print("PASS: all five metrics, stratified trajectory bootstrap, threshold ties and duplicate-window checks.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="ORIGINAL experiments_results; not predictive_repeat_results")
    parser.add_argument("--xgboost-root", type=Path, help="Original full XGBoost results; omit only for a GNN-only report")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, default=list(TRACKS))
    parser.add_argument("--include-train", action="store_true", help="Also export saved fitted-training metrics/CIs to CSV; never relabel validation")
    parser.add_argument("--ci-config", type=Path, help="Defaults to repo/predictive_repeats.json; reads bootstrap count/seed only")
    parser.add_argument("--bootstrap", type=int, help="Override bootstrap resample count")
    parser.add_argument("--bootstrap-seed", type=int, help="Override bootstrap RNG seed")
    parser.add_argument("--digits", type=int, choices=(3, 4), default=3, help="LaTeX decimals; CSV keeps full precision")
    parser.add_argument("--source-map", type=Path, help="Optional exact summary/dataset/CSV paths for ambiguous archives; see README")
    parser.add_argument("--verify-data-hashes", action="store_true", help="Read original graph metadata again instead of using this output's verified cache")
    parser.add_argument("--preflight", action="store_true", help="Check all evidence/identities/metrics, without bootstrap or output writes")
    parser.add_argument("--self-test", action="store_true", help="Small statistics/validation test; needs only --repo")
    args = parser.parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    if args.self_test:
        _, stats = load_helpers(args.repo)
        self_test(stats)
        return 0
    if args.data_root is None or args.out_root is None:
        parser.error("--data-root and --out-root are required except with --self-test")
    # Reject symlink spelling before resolve() removes it.
    if args.out_root.expanduser().is_symlink():
        raise EvidenceError("Output root is a symlink; choose an ordinary new directory.")
    args.data_root, args.out_root = args.data_root.expanduser().resolve(), args.out_root.expanduser().resolve()
    args.xgboost_root = args.xgboost_root.expanduser().resolve() if args.xgboost_root else None
    args.tracks = [t for t in TRACKS if t in args.tracks]
    args.splits = ("train", *SPLITS) if args.include_train else SPLITS
    args.ci_config = (args.ci_config or args.repo / "predictive_repeats.json").expanduser().resolve()
    validate_paths(args)
    inputs = [args.data_root] + ([args.xgboost_root] if args.xgboost_root else [])
    protect_inputs(args.repo, inputs, args.out_root)
    driver, stats = load_helpers(args.repo)
    cfg = read_json(args.ci_config) if args.ci_config.is_file() else {}
    args.bootstrap = args.bootstrap if args.bootstrap is not None else cfg.get("bootstrap_replicates", 1000)
    args.bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else cfg.get("bootstrap_seed", 202609)
    if args.bootstrap < 1 or not 0 <= args.bootstrap_seed < 2**32:
        parser.error("bootstrap must be positive and bootstrap-seed must be in [0, 2**32)")
    overrides = read_json(args.source_map.expanduser().resolve()) if args.source_map else {}
    if not isinstance(overrides, dict):
        raise EvidenceError("--source-map must contain a JSON object keyed by track/step.")
    tasks = discover_tasks(args, overrides)
    identity = {"version": VERSION, "runner_sha256": file_hash(Path(__file__)),
                "repo": str(args.repo), "data_root": str(args.data_root), "xgboost_root": str(args.xgboost_root),
                "source_sha256": {name: file_hash(args.repo / name) for name in (*HELPERS, "temporal_graph_dataset.py")},
                 "recovery_source_sha256": {name: file_hash(args.repo / name) for name in RECOVERY_CONTRACTS
                                            if (args.repo / name).is_file()},
                "tracks": args.tracks, "splits": args.splits, "bootstrap": args.bootstrap,
                "bootstrap_seed": args.bootstrap_seed, "source_map": overrides}
    marker = args.out_root / OWNER
    if marker.is_file() and read_json(marker).get("identity_sha256") != digest(identity):
        raise EvidenceError("Source/settings/input roots differ from this report's manifest. Choose a new --out-root.")
    reader = Reader(args, driver, stats, overrides)
    if args.preflight:
        return run_report(args, reader, tasks, identity)
    with output_lock(args.out_root):
        atomic_json(marker, {"identity": identity, "identity_sha256": digest(identity)})
        (args.out_root / "sources").mkdir(exist_ok=True)
        return run_report(args, reader, tasks, identity)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
