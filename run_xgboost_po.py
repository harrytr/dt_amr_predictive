#!/usr/bin/env python3
"""Matched partial-observation XGBoost: original data and five predictive repeats.

Train only XGBoost, on the EXISTING PO tensors used by the corresponding GNN.
Steps 4 and 8 fit separate canonical/shifted models; Step 9 reuses Step 4.
Two variants retain/remove admission and discharge counts on EVERY history day.

PO inputs: observed-positive patient count, admissions, discharges, and the
masked (zero) incident-event channel. No latent resistance state or unmasked
incident events, contact topology, future events, identifiers or regime labels
are predictors. The target is the unchanged saved seven-day GNN label.

All inputs are read-only. A separate owned output folder holds models, cached
features, aligned predictions, all-metric intervals, paired differences and TeX.
No original driver, simulator, converter, GNN trainer or GNN inference is run.

Save as addons/run_xgboost_po.py. See README_xgboost_po.txt for commands.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

sys.dont_write_bytecode = True
VERSION = "1.0.0"
MODE = "partial_observation"
TASK = "endogenous_importation_majority_h7"
HISTORY = 7
STEP4 = "step4_baseline"
STEP8 = "step8_distribution_shift"
STEP9 = "step9_baseline_to_shift_test"
STEPS = (STEP4, STEP8, STEP9)
EDGE_ALIAS = "step5_no_edge_weights"
SPLITS = ("train", "validation", "test")
METRICS = ("auroc", "accuracy", "macro_f1", "sensitivity", "specificity")
VARIANTS = ("all_summaries", "without_counts")
MODELS = ("GNN", "XGB_all", "XGB_no_counts")
MODEL_NAMES = {"GNN": "GNN + Transformer", "XGB_all": "XGBoost: all PO summaries",
               "XGB_no_counts": "XGBoost: without counts"}
STEP_NAMES = {STEP4: "Step 4: canonical", STEP8: "Step 8: shifted train/test",
              STEP9: "Step 9: zero-shot", EDGE_ALIAS: "Step 5: no edge weights"}
KINDS = ("canonical_train", "canonical_test", "shift_train", "shift_test")
DAILY_FEATURES = ("observed_positive_patients", "patient_admissions", "patient_discharges",
                  "masked_recent_events")
HELPERS = ("experiments_pbc7.py", "experiments_xgboost.py", "predictive_repeats.py",
           "summarize_predictive_repeats.py", "temporal_graph_dataset.py", "convert_to_pt.py")
OWNER = "xgboost_po_manifest.json"
FEATURE_CONTRACT = {
    "observation": MODE, "history_days": 7, "forecast_days": 7, "sliding_step": 1,
    "daily_features": list(DAILY_FEATURES),
    "state_input": "count of patient nodes with PO x[:,1] == 1 (observed positive)",
    "event_input": "PO x[:,4:6] must both be zero; zero placeholder is retained",
    "flows": "patient identifier-set differences between consecutive saved days",
    "first_day_flows": "NaN (unavailable), never inferred as zero or initial census",
    "ordering": "seven daily vectors concatenated chronologically, oldest first",
    "all_summaries_columns": 28, "without_counts_columns": 14,
    "without_counts": "remove admissions/discharges on all seven days in every partition",
    "constant_columns": "seven event placeholders are zero in either variant",
    "not_predictors": ["future outcome counters", "labels", "latent resistance states",
                       "contact edges", "regime identity", "node identifiers", "calendar day"],
    "eligibility": "unchanged complete histories; saved future labels and terminal windows retained",
    "threshold": "probability > 0.5; exact equality predicts class zero",
}


class EvidenceError(RuntimeError):
    """Missing, incompatible, ambiguous or changed source evidence."""


class MissingEvidence(EvidenceError):
    """Required inputs are absent; no substitute data will be generated."""


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise MissingEvidence(f"Missing saved file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise MissingEvidence(f"Missing saved predictions: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, sort_keys=True,
                              allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: clean(v) for k, v in row.items()} for row in rows])
    os.replace(temp, path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def natural_key(value: Any) -> List[Any]:
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r"(\d+)", str(value))]


def integer(value: Any, label: str) -> int:
    if hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise EvidenceError(f"Missing or invalid integer {label}: {value}") from exc
    if not math.isfinite(number) or number != int(number):
        raise EvidenceError(f"Expected an integer {label}: {value}")
    return int(number)


def import_source(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"Cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Helpers:
    driver: Any
    xgb: Any
    repeats: Any
    stats: Any
    hashes: Dict[str, str]
    source: Path


def load_helpers(source: Path, tag: str, expected: Optional[Mapping[str, str]] = None) -> Helpers:
    hashes = {}
    for name in HELPERS:
        path = source / name
        if not path.is_file():
            raise MissingEvidence(f"Missing helper from the supplied codebase: {path}")
        hashes[name] = file_hash(path)
        if expected is not None and expected.get(name) != hashes[name]:
            raise EvidenceError(f"Frozen repeat source changed: {path}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    old = sys.modules.get("predictive_repeats")
    sys.path.insert(0, str(source))
    try:
        driver = import_source(source / HELPERS[0], f"_po_driver_{tag}")
        xgb = import_source(source / HELPERS[1], f"_po_xgb_{tag}")
        repeats = import_source(source / HELPERS[2], "predictive_repeats")
        stats = import_source(source / HELPERS[3], f"_po_stats_{tag}")
    finally:
        sys.path.pop(0)
        if old is not None:
            sys.modules["predictive_repeats"] = old
        else:
            sys.modules.pop("predictive_repeats", None)
    if tuple(stats.METRICS) != METRICS or tuple(repeats.STEPS) != STEPS:
        raise EvidenceError("Reference task or metric scope differs; refusing a silent change.")
    for name in ("binary_metrics", "trajectory_bootstrap", "mean_interval"):
        if not callable(getattr(stats, name, None)):
            raise EvidenceError(f"The summariser lacks {name}.")
    for name in ("xgb_parameters", "XGBoostBinaryModel", "original_split", "array_hash"):
        if not callable(getattr(xgb, name, None)):
            raise EvidenceError(f"The XGBoost helper lacks {name}.")
    return Helpers(driver, xgb, repeats, stats, hashes, source)


def copy_sources(helper: Helpers, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, sha in helper.hashes.items():
        target = destination / name
        if target.exists():
            if file_hash(target) != sha:
                raise EvidenceError(f"Output source snapshot changed: {target}")
        else:
            shutil.copy2(helper.source / name, target)


def protect_inputs(repo: Path, roots: Sequence[Path], output: Optional[Path]) -> None:
    """Defence against accidental Python writes, not a sandbox for untrusted code."""
    def forbidden(raw: Any) -> bool:
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return False
        path = Path(os.fsdecode(raw))
        if any(inside(path, root) for root in roots):
            return True
        return inside(path, repo) and (output is None or not inside(path, output))

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        paths: Sequence[Any] = ()
        if event == "open":
            mode, flags = args[1:3]
            writes = isinstance(mode, str) and any(c in mode for c in "wax+")
            writes |= isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            paths = args[:1] if writes else ()
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "shutil.rmtree"):
            paths = args[:1]
        elif event in ("os.rename", "os.replace"):
            paths = args[:2]
        elif event in ("os.symlink", "os.link"):
            paths = args[1:2]
        if any(forbidden(path) for path in paths):
            raise PermissionError(f"Read-only input protection: {event}: {paths}")
    sys.addaudithook(audit)


def validate_paths(args: argparse.Namespace) -> List[Path]:
    if not args.repo.is_dir():
        raise MissingEvidence(f"Repository not found: {args.repo}")
    roots = [p for p in (args.data_root, args.repeat_root, args.xgboost_root) if p is not None]
    for path in roots:
        if not path.is_dir():
            raise MissingEvidence(f"Input directory is missing: {path}")
    out = args.out_root
    if out == args.repo or inside(args.repo, out) or any(inside(out, p) or inside(p, out) for p in roots):
        raise EvidenceError("Output must be a separate tree, not an input directory or its ancestor.")
    if not out.parent.is_dir():
        raise EvidenceError(f"Output parent must already exist: {out.parent}")
    if out.exists():
        if not out.is_dir() or (any(out.iterdir()) and not (out / OWNER).is_file()):
            raise EvidenceError("Nonempty output directory is not owned by this runner. Use a new --out-root.")
        for path in out.rglob("*"):
            if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
                raise EvidenceError(f"Linked output artifact is not allowed: {path}")
    if args.data_root is not None and ((args.data_root / "run_manifest.json").exists() or
            any(re.fullmatch(r"repeat_\d+", p.name) for p in (args.data_root, *args.data_root.parents))):
        raise EvidenceError("--data-root must be ORIGINAL experiments_results, not repetition data.")
    return roots


@contextlib.contextmanager
def output_lock(root: Path):
    import fcntl
    root.mkdir(exist_ok=True)
    with (root / ".xgboost_po.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvidenceError(f"Another PO runner is using {root}.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Evidence:
    def __init__(self) -> None:
        self.small: Dict[str, Dict[str, Any]] = {}
        self.datasets: Dict[str, List[Dict[str, Any]]] = {}

    def record(self, path: Path) -> str:
        if not path.is_file():
            raise MissingEvidence(f"Missing source evidence: {path}")
        st = path.stat()
        prior = self.small.get(str(path))
        if prior:
            if (st.st_size, st.st_mtime_ns) != (prior["bytes"], prior["mtime_ns"]):
                raise EvidenceError(f"Source changed during this run: {path}")
            return prior["sha256"]
        sha = file_hash(path)
        self.small[str(path)] = {"path": str(path), "sha256": sha,
                                 "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        return sha

    def json(self, path: Path) -> Any:
        self.record(path)
        return read_json(path)

    def csv(self, path: Path) -> List[Dict[str, str]]:
        self.record(path)
        return read_csv(path)

    def check_unchanged(self) -> None:
        for path, item in self.small.items():
            if file_hash(Path(path)) != item["sha256"]:
                raise EvidenceError(f"Source evidence changed: {path}")
        for folder, expected in self.datasets.items():
            if physical_inventory(Path(folder)) != expected:
                raise EvidenceError(f"Tensor inventory changed: {folder}")


def physical_inventory(folder: Path) -> List[Dict[str, Any]]:
    result = []
    if folder.is_dir():
        for path in folder.glob("*.pt"):
            st = path.stat()
            result.append({"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns})
    return sorted(result, key=lambda r: natural_key(r["name"]))


def has_pt(path: Path) -> bool:
    return path.is_dir() and next(path.glob("*.pt"), None) is not None


def rebase(raw: Any, root: Path, anchor: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute() and inside(path, root):
        return path.resolve()
    if anchor in path.parts:
        return root.joinpath(*path.parts[path.parts.index(anchor):]).resolve()
    if not path.is_absolute():
        return (root / anchor / "work" / path).resolve()
    return None


def original_summary(archive: Path, step: str, evidence: Evidence) -> Path:
    candidates = []
    for path in sorted((archive / step).rglob("metrics_summary.json")):
        if any((re.fullmatch(r"T\d+", p) and p != "T7") or
               (re.fullmatch(r"h\d+", p) and p != "h7") for p in path.relative_to(archive).parts):
            continue
        val = evidence.json(path)
        if val.get("task") == TASK and int(val.get("config", {}).get("T", -1)) == HISTORY:
            candidates.append(path)
    if not candidates:
        raise MissingEvidence(f"No original PO T7/H7 summary: {archive / step}")
    if len({evidence.record(p) for p in candidates}) > 1:
        raise EvidenceError(f"Multiple different original PO fits for {step}: {candidates}")
    return candidates[0]


def choose_folder(candidates: Sequence[Optional[Path]], root: Path, label: str) -> Path:
    seen = []
    for path in candidates:
        if path is None:
            continue
        path = path.resolve()
        if path in seen:
            continue
        seen.append(path)
        if not inside(path, root):
            raise EvidenceError(f"{label}: source falls outside its correct analysis root: {path}")
        if has_pt(path):
            return path
    raise MissingEvidence(f"Missing EXISTING PO tensors for {label}. No regeneration. Searched: {seen}")


def validate_gnn_config(value: Mapping[str, Any], step: str, label: str) -> Dict[str, Any]:
    cfg = dict(value.get("config", {}))
    if value.get("task") != TASK or int(cfg.get("T", -1)) != 7 or int(cfg.get("sliding_step", -1)) != 1:
        raise EvidenceError(f"{label}: expected saved seven-day task and stride 1.")
    if "split_seed" not in cfg:
        raise EvidenceError(f"{label}: original split seed is missing; not guessed.")
    if step == STEP9 and cfg.get("train_model") not in (False, "false", 0):
        raise EvidenceError(f"{label}: reference Step 9 is not explicitly evaluation-only.")
    return cfg


def old_xgb_tag(step: str, variant: str) -> str:
    if variant == "all_summaries":
        return step
    return {STEP4: "step5_no_admission_turnover", STEP8: "step8_no_admission_turnover",
            STEP9: "step9_no_admission_turnover"}[step]


def checked_parameters(record: Mapping[str, Any], helper: Helpers, label: str,
                       seed: Optional[int] = None, threads: Optional[int] = None) -> Dict[str, Any]:
    params = record.get("parameters")
    if not isinstance(params, dict) or "random_state" not in params or "n_jobs" not in params:
        raise EvidenceError(f"Saved XGBoost parameters missing: {label}")
    expected = helper.xgb.xgb_parameters(int(params["random_state"]), int(params["n_jobs"]))
    if params != expected:
        raise EvidenceError(f"{label}: saved GT recipe differs from its helper; resolve before PO fitting.")
    if seed is not None and int(params["random_state"]) != seed:
        raise EvidenceError(f"{label}: XGBoost seed differs from the repetition manifest.")
    if threads is not None and int(params["n_jobs"]) != threads:
        raise EvidenceError(f"{label}: XGBoost thread setting differs from the saved repetition config.")
    return dict(params)


def check_completion(folder: Path, evidence: Evidence, required: Sequence[str],
                     marker_name: str = "complete.json") -> Dict[str, Any]:
    marker = evidence.json(folder / marker_name)
    hashes = marker.get("files", marker.get("artifacts", {}))
    if not set(required) <= set(hashes):
        raise MissingEvidence(f"Incomplete reference artifact record: {folder / marker_name}")
    for name in required:
        path = folder / name
        if Path(name).name != name or evidence.record(path) != hashes[name]:
            raise EvidenceError(f"Reference artifact no longer matches its completion record: {path}")
    return marker


def check_runtime(path: Path, evidence: Evidence) -> Dict[str, Any]:
    current = importlib.metadata.version("xgboost")
    if not path.is_file():
        return {"current_xgboost": current, "saved_xgboost": None,
                "note": "Historical XGBoost version unavailable; bitwise fit equivalence is not claimed."}
    record = evidence.json(path)
    software = record.get("software", {})
    packages = record.get("packages", software.get("packages", software))
    old = record.get("xgboost_version", packages.get("xgboost"))
    if old is not None and str(old) != current:
        raise EvidenceError(f"Use the original XGBoost environment: saved {old}, current {current}; source {path}")
    return {"current_xgboost": current, "saved_xgboost": old}


def original_plan(args: argparse.Namespace, helper: Helpers, evidence: Evidence) -> Dict[str, Any]:
    root = args.data_root
    work = root / f"TRACK_{MODE}" / "work"
    archive = work / "repro_artifacts_steps_1_9"
    paths = {step: original_summary(archive, step, evidence) for step in STEPS}
    summaries = {step: evidence.json(path) for step, path in paths.items()}
    configs = {step: validate_gnn_config(summaries[step], step, str(paths[step])) for step in STEPS}
    if configs[STEP4]["split_seed"] != configs[STEP9]["split_seed"]:
        raise EvidenceError("Original Step 9 split seed differs from Step 4.")
    mapping = {"canonical_train": (STEP4, "data_folder", helper.driver.BASELINE_TRAIN_REL, "train_folder"),
               "canonical_test": (STEP4, "test_folder", helper.driver.BASELINE_TEST_REL, "test_folder"),
               "shift_train": (STEP8, "data_folder", helper.driver.STEP8_TRAIN_REL, "train_folder"),
               "shift_test": (STEP8, "test_folder", helper.driver.STEP8_TEST_REL, "test_folder")}
    folders = {}
    for kind, (step, field, default, archived) in mapping.items():
        folders[kind] = choose_folder([
            rebase(configs[step].get(field), root, f"TRACK_{MODE}"), work / default,
            paths[step].parent / archived, *sorted((archive / step).rglob(archived))], root, kind)
    # Step 9 must be scored against the same shifted cohort, not another source.
    nine_test = rebase(configs[STEP9].get("test_folder"), root, f"TRACK_{MODE}")
    params, parameter_records = {}, {}
    for step in STEPS:
        params[step] = {}
        for variant in VARIANTS:
            tag = old_xgb_tag(step, variant)
            path = args.xgboost_root / "03_experiments" / "ground_truth" / tag / "metrics.json"
            record = evidence.json(path)
            if record.get("mode") != "ground_truth" or record.get("step") != tag:
                raise EvidenceError(f"Wrong original GT parameter source: {path}")
            task_marker = path.parent / "_task.json"
            if task_marker.is_file():
                check_completion(path.parent, evidence, ["metrics.json"], "_task.json")
            params[step][variant] = checked_parameters(record, helper, str(path))
            parameter_records[f"{step}/{variant}"] = {"path": str(path), "sha256": evidence.record(path)}
    for variant in VARIANTS:
        if params[STEP4][variant] != params[STEP9][variant]:
            raise EvidenceError("Original canonical/zero-shot GT XGBoost recipes differ.")
    ci = evidence.json(args.ci_config)
    required_ci = ("bootstrap_replicates", "bootstrap_seed")
    if not all(k in ci for k in required_ci):
        raise EvidenceError(f"Explicit bootstrap settings missing from {args.ci_config}")
    alias_path = None
    if (archive / EDGE_ALIAS).exists():
        alias_path = original_summary(archive, EDGE_ALIAS, evidence)
    return {"name": "original", "analysis": "original", "repeat": 0,
            "gnn_paths": paths, "configs": configs, "datasets": folders, "params": params,
            "parameter_sources": parameter_records, "nine_recorded_test": nine_test,
            "edge_alias_summary": alias_path,
            "bootstrap": int(ci["bootstrap_replicates"]), "bootstrap_seed": int(ci["bootstrap_seed"]),
            "runtime": check_runtime(args.xgboost_root / "00_provenance" / "run_identity.json", evidence)}


def repeat_manifest(root: Path, evidence: Evidence) -> Dict[str, Any]:
    manifest = evidence.json(root / "run_manifest.json")
    ident = manifest.get("identity", {})
    # The frozen repeat code uses its own canonical JSON digest convention.
    check = hashlib.sha256(json.dumps(ident, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if check != manifest.get("identity_sha256"):
        raise EvidenceError("Repeat manifest identity is inconsistent.")
    if MODE not in ident.get("config", {}).get("tracks", []):
        raise EvidenceError("The reference repeat design has no partial-observation track.")
    reps = ident.get("repeats", [])
    if len({int(r["repeat"]) for r in reps}) != len(reps) or not reps:
        raise EvidenceError("Duplicate or absent repetitions in reference manifest.")
    return manifest


def repeat_plan(args: argparse.Namespace, helper: Helpers, evidence: Evidence,
                manifest: Dict[str, Any], rep: Dict[str, Any]) -> Dict[str, Any]:
    root, number = args.repeat_root, int(rep["repeat"])
    base = root / f"repeat_{number:03d}"
    ready = evidence.json(base / "repeat_complete.json")
    if ready.get("identity_sha256") != manifest["identity_sha256"]:
        raise EvidenceError(f"Wrong completed-repetition identity: {base}")
    if evidence.json(base / "repeat_manifest.json") != rep:
        raise EvidenceError(f"Per-repeat design disagrees with root manifest: {base}")
    cfg = manifest["identity"]["config"]
    paths, configs = {}, {}
    for step in STEPS:
        folder = base / "gnn" / MODE / step
        check_completion(folder, evidence, ["metrics_summary.json", "trained_model.pt",
                                           *[f"{s}_predictions.csv" for s in SPLITS]])
        paths[step] = folder / "metrics_summary.json"
        configs[step] = validate_gnn_config(evidence.json(paths[step]), step, str(paths[step]))
        for key in ("model_seed", "eval_seed"):
            if configs[step].get(key) != rep[key]:
                raise EvidenceError(f"{paths[step]}: {key} differs from repetition manifest.")
        if configs[step]["split_seed"] != cfg["split_seed"]:
            raise EvidenceError(f"{paths[step]}: split seed differs from saved design.")
    if evidence.record(paths[STEP4].parent / "trained_model.pt") != evidence.record(paths[STEP9].parent / "trained_model.pt"):
        raise EvidenceError(f"{base}: saved GNN Step 9 is not the canonical checkpoint.")
    work = base / f"TRACK_{MODE}" / "work"
    mapping = {"canonical_train": (STEP4, "data_folder", helper.driver.BASELINE_TRAIN_REL),
               "canonical_test": (STEP4, "test_folder", helper.driver.BASELINE_TEST_REL),
               "shift_train": (STEP8, "data_folder", helper.driver.STEP8_TRAIN_REL),
               "shift_test": (STEP8, "test_folder", helper.driver.STEP8_TEST_REL)}
    folders = {}
    for kind, (step, field, default) in mapping.items():
        folders[kind] = choose_folder([rebase(configs[step].get(field), root, base.name), work / default], root, kind)
    params, parameter_records = {}, {}
    for step in STEPS:
        params[step] = {}
        for variant in VARIANTS:
            folder = base / "xgb" / step / variant
            check_completion(folder, evidence, ["metrics.json"])
            path = folder / "metrics.json"
            record = evidence.json(path)
            if record.get("mode") != "ground_truth" or record.get("step") != old_xgb_tag(step, variant):
                raise EvidenceError(f"Wrong reference GT XGBoost parameter record: {path}")
            params[step][variant] = checked_parameters(record, helper, str(path), int(rep["model_seed"]), int(cfg["threads"]))
            parameter_records[f"{step}/{variant}"] = {"path": str(path), "sha256": evidence.record(path)}
    return {"name": base.name, "analysis": "repeat", "repeat": number, "rep": rep,
            "gnn_paths": paths, "configs": configs, "datasets": folders, "params": params,
            "parameter_sources": parameter_records, "nine_recorded_test": None,
            "edge_alias_summary": None, "bootstrap": int(cfg["bootstrap_replicates"]),
            "bootstrap_seed": int(cfg["bootstrap_seed"]) + number,
            "runtime": check_runtime(root / "runtime.json", evidence)}


def repeat_file_expectations(plan: Dict[str, Any], kind: str, args: argparse.Namespace,
                             helper: Helpers, evidence: Evidence) -> Dict[str, Dict[str, Any]]:
    regime, split = kind.rsplit("_", 1)
    jobs = [j for j in plan["rep"]["jobs"] if (j["regime"], j["split"]) == (regime, split)]
    members, expected = [], {}
    for job in jobs:
        path = args.repeat_root / "data_cache" / job["fingerprint"][:24] / MODE / "pt_complete.json"
        info = evidence.json(path)
        ident = {"job": job["fingerprint"], "mode": MODE, "converter_sha256": helper.hashes["convert_to_pt.py"]}
        if info.get("identity") != ident or len(info.get("files", [])) != int(job["days"]):
            raise EvidenceError(f"PO source cache identity mismatch: {path}")
        members.append({"sim_id": job["sim_id"], "semantic_sha256": info["semantic_sha256"]})
        for item in info["files"]:
            name = item["name"]
            if Path(name).name != name or name in expected:
                raise EvidenceError(f"Duplicate/unsafe tensor filename in {path}")
            expected[name] = {**item, "sim_id": job["sim_id"]}
    pooled = evidence.json(plan["datasets"][kind] / "dataset_complete.json")
    if (not jobs or pooled.get("identity") != helper.repeats.digest(members) or
            pooled.get("trajectories") != members or pooled.get("daily_graphs") != len(expected)):
        raise EvidenceError(f"Wrong pooled PO dataset: {plan['datasets'][kind]}")
    return expected


def canonical_names(folder: Path, inventory: List[Dict[str, Any]], evidence: Evidence) -> Dict[str, str]:
    names = {r["name"]: r["name"] for r in inventory}
    path = folder / "manifest.csv"
    if path.is_file():
        rows = evidence.csv(path)
        if rows and "archived_name" in rows[0]:
            mapped = {r["archived_name"]: Path(r["relative_source_path"]).name for r in rows
                      if r.get("archived_name", "").endswith(".pt")}
            if set(mapped) != set(names) or len(set(mapped.values())) != len(mapped):
                raise EvidenceError(f"Cannot restore original filename order: {path}")
            names = mapped
    return names


def validate_po_graph(graph: Any, label: str) -> Tuple[float, frozenset]:
    import numpy as np
    x = getattr(graph, "x", None)
    names = getattr(graph, "node_names", None)
    if x is None or names is None:
        raise EvidenceError(f"Missing PO x or patient identifiers: {label}")
    x = x.detach().cpu().numpy()
    names = list(map(str, names))
    if x.ndim != 2 or x.shape != (len(names), 6) or len(set(names)) != len(names):
        raise EvidenceError(f"Expected original six PO columns and unique aligned node_names: {label}")
    if not np.isfinite(x).all() or not np.isin(x[:, 0:2], [0, 1]).all():
        raise EvidenceError(f"Non-binary PO role/observed-status values or invalid features: {label}")
    if np.any(x[:, 4:6] != 0):
        raise EvidenceError(f"Incident-event channels are not masked: {label}. Refusing possible GT leakage.")
    patient = x[:, 0] == 0
    return float(x[patient, 1].sum()), frozenset(names[i] for i in np.flatnonzero(patient))


def extract_dataset(plan: Dict[str, Any], kind: str, args: argparse.Namespace,
                    helper: Helpers, evidence: Evidence) -> Dict[str, Any]:
    """Read tensors once; cache only derived features/metadata under the new output."""
    import numpy as np
    import torch
    folder = plan["datasets"][kind]
    inventory = physical_inventory(folder)
    if not inventory:
        raise MissingEvidence(f"No existing PO tensors: {folder}")
    evidence.datasets[str(folder)] = inventory
    expected = repeat_file_expectations(plan, kind, args, helper, evidence) if plan["repeat"] else {}
    if expected and set(expected) != {r["name"] for r in inventory}:
        raise EvidenceError(f"Missing/extra tensors relative to repeat manifest: {folder}")
    names = canonical_names(folder, inventory, evidence)
    key = digest({"folder": str(folder), "inventory": inventory, "names": names,
                  "expected": expected, "contract": FEATURE_CONTRACT, "script": file_hash(Path(__file__))})
    cache = args.out_root / "feature_cache" / key
    marker = cache / "complete.json"
    if marker.is_file():
        saved = read_json(marker)
        if saved.get("fingerprint") != key:
            raise EvidenceError(f"Feature-cache identity changed: {cache}")
        for name, sha in saved["files"].items():
            if Path(name).name != name or file_hash(cache / name) != sha:
                raise EvidenceError(f"Feature cache changed: {cache / name}")
        audit = read_json(cache / "feature_audit.json")
        recorded = {r["name"]: r for r in audit["source_files"]}
        for item in inventory:
            if expected and (item["bytes"] != expected[item["name"]]["bytes"] or
                             recorded[item["name"]]["sha256"] != expected[item["name"]]["sha256"]):
                raise EvidenceError(f"Repeat-source cache mismatch: {folder / item['name']}")
            if args.verify_data_hashes and file_hash(folder / item["name"]) != recorded[item["name"]]["sha256"]:
                raise EvidenceError(f"Source tensor changed: {folder / item['name']}")
        with np.load(cache / "features.npz", allow_pickle=False) as data:
            X, y = data["X"], data["y"]
        samples = read_json(cache / "samples.json")
        print(f"  [reuse features] {plan['name']}/{kind}: {len(samples)} windows", flush=True)
        return {"X": X, "y": y, "rows": samples, "audit": audit, "identity": audit["data_identity"]}
    print(f"  [read PO tensors] {plan['name']}/{kind}: {len(inventory)} daily graphs", flush=True)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    hashes = []
    order = sorted(inventory, key=lambda r: natural_key(names[r["name"]]))
    for index, item in enumerate(order, 1):
        path = folder / item["name"]
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if len(payload) != item["bytes"]:
            raise EvidenceError(f"Tensor changed during extraction: {path}")
        if expected and sha != expected[item["name"]]["sha256"]:
            raise EvidenceError(f"PO tensor is not the one used in the repetition: {path}")
        graph = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        sid = str(getattr(graph, "sim_id", ""))
        day = integer(getattr(graph, "day", None), "day")
        y = integer(getattr(graph, "y_h7_endog_majority", None), "saved label")
        if not sid.strip() or y not in (0, 1):
            raise EvidenceError(f"Missing/invalid saved trajectory identity or target: {path}")
        if expected and (sid != expected[item["name"]]["sim_id"] or day != expected[item["name"]]["day"]):
            raise EvidenceError(f"PO tensor metadata differs from its reference manifest: {path}")
        observed, patients = validate_po_graph(graph, str(path))
        groups.setdefault(sid, []).append({"day": day, "file": names[item["name"]],
                                          "y": y, "observed": observed, "patients": patients})
        hashes.append({**item, "original_name": names[item["name"]], "sha256": sha, "sim_id": sid, "day": day})
        del graph, payload
        if index % 1000 == 0:
            print(f"    read {index}/{len(inventory)}", flush=True)
    features, samples, daily = [], [], []
    for sid, days in groups.items():
        days.sort(key=lambda r: r["day"])
        seq = [r["day"] for r in days]
        if len(set(seq)) != len(seq) or any(b != a + 1 for a, b in zip(seq, seq[1:])):
            raise EvidenceError(f"Duplicate/noncontiguous days in {folder}/{sid}")
        previous, values = None, []
        for row in days:
            current = row.pop("patients")
            admissions = float(len(current - previous)) if previous is not None else float("nan")
            discharges = float(len(previous - current)) if previous is not None else float("nan")
            values.append([row["observed"], admissions, discharges, 0.0])
            daily.append({"sim_id": sid, "day": row["day"], **dict(zip(DAILY_FEATURES, values[-1]))})
            previous = current
        for start in range(len(days) - HISTORY + 1):
            last = days[start + HISTORY - 1]
            features.append(np.asarray(values[start:start + HISTORY], dtype=np.float32).reshape(-1))
            samples.append({"sim_id": sid, "day": last["day"], "start_day": days[start]["day"],
                            "last_file": last["file"], "y": last["y"]})
    if not samples:
        raise EvidenceError(f"No original eligible seven-day windows: {folder}")
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray([r["y"] for r in samples], dtype=np.int64)
    if X.shape != (len(samples), 28) or np.any(X[:, 3::4] != 0):
        raise EvidenceError("Internal PO feature construction failed.")
    if physical_inventory(folder) != inventory:
        raise EvidenceError(f"Tensor folder changed during extraction: {folder}")
    data_identity = digest({"source_files": hashes, "window_keys": samples, "feature_contract": FEATURE_CONTRACT})
    audit = {"source": str(folder), "analysis": plan["analysis"], "repeat": plan["repeat"],
             "track": MODE, "kind": kind, "source_files": hashes, "contract": FEATURE_CONTRACT,
             "window_count": len(samples), "trajectories": len(groups), "columns": 28,
             "column_order": [f"lag_{lag}_{name}" for lag in range(6, -1, -1) for name in DAILY_FEATURES],
             "data_identity": data_identity, "feature_array_hash": helper.xgb.array_hash(X, y),
             "verification": "all tensors read and hashed; repeat tensors checked against completed PO manifests" if expected
                             else "original saved PO tensors; original natural-file/group/day order reconstructed"}
    if not args.preflight:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / "features.tmp.npz").open("wb") as stream:
            np.savez_compressed(stream, X=X, y=y)
        os.replace(cache / "features.tmp.npz", cache / "features.npz")
        atomic_json(cache / "samples.json", samples)
        atomic_json(cache / "feature_audit.json", audit)
        write_csv(cache / "daily_features.csv", daily)
        artifacts = {name: file_hash(cache / name) for name in
                     ("features.npz", "samples.json", "feature_audit.json", "daily_features.csv")}
        atomic_json(marker, {"fingerprint": key, "files": artifacts})
    return {"X": X, "y": y, "rows": samples, "audit": audit, "identity": data_identity}


def window_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row["sim_id"]), integer(row["day"], "prediction day")


def aligned_predictions(raw: Sequence[Mapping[str, Any]], expected: List[Dict[str, Any]],
                        label: str) -> List[Dict[str, Any]]:
    lookup = {window_key(row): row for row in raw}
    keys = [window_key(r) for r in expected]
    if len(lookup) != len(raw) or len(set(keys)) != len(keys) or set(lookup) != set(keys):
        raise EvidenceError(f"{label}: prediction trajectory/day identities do not match the GNN dataset.")
    result = []
    for item in expected:
        row = lookup[window_key(item)]
        if integer(row["y"], "label") != item["y"] or integer(row["start_day"], "history start") != item["start_day"]:
            raise EvidenceError(f"{label}: labels or seven-day histories differ.")
        if row.get("last_file") and Path(str(row["last_file"])).name != item["last_file"]:
            raise EvidenceError(f"{label}: source graph filenames differ.")
        probability = float(row.get("probability", row.get("probability_class_1")))
        pred = integer(row["prediction"], "prediction")
        if not math.isfinite(probability) or not 0 <= probability <= 1 or pred != int(probability > .5):
            raise EvidenceError(f"{label}: invalid probability or altered classification threshold.")
        result.append({**item, "probability": probability, "prediction": pred})
    return result


def verify_points(rows: List[Dict[str, Any]], block: Mapping[str, Any],
                  stats: Any, label: str) -> Dict[str, Any]:
    import numpy as np
    y = [r["y"] for r in rows]
    p = [r["probability"] for r in rows]
    actual = stats.binary_metrics(y, p)
    saved = block.get("metrics", {})
    for metric in METRICS:
        key = {"auroc": "roc_auc", "macro_f1": "f1_macro"}.get(metric, metric)
        if key not in saved and metric in ("sensitivity", "specificity") and "confusion_matrix" in block:
            continue
        if key not in saved:
            raise EvidenceError(f"{label}: no saved {key} for point-estimate verification.")
        left, right = saved[key], actual[metric]
        undefined = left is None or (isinstance(left, float) and not math.isfinite(left))
        if undefined:
            if right is not None:
                raise EvidenceError(f"{label}: {metric} no longer agrees with its original undefined value.")
        elif right is None or not math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-9):
            raise EvidenceError(f"{label}: saved {metric}={left}, recalculated={right}.")
    if "confusion_matrix" in block:
        cm = [[actual["tn"], actual["fp"]], [actual["fn"], actual["tp"]]]
        if not np.array_equal(block["confusion_matrix"], cm):
            raise EvidenceError(f"{label}: confusion counts differ from saved results.")
    if "n_samples" in block and int(block["n_samples"]) != len(rows):
        raise EvidenceError(f"{label}: saved sample count differs.")
    return actual


def gnn_predictions(path: Path, expected: Mapping[str, List[Dict[str, Any]]],
                    helper: Helpers, evidence: Evidence, require_train: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
    import numpy as np
    saved = evidence.json(path)
    output, notes = {}, {}
    for split in SPLITS:
        block = saved.get(split)
        if block is None and split == "train" and not require_train:
            notes[split] = "Original fitted-training predictions were not archived; validation is not relabelled."
            continue
        if not isinstance(block, dict):
            raise MissingEvidence(f"Missing saved GNN {split} block: {path}")
        csv_path = path.parent / f"{split}_predictions.csv"
        if csv_path.is_file():
            rows = aligned_predictions(evidence.csv(csv_path), expected[split], str(csv_path))
            notes[split] = "Exact identifiers, history windows and labels matched to saved prediction CSV."
        else:
            y = block.get("y_true")
            probabilities = np.asarray(block.get("probabilities", []), dtype=float)
            pred = block.get("y_pred")
            if y != [r["y"] for r in expected[split]] or probabilities.shape != (len(expected[split]), 2):
                raise EvidenceError(f"{path}/{split}: original JSON labels/order differ from reconstructed windows.")
            if not isinstance(pred, list) or len(pred) != len(y):
                raise MissingEvidence(f"Missing saved GNN decisions: {path}/{split}")
            raw = [{**r, "probability": float(p[1]), "prediction": d}
                   for r, p, d in zip(expected[split], probabilities, pred)]
            rows = aligned_predictions(raw, expected[split], f"{path}/{split}")
            notes[split] = "Historical natural-file/group/day order reconstructed from original PO tensors; all saved labels/metrics verified."
        if "y_true" in block:
            probabilities = np.asarray(block.get("probabilities", []), dtype=float)
            if ([r["y"] for r in rows] != block["y_true"] or
                    [r["prediction"] for r in rows] != block.get("y_pred") or
                    probabilities.shape != (len(rows), 2) or
                    not np.allclose(probabilities[:, 1], [r["probability"] for r in rows], rtol=1e-7, atol=1e-9) or
                    not np.array_equal(probabilities.argmax(axis=1), [r["prediction"] for r in rows])):
                raise EvidenceError(f"GNN CSV/JSON/order discrepancy: {path}/{split}")
        verify_points(rows, block, helper.stats, f"{path}/{split}")
        output[split] = rows
    return output, notes


def partitions(train: Dict[str, Any], test: Dict[str, Any], seed: int,
               helper: Helpers) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    import numpy as np
    ti, vi = helper.xgb.original_split(train["rows"], seed)
    by_split = {"train": (train, ti), "validation": (train, vi),
                "test": (test, np.arange(len(test["rows"]))) }
    expected = {s: [data["rows"][int(i)] for i in idx] for s, (data, idx) in by_split.items()}
    ids = {s: {r["sim_id"] for r in rows} for s, rows in expected.items()}
    if ids["train"] & ids["validation"] or (ids["train"] | ids["validation"]) & ids["test"]:
        raise EvidenceError("Trajectory overlap across training/validation/test partitions.")
    if set(train["y"][ti].tolist()) != {0, 1}:
        raise EvidenceError("Training partition has one class; it is not redrawn or relabelled.")
    return by_split, expected


def prepare_context(plan: Dict[str, Any], args: argparse.Namespace, helper: Helpers,
                    evidence: Evidence) -> Dict[str, Any]:
    datasets = {kind: extract_dataset(plan, kind, args, helper, evidence) for kind in KINDS}
    by_step, expected_by_step, references, notes = {}, {}, {}, {}
    for step in STEPS:
        train = datasets["shift_train" if step == STEP8 else "canonical_train"]
        test = datasets["canonical_test" if step == STEP4 else "shift_test"]
        parts, expected = partitions(train, test, int(plan["configs"][step]["split_seed"]), helper)
        refs, checks = gnn_predictions(plan["gnn_paths"][step], expected, helper, evidence, bool(plan["repeat"]))
        by_step[step], expected_by_step[step], references[step], notes[step] = parts, expected, refs, checks
    # Saved Step 8/9 labels and identifiers must point to the same shifted test set.
    if expected_by_step[STEP8]["test"] != expected_by_step[STEP9]["test"]:
        raise EvidenceError("Step 8 and Step 9 do not share the shifted test cohort.")
    for split in ("train", "validation"):
        if expected_by_step[STEP4][split] != expected_by_step[STEP9][split]:
            raise EvidenceError("Step 9 does not retain the canonical source split.")
        if plan["repeat"]:
            import numpy as np
            left, right = references[STEP4][split], references[STEP9][split]
            if ([r["prediction"] for r in left] != [r["prediction"] for r in right] or
                    not np.allclose([r["probability"] for r in left], [r["probability"] for r in right],
                                    rtol=1e-6, atol=1e-8)):
                raise EvidenceError("Repeated GNN Step 9 source predictions differ from Step 4.")
    alias_verified = False
    if plan["edge_alias_summary"]:
        alias = evidence.json(plan["edge_alias_summary"])
        alias_cfg = validate_gnn_config(alias, EDGE_ALIAS, str(plan["edge_alias_summary"]))
        if alias_cfg["split_seed"] != plan["configs"][STEP4]["split_seed"]:
            raise EvidenceError("No-edge-weight XGBoost alias would use a different validation split.")
        gnn_predictions(plan["edge_alias_summary"], expected_by_step[STEP4], helper, evidence, False)
        alias_verified = True
    plan["alias_verified"] = alias_verified
    dataset_info = {k: {"path": str(plan["datasets"][k]), "identity": d["identity"],
                        "n_windows": len(d["rows"]), "n_trajectories": d["audit"]["trajectories"]}
                    for k, d in datasets.items()}
    info = {"name": plan["name"], "analysis": plan["analysis"], "repeat": plan["repeat"], "track": MODE,
            "datasets": dataset_info, "reference_metrics_sha256": {k: evidence.record(p) for k, p in plan["gnn_paths"].items()},
            "reference_prediction_alignment": notes,
            "reference_prediction_digests": {s: {k: digest(v) for k, v in rows.items()} for s, rows in references.items()},
            "split_seeds": {s: plan["configs"][s]["split_seed"] for s in STEPS},
            "parameters": plan["params"], "parameter_sources": plan["parameter_sources"],
            "bootstrap_replicates": plan["bootstrap"], "bootstrap_seed": plan["bootstrap_seed"],
            "repeat_design": plan.get("rep"), "helper_hashes": helper.hashes, "runtime": plan["runtime"],
            "original_no_edge_alias_verified": alias_verified,
            "original_gnn_seed_note": "Historical GNN model/evaluation seeds may be absent; saved GNN predictions are used without inference.",
            "feature_contract": FEATURE_CONTRACT}
    if not args.preflight:
        dest = args.out_root / plan["name"]
        old = dest / "comparison_manifest.json"
        if old.exists() and read_json(old) != clean(info):
            raise EvidenceError(f"Previously prepared input comparison changed: {old}. Use a new output folder.")
        atomic_json(old, info)
        for kind, dataset in datasets.items():
            atomic_json(args.out_root / "dataset_audits" / plan["name"] / f"{kind}.json", dataset["audit"])
        for step, values in references.items():
            for split, rows in values.items():
                write_csv(dest / "gnn_reference" / step / f"{split}_predictions.csv", rows)
    return {"plan": plan, "datasets": datasets, "parts": by_step, "expected": expected_by_step,
            "references": references, "manifest": info}


def selected_features(data: Dict[str, Any], variant: str) -> Any:
    import numpy as np
    if variant == "all_summaries":
        return data["X"]
    # Preserve the same 28 -> 14 column selection as the GT experiment.
    keep = [day * 4 + column for day in range(HISTORY) for column in (0, 3)]
    return np.ascontiguousarray(data["X"][:, keep])


def owned_task_complete(folder: Path, fingerprint: str) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    marker = read_json(path)
    if marker.get("fingerprint") != fingerprint:
        raise EvidenceError(f"Completed PO task has different inputs/settings: {folder}")
    required = {"metrics.json", "model.json", *[f"{s}_predictions.csv" for s in SPLITS]}
    if not required <= set(marker.get("files", {})):
        raise EvidenceError(f"Incomplete/corrupt completion marker: {path}")
    for name, sha in marker["files"].items():
        if Path(name).name != name or not (folder / name).is_file() or file_hash(folder / name) != sha:
            raise EvidenceError(f"Completed PO output changed: {folder / name}")
    return True


def fit_context(context: Dict[str, Any], args: argparse.Namespace, helper: Helpers) -> int:
    import numpy as np
    plan, fitted = context["plan"], 0
    for step in STEPS:
        for variant in VARIANTS:
            params = dict(plan["params"][step][variant])
            if args.threads is not None:
                params["n_jobs"] = args.threads  # Execution setting only; recorded explicitly.
            folder = args.out_root / plan["name"] / "xgb" / step / variant
            source = args.out_root / plan["name"] / "xgb" / STEP4 / variant / "model.json" if step == STEP9 else None
            feature_parts = {}
            for split, (data, idx) in context["parts"][step].items():
                feature_parts[split] = (selected_features(data, variant)[idx], data["y"][idx], context["expected"][step][split])
            input_hashes = {s: helper.xgb.array_hash(X, y) for s, (X, y, _) in feature_parts.items()}
            source_hash = file_hash(source) if source is not None else None
            identity = {"comparison": digest(context["manifest"]), "step": step, "variant": variant,
                        "parameters": params, "arrays": input_hashes, "source_checkpoint": source_hash,
                        "recipe_sha256": helper.hashes["experiments_xgboost.py"]}
            fp = digest(identity)
            if owned_task_complete(folder, fp):
                print(f"  [reuse fit/evaluation] {plan['name']}/{step}/{variant}", flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            model = helper.xgb.XGBoostBinaryModel(**params)
            model_path, ready_path = folder / "model.json", folder / "model_ready.json"
            if ready_path.exists():
                record = read_json(ready_path)
                if record.get("fingerprint") != fp or not model_path.is_file() or file_hash(model_path) != record["model_sha256"]:
                    raise EvidenceError(f"Interrupted PO model has inconsistent inputs or contents: {folder}")
                model.load_model(model_path)
                print(f"  [reuse checkpoint] {plan['name']}/{step}/{variant}", flush=True)
            elif source is not None:
                model.load_model(source)
                shutil.copy2(source, model_path)
                print(f"  [zero-shot only] {plan['name']}/{variant}", flush=True)
            else:
                print(f"  [fit XGBoost] {plan['name']}/{step}/{variant}: 150 rounds", flush=True)
                X, y, _ = feature_parts["train"]
                model.fit(X, y)
                temp = folder / "model.tmp.json"
                model.save_model(temp)
                os.replace(temp, model_path)
                fitted += 1
            model_hash = file_hash(model_path)
            if source_hash is not None and model_hash != source_hash:
                raise EvidenceError("Step 9 changed the canonical XGBoost checkpoint.")
            atomic_json(ready_path, {"fingerprint": fp, "model_sha256": model_hash,
                                    "source_model": str(source) if source else None})
            result = {"track": MODE, "analysis": plan["analysis"], "repeat": plan["repeat"],
                      "step": step, "variant": variant, "feature_width": 28 if variant == "all_summaries" else 14,
                      "parameters": params, "matched_GT_parameters": plan["params"][step][variant],
                      "fit_status": "zero_shot_baseline_model" if source else "trained_PO_XGBoost",
                      "input_array_sha256": input_hashes, "source_model_sha256": source_hash,
                      "model_sha256": model_hash, "feature_contract": FEATURE_CONTRACT,
                      "reference_comparison_sha256": digest(context["manifest"])}
            for split, (X, y, samples) in feature_parts.items():
                if step == STEP9 and split != "test":
                    rows = aligned_predictions(read_csv(source.parent / f"{split}_predictions.csv"), samples,
                                               f"canonical PO XGBoost {split}")
                else:
                    probability = model.predict_proba(X)[:, 1]
                    raw = [{**r, "probability": float(p), "prediction": int(p > .5)} for r, p in zip(samples, probability)]
                    rows = aligned_predictions(raw, samples, f"{plan['name']}/{step}/{variant}/{split}")
                write_csv(folder / f"{split}_predictions.csv", rows)
                result[split] = helper.stats.binary_metrics(y, [r["probability"] for r in rows])
            if file_hash(model_path) != model_hash or (source is not None and file_hash(source) != source_hash):
                raise EvidenceError("Model parameters changed during evaluation.")
            atomic_json(folder / "metrics.json", result)
            artifacts = {name: file_hash(folder / name) for name in
                         ("model.json", "metrics.json", "model_ready.json", *[f"{s}_predictions.csv" for s in SPLITS])}
            atomic_json(folder / "complete.json", {"fingerprint": fp, "identity": identity, "files": artifacts})
    return fitted


def bootstrap_input(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    ids = sorted({r["sim_id"] for r in rows})
    if all("__sim_" in sid and sid.split("__sim_", 1)[0] for sid in ids):
        return rows, "within recorded simulator family"
    # Do not incorrectly stratify legacy one-off IDs as separate families.
    mapping = {sid: f"cohort__sim_{i:09d}" for i, sid in enumerate(ids)}
    return [{**r, "sim_id": mapping[r["sim_id"]]} for r in rows], "unstratified; family identifiers unavailable"


def joint_bootstrap(stats: Any, rows: List[Dict[str, Any]], probabilities: Dict[str, Any],
                    repetitions: int, seed: int) -> Dict[str, Any]:
    """Reference trajectory sampling; reuse each metric calculation for paired contrasts."""
    import numpy as np
    ci_rows, scheme = bootstrap_input(rows)
    y = np.asarray([r["y"] for r in rows], dtype=int)
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(ci_rows):
        groups.setdefault(r["sim_id"], []).append(i)
    families: Dict[str, List[str]] = {}
    for sid in sorted(groups):
        families.setdefault(sid.split("__sim_", 1)[0], []).append(sid)
    values = {model: {m: [] for m in METRICS} for model in probabilities}
    pairs = [("GNN", model) for model in ("XGB_all", "XGB_no_counts") if "GNN" in probabilities and model in probabilities]
    differences = {b: {m: [] for m in METRICS} for _, b in pairs}
    # Identical probability vectors share calculation, but remain separate table rows.
    unique, aliases = {}, {}
    for model, p in probabilities.items():
        p = np.asarray(p, dtype=float)
        key = hashlib.sha256(p.tobytes()).hexdigest()
        if key not in unique:
            unique[key] = p
        aliases[model] = key
    rng = np.random.default_rng(seed)
    if len(groups) >= 2:
        for _ in range(repetitions):
            sampled = []
            for family in sorted(families):
                ids = families[family]
                sampled.extend(rng.choice(ids, len(ids), replace=True).tolist())
            idx = np.asarray([i for sid in sampled for i in groups[sid]], dtype=int)
            points = {key: stats.binary_metrics(y[idx], p[idx]) for key, p in unique.items()}
            for model, key in aliases.items():
                for metric in METRICS:
                    if points[key][metric] is not None:
                        values[model][metric].append(points[key][metric])
            for a, b in pairs:
                for metric in METRICS:
                    left, right = points[aliases[a]][metric], points[aliases[b]][metric]
                    if left is not None and right is not None:
                        differences[b][metric].append(left - right)
    def limits(item: Dict[str, List[float]]) -> Dict[str, Any]:
        return {m: {"low": float(np.quantile(v, .025)) if v else None,
                    "high": float(np.quantile(v, .975)) if v else None,
                    "valid_bootstraps": len(v), "requested_bootstraps": repetitions,
                    "n_trajectories": len(groups), "n_families": len(families)} for m, v in item.items()}
    return {"marginal": {name: limits(v) for name, v in values.items()},
            "paired": {name: limits(v) for name, v in differences.items()},
            "stratification": scheme, "seed": seed, "repetitions": repetitions,
            "method": "95% percentile whole-trajectory bootstrap, conditional on fixed fits"}


def cached_intervals(root: Path, helper: Helpers, rows: List[Dict[str, Any]],
                     probabilities: Dict[str, Any], count: int, seed: int) -> Dict[str, Any]:
    import numpy as np
    key = digest({"windows": [(r["sim_id"], r["day"], r["start_day"], r["y"]) for r in rows],
                  "probabilities": {k: helper.xgb.array_hash(np.asarray(p, dtype=float)) for k, p in probabilities.items()},
                  "count": count, "seed": seed, "stats": helper.hashes["summarize_predictive_repeats.py"],
                  "implementation": VERSION})
    path = root / "interval_cache" / f"{key}.json"
    if path.is_file():
        saved = read_json(path)
        if saved.get("fingerprint") != key or digest(saved["result"]) != saved.get("result_sha256"):
            raise EvidenceError(f"Changed interval cache: {path}")
        return saved["result"]
    result = joint_bootstrap(helper.stats, rows, probabilities, count, seed)
    atomic_json(path, {"fingerprint": key, "result_sha256": digest(result), "result": result})
    return result


def loaded_task(folder: Path) -> Dict[str, Any]:
    marker = read_json(folder / "complete.json")
    if not owned_task_complete(folder, marker["fingerprint"]):
        raise MissingEvidence(f"PO comparison is not complete: {folder}")
    return read_json(folder / "metrics.json")


def summarize(root: Path, helpers: Dict[str, Helpers], expected_names: Sequence[str]) -> Dict[str, Any]:
    individual, conditional, paired, missing = [], [], [], []
    finished = []
    original_alias = False
    for name in expected_names:
        folder = root / name
        manifest_path = folder / "comparison_manifest.json"
        if not manifest_path.is_file():
            missing.append(name)
            continue
        meta = read_json(manifest_path)
        helper = helpers[meta["analysis"]]
        try:
            saved = {(step, variant): loaded_task(folder / "xgb" / step / variant)
                     for step in STEPS for variant in VARIANTS}
        except MissingEvidence:
            missing.append(name)
            continue
        print(f"[intervals] {name}: all five metrics, matched model resamples", flush=True)
        for step in STEPS:
            for split in SPLITS:
                first_path = folder / "xgb" / step / "all_summaries" / f"{split}_predictions.csv"
                raw = read_csv(first_path)
                expected = [{"sim_id": r["sim_id"], "day": int(r["day"]), "start_day": int(r["start_day"]),
                             "last_file": r["last_file"], "y": int(r["y"])} for r in raw]
                model_rows = {}
                ref = folder / "gnn_reference" / step / f"{split}_predictions.csv"
                if ref.is_file():
                    model_rows["GNN"] = aligned_predictions(read_csv(ref), expected, str(ref))
                    if digest(model_rows["GNN"]) != meta["reference_prediction_digests"][step][split]:
                        raise EvidenceError(f"Copied GNN reference predictions changed: {ref}")
                elif split in meta["reference_prediction_digests"][step]:
                    raise EvidenceError(f"Copied GNN reference predictions are missing: {ref}")
                for variant, model in zip(VARIANTS, ("XGB_all", "XGB_no_counts")):
                    path = folder / "xgb" / step / variant / f"{split}_predictions.csv"
                    model_rows[model] = aligned_predictions(read_csv(path), expected, str(path))
                probabilities = {model: [r["probability"] for r in rows] for model, rows in model_rows.items()}
                results = {model: helper.stats.binary_metrics([r["y"] for r in expected], p) for model, p in probabilities.items()}
                for variant, model in zip(VARIANTS, ("XGB_all", "XGB_no_counts")):
                    if results[model] != saved[step, variant][split]:
                        raise EvidenceError(f"PO stored metrics disagree with saved predictions: {name}/{step}/{variant}/{split}")
                ci = cached_intervals(root, helper, expected, probabilities,
                                      meta["bootstrap_replicates"], meta["bootstrap_seed"])
                common = {"analysis": meta["analysis"], "repeat": meta["repeat"], "mode": MODE, "step": step,
                          "split": split, "n_trajectories": len({r["sim_id"] for r in expected}),
                          "bootstrap_seed": meta["bootstrap_seed"]}
                if meta["repeat_design"] is not None:
                    rep = meta["repeat_design"]
                    common.update(model_seed=rep["model_seed"], parameter_seed=rep["parameter_seed"],
                                  **{f"multiplier_{k}": v for k, v in rep["parameter_multipliers"].items()})
                for model, point in results.items():
                    individual.append({**common, "model": model, **point})
                    for metric in METRICS:
                        conditional.append({**common, "model": model, "metric": metric, "point": point[metric],
                                            **ci["marginal"][model][metric], "stratification": ci["stratification"],
                                            "ci_method": ci["method"]})
                for model, intervals in ci["paired"].items():
                    for metric in METRICS:
                        a, b = results["GNN"][metric], results[model][metric]
                        paired.append({**common, "comparison": f"GNN minus {model}", "metric": metric,
                                       "point": a - b if a is not None and b is not None else None,
                                       **intervals[metric], "stratification": ci["stratification"], "ci_method": ci["method"]})
        if name == "original":
            original_alias = bool(meta["original_no_edge_alias_verified"])
        finished.append(name)
    repeat_rows = [r for r in individual if r["analysis"] == "repeat"]
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in repeat_rows:
        grouped.setdefault((row["step"], row["model"], row["split"]), []).append(row)
    expected_repeats = len([name for name in expected_names if name != "original"])
    means, repeat_pairs = [], []
    for (step, model, split), rows in grouped.items():
        rows.sort(key=lambda r: r["repeat"])
        for metric in METRICS:
            means.append({"mode": MODE, "step": step, "model": model, "split": split, "metric": metric,
                          "n_requested": expected_repeats, "run_complete": len(rows) == expected_repeats,
                          **helpers["repeat"].stats.mean_interval([r[metric] for r in rows])})
    for step in STEPS:
        for split in SPLITS:
            gnn = {r["repeat"]: r for r in grouped.get((step, "GNN", split), [])}
            for model in ("XGB_all", "XGB_no_counts"):
                xgb = {r["repeat"]: r for r in grouped.get((step, model, split), [])}
                common = sorted(set(gnn) & set(xgb))
                if not common:
                    continue
                for metric in METRICS:
                    differences = [gnn[i][metric] - xgb[i][metric]
                                   if gnn[i][metric] is not None and xgb[i][metric] is not None else None for i in common]
                    repeat_pairs.append({"mode": MODE, "step": step, "split": split,
                                         "comparison": f"GNN minus {model}", "metric": metric,
                                         "n_requested": expected_repeats, "run_complete": len(common) == expected_repeats,
                                         **helpers["repeat"].stats.mean_interval(differences)})
    dest = root / "summary"
    write_csv(dest / "per_run_metrics.csv", individual)
    write_csv(dest / "original_po_metrics_ci.csv", [r for r in conditional if r["analysis"] == "original"])
    write_csv(dest / "repeat_conditional_metrics_ci.csv", [r for r in conditional if r["analysis"] == "repeat"])
    write_csv(dest / "paired_conditional_metrics_ci.csv", paired)
    write_csv(dest / "repeated_po_mean_ci.csv", means)
    write_csv(dest / "repeated_po_paired_ci.csv", repeat_pairs)
    table_exports(dest, conditional, paired, means, repeat_pairs, original_alias, expected_repeats, len(finished) - int("original" in finished))
    summary = {"complete": not missing, "expected_analyses": list(expected_names), "completed_analyses": finished,
               "pending": missing, "track": MODE, "steps": list(STEPS), "variants": list(VARIANTS),
               "original_no_edge_weight_alias": original_alias, "all_metrics": list(METRICS),
               "maximum_new_XGBoost_fits": 4 * len(expected_names), "GNN_refits": 0, "GNN_inference_runs": 0,
               "new_simulations": 0, "tensor_conversion_runs": 0,
               "original_intervals": "conditional whole-trajectory percentile bootstrap",
               "repeat_intervals": "untruncated Student-t intervals of equally weighted matched repetition means",
               "pair_direction": "GNN minus PO XGBoost (positive favours GNN for these metrics)",
               "feature_contract": FEATURE_CONTRACT}
    atomic_json(dest / "summary.json", summary)
    return summary


def latex_escape(value: Any) -> str:
    mapping = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def latex_cell(row: Optional[Dict[str, Any]], across: bool = False, digits: int = 4) -> str:
    if row is None:
        return r"\textit{Not available}"
    value = row.get("mean" if across else "point")
    low, high = row.get("ci95_low" if across else "low"), row.get("ci95_high" if across else "high")
    if value is None:
        return r"\textit{Undefined}"
    point = f"{value:.{digits}f}"
    bounds = f"({low:.{digits}f}, {high:.{digits}f})" if low is not None and high is not None else "(CI unavailable)"
    return r"\shortstack{" + point + r"\\" + bounds + "}"


def latex_table(path: Path, rows: List[Dict[str, Any]], caption: str, label: str,
                across: bool, paired: bool = False) -> None:
    if not rows:
        path.write_text("% No completed comparisons available for this table.\n", encoding="utf-8")
        return
    field = "comparison" if paired else "model"
    lookup = {}
    for r in rows:
        lookup.setdefault((r["step"], r[field], r["split"]), {})[r["metric"]] = r
    headers = (r"Evaluation & Comparison & Split" if paired else r"Evaluation & Model & Split")
    headers += r" & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\"
    lines = [r"% Requires: booktabs,longtable,array,pdflscape", r"\begin{landscape}",
             r"\begingroup\footnotesize", r"\setlength{\tabcolsep}{1.5pt}",
             r"\renewcommand{\arraystretch}{1.20}", r"\setlength{\LTcapwidth}{\linewidth}",
             r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.16\linewidth}>{\raggedright\arraybackslash}p{0.075\linewidth}*{5}{>{\centering\arraybackslash}p{0.12\linewidth}}@{}}",
             r"\caption{" + caption + r"}\label{" + label + r"}\\",
             r"\toprule", headers, r"\midrule\endfirsthead",
             r"\multicolumn{8}{@{}l}{\textit{Table \thetable{} continued}}\\",
             r"\toprule", headers, r"\midrule\endhead", r"\bottomrule\endfoot"]
    ordered_models = ("GNN minus XGB_all", "GNN minus XGB_no_counts") if paired else MODELS
    for step in STEPS:
        for model in ordered_models:
            entries = [(split, lookup.get((step, model, split))) for split in SPLITS]
            entries = [(split, val) for split, val in entries if val]
            for i, (split, val) in enumerate(entries):
                model_text = ("GNN minus " + MODEL_NAMES[model.split(" minus ", 1)[1]]) if paired else MODEL_NAMES[model]
                cells = [latex_escape(STEP_NAMES[step]) if i == 0 else "",
                         latex_escape(model_text) if i == 0 else "", split.title()]
                cells += [latex_cell(val.get(m), across) for m in METRICS]
                lines.append(" & ".join(cells) + (r" \\*" if i < len(entries) - 1 else r" \\"))
            if entries:
                lines.append(r"\addlinespace[3pt]")
    lines += [r"\end{longtable}", r"\endgroup", r"\end{landscape}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_exports(dest: Path, conditional: List[Dict[str, Any]], paired: List[Dict[str, Any]],
                  means: List[Dict[str, Any]], repeat_pairs: List[Dict[str, Any]],
                  alias: bool, expected_repeats: int, completed_repeats: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    original = [r for r in conditional if r["analysis"] == "original" and r["split"] != "train"]
    original_paired = [r for r in paired if r["analysis"] == "original" and r["split"] != "train"]
    notes = (r"PO denotes partial observation. Both XGBoost variants use seven ordered daily PO summaries; "
             r"without counts removes admissions and discharges on every day. Incident-event channels remain "
             r"masked to zero. GNN denotes the existing GraphSAGE + Transformer model; its predictions are reused. "
             r"Steps 4 and 8 fit canonical and shifted XGBoost models respectively; Step 9 reuses the "
             r"corresponding canonical checkpoint without retraining. The 0.5 threshold is unchanged.")
    latex_table(dest / "original_po_comparison.tex", original,
                r"Original-data PO comparison on exactly matched validation and test windows. Entries are "
                r"point estimates with 95\% conditional whole-trajectory bootstrap intervals for all five metrics. "
                r"Overlapping windows are resampled together, within recorded simulator family. " + notes,
                "tab:xgboost-po-original", False)
    latex_table(dest / "original_po_paired.tex", original_paired,
                r"Original-data paired PO GNN-minus-XGBoost differences. Positive differences favour the GNN. "
                r"Intervals use identical whole-trajectory resamples for the compared models and are "
                r"conditional on the fixed fitted models, not independent training repetitions.",
                "tab:xgboost-po-original-paired", False, True)
    repeat_status = f"Completed matched repetitions: {completed_repeats}/{expected_repeats}. "
    if completed_repeats < expected_repeats:
        repeat_status = r"\textbf{Provisional partial results.} " + repeat_status
    latex_table(dest / "five_repeat_po_comparison.tex", means,
                repeat_status + r"PO predictive model comparison using each repetition's own saved training "
                r"and test datasets. Entries are equally weighted repeat means with 95\% Student-$t$ intervals "
                r"across independent joint repetitions; limits are not clipped to metric bounds. "
                r"These are not intervals around original-model estimates. Train is the fitted subset; "
                r"validation is held-out. " + notes, "tab:xgboost-po-repeats", True)
    latex_table(dest / "five_repeat_po_paired.tex", repeat_pairs,
                repeat_status + r"Paired PO GNN-minus-XGBoost performance differences across the same "
                r"independent repetitions. Positive differences favour the GNN. Entries are mean paired "
                r"differences with untruncated 95\% Student-$t$ confidence intervals. Repetitions, not "
                r"overlapping windows, are the independent units.", "tab:xgboost-po-repeats-paired", True, True)
    lookup = {(r["step"], r["model"], r["split"], r["metric"]): r for r in original}
    lines = [r"% Row fragments for the PO panel of the original Table 4; 12 columns.",
             r"% Insert each step's XGBoost rows after its existing PO GNN row.",
             r"% Original fitted-model estimates, NOT five-repeat means. Requires \shortstack."]
    for step in (STEP4, EDGE_ALIAS, STEP8, STEP9):
        if step == EDGE_ALIAS and not alias:
            continue
        source_step = STEP4 if step == EDGE_ALIAS else step
        models = ("XGB_all",) if step == EDGE_ALIAS else ("XGB_all", "XGB_no_counts")
        lines.append("% " + STEP_NAMES[step] + (": canonical XGBoost reused; edge weights do not enter PO summaries." if step == EDGE_ALIAS else ""))
        for model in models:
            if not any(key[0:2] == (source_step, model) for key in lookup):
                continue
            cells = ["", latex_escape("XGB: all PO summaries" if model == "XGB_all" else "XGB: without counts")]
            cells += [latex_cell(lookup.get((source_step, model, split, m)), False, 3)
                      for split in ("validation", "test") for m in METRICS]
            lines.append(" & ".join(cells) + r" \\")
    (dest / "table4_po_xgboost_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / "methods_po_comparison.tex").write_text(
        r"XGBoost comparators were fitted to the saved partial-observation datasets using seven ordered days "
        r"of observed-positive patient counts, admission and discharge counts, and the incident-event "
        r"channels masked to zero in the graph representation. The count ablation removed admissions and "
        r"discharges from every history day and data partition. Latent AMR states, unmasked incident events, "
        r"contact topology and future outcomes were not used as predictors. Training/validation trajectories, "
        r"history windows, stored targets and external test cohorts were matched to the corresponding "
        r"partial-observation GNN evaluation. XGBoost hyperparameters and seeds were taken from the corresponding "
        r"ground-truth XGBoost experiment. Canonical and shifted models were fitted separately; zero-shot "
        r"evaluation reused the canonical model without retraining or test-set threshold adjustment. "
        r"Original-data results use conditional whole-trajectory bootstrap intervals, whereas the separate "
        r"repeated analysis reports equally weighted means and Student-$t$ intervals across independent "
        r"repetitions. Paired differences use matched models within the same datasets or repetitions." + "\n",
        encoding="utf-8")


def make_bundle(root: Path) -> Path:
    target = root / "xgboost_po_review_bundle.zip"
    temp = root / "xgboost_po_review_bundle.tmp.zip"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == ".zip" or path.name.startswith(".") or ".tmp" in path.name:
                continue
            rel = path.relative_to(root)
            if rel.parts[0] in ("feature_cache", "interval_cache") or path.name == "model.json":
                continue
            archive.write(path, rel.as_posix())
    os.replace(temp, target)
    return target


def statistic_contract(helper: Helpers) -> None:
    import numpy as np
    rows = [{"sim_id": f"family_{i // 3}__sim_{i:03d}", "day": 7 + j, "start_day": 1 + j,
             "last_file": f"f{i}_{j}.pt", "y": int((i + j) % 3 == 0)} for i in range(6) for j in range(5)]
    p = np.asarray([.15 + .7 * ((i * 7) % 11) / 10 for i in range(len(rows))])
    y = np.asarray([r["y"] for r in rows])
    reference = helper.stats.trajectory_bootstrap(rows, y, p, 37, 719)
    combined = joint_bootstrap(helper.stats, rows, {"GNN": p, "XGB_all": p}, 37, 719)
    for metric in METRICS:
        for key in ("low", "high", "valid_bootstraps", "requested_bootstraps", "n_trajectories", "n_families"):
            if combined["marginal"]["GNN"][metric][key] != reference[metric][key]:
                raise EvidenceError(f"Reference bootstrap contract failed: {metric}/{key}")
        if combined["paired"]["XGB_all"][metric]["low"] != 0 or combined["paired"]["XGB_all"][metric]["high"] != 0:
            raise EvidenceError("Identity paired-comparison test failed.")
    mean = helper.stats.mean_interval([.6, .7, .8, .9, 1.0])
    from scipy.stats import t
    expected = np.mean([.6, .7, .8, .9, 1.0])
    half = t.ppf(.975, 4) * np.std([.6, .7, .8, .9, 1.0], ddof=1) / np.sqrt(5)
    if not math.isclose(mean["ci95_low"], expected - half) or not math.isclose(mean["ci95_high"], expected + half):
        raise EvidenceError("Repeat-interval contract failed.")


def self_test(helper: Helpers) -> None:
    """Small real-XGBoost and tensor-feature tests, without touching study inputs."""
    import numpy as np
    import tempfile
    import torch
    from types import SimpleNamespace
    statistic_contract(helper)
    graph = SimpleNamespace(x=torch.tensor([[0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.],
                                            [1., 1., 0., 0., 0., 0.]]), node_names=["p1", "p2", "staff"])
    count, ids = validate_po_graph(graph, "self-test")
    if count != 1 or ids != frozenset(["p1", "p2"]):
        raise EvidenceError("Observed-positive patient count test failed.")
    graph.x[0, 4] = 1
    try:
        validate_po_graph(graph, "deliberate unmasked incident-event test")
    except EvidenceError:
        pass
    else:
        raise EvidenceError("Unmasked incident-event input was not rejected.")
    graph.x[0, 4] = 0
    graph.x[0, 1] = 2
    try:
        validate_po_graph(graph, "deliberate latent-state test")
    except EvidenceError:
        pass
    else:
        raise EvidenceError("Latent-state input was not rejected.")
    with tempfile.TemporaryDirectory(prefix="xgboost_po_selftest_") as tmp:
        base = Path(tmp)
        folder = base / "TRACK_partial_observation" / "data"
        folder.mkdir(parents=True)
        for i in range(4):
            sid = f"family_{i // 2}__sim_{i:03d}"
            for day in range(1, 11):
                graph = SimpleNamespace(
                    x=torch.tensor([[0., float((i + day) % 2), 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]]),
                    node_names=["fixed", f"patient_{day // 3}"], sim_id=sid, day=day,
                    y_h7_endog_majority=torch.tensor([int((i + day) % 2)]))
                torch.save(graph, folder / f"{sid}__amr_r0_t{day}.pt")
        args = argparse.Namespace(out_root=base / "out", preflight=False, verify_data_hashes=False)
        plan = {"name": "original", "analysis": "original", "repeat": 0, "datasets": {"canonical_train": folder}}
        data = extract_dataset(plan, "canonical_train", args, helper, Evidence())
        second = extract_dataset(plan, "canonical_train", args, helper, Evidence())
        if not np.array_equal(data["X"], second["X"], equal_nan=True) or data["rows"] != second["rows"]:
            raise EvidenceError("Feature-cache reuse test failed.")
        # Verify full seven-day ordering and the mask, including the unknown first-day flows.
        X = data["X"]
        if X.shape != (16, 28) or not np.isnan(X[0, 1:3]).all() or np.any(X[:, 3::4] != 0):
            raise EvidenceError("PO seven-day feature layout failed.")
        selected = selected_features(data, "without_counts")
        if selected.shape != (16, 14) or not np.array_equal(selected[:, 0::2], X[:, 0::4]):
            raise EvidenceError("Every-day count ablation failed.")
        params = helper.xgb.xgb_parameters(0, 1)
        model = helper.xgb.XGBoostBinaryModel(**params).fit(X, data["y"])
        path = base / "tiny_model.json"
        model.save_model(path)
        before = file_hash(path)
        loaded = helper.xgb.XGBoostBinaryModel(**params)
        loaded.load_model(path)
        if not np.array_equal(model.predict_proba(X), loaded.predict_proba(X)) or file_hash(path) != before:
            raise EvidenceError("Native XGBoost model save/load failed.")
        rows = [{**r, "probability": .5, "prediction": 0} for r in data["rows"]]
        aligned_predictions(rows, data["rows"], "threshold equality")
        rows[0]["start_day"] += 1
        try:
            aligned_predictions(rows, data["rows"], "deliberate history mismatch")
        except EvidenceError:
            pass
        else:
            raise EvidenceError("Changed history window was not rejected.")
    print("PASS: PO feature/masking tests, chronological windows, count ablation, cache reuse, "
          "native XGBoost fit/save/load, all five metrics, paired bootstrap and Student-t intervals.", flush=True)


def select_repeats(value: str, reps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ids = {int(r["repeat"]) for r in reps}
    requested = ids if value == "all" else {int(v.strip()) for v in value.split(",") if v.strip()}
    if not requested or not requested <= ids:
        raise EvidenceError(f"Unknown --repeat-ids. Available: {sorted(ids)}")
    return [r for r in reps if int(r["repeat"]) in requested]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="ORIGINAL experiments_results (PO tensors and GNN summaries)")
    parser.add_argument("--xgboost-root", type=Path, help="Original GT XGBoost outputs; exact hyperparameter/seed reference")
    parser.add_argument("--repeat-root", "--reference-root", dest="repeat_root", type=Path,
                        help="Completed predictive_repeat_results, with source_snapshot and saved PO tensors")
    parser.add_argument("--out-root", type=Path, help="New separate output folder")
    parser.add_argument("--scope", choices=("both", "original", "repeats"), default="both")
    parser.add_argument("--repeat-ids", default="all", help="all or comma-separated IDs; completed fits are reused")
    parser.add_argument("--ci-config", type=Path, help="Original bootstrap count/seed; defaults to repo/predictive_repeats.json")
    parser.add_argument("--threads", type=int, help="Optional execution-thread override; default reuses saved GT settings")
    parser.add_argument("--preflight", action="store_true", help="Verify sources, data, splits and labels; no fitting or output writes")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild tables from completed add-on outputs; no fitting")
    parser.add_argument("--verify-data-hashes", action="store_true", help="Rehash all PO tensors even when using verified feature caches")
    parser.add_argument("--self-test", action="store_true", help="Small tests with synthetic fixtures; only --repo is needed")
    args = parser.parse_args(argv)
    args.repo = args.repo.expanduser().resolve()
    if args.self_test:
        self_test(load_helpers(args.repo, "selftest"))
        return 0
    if args.out_root is None:
        parser.error("--out-root is required")
    if args.preflight and args.summarize_only:
        parser.error("--preflight and --summarize-only are separate modes")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be positive")
    for name in ("data_root", "repeat_root", "xgboost_root", "out_root"):
        value = getattr(args, name)
        if value is not None:
            expanded = value.expanduser()
            if name == "out_root" and any(p.is_symlink() for p in (expanded, *expanded.parents)):
                raise EvidenceError("The output path must not pass through symbolic links.")
            setattr(args, name, expanded.resolve())
    if args.scope in ("both", "original") and (args.data_root is None or args.xgboost_root is None):
        parser.error("Original analysis needs --data-root and --xgboost-root")
    if args.scope in ("both", "repeats") and args.repeat_root is None:
        parser.error("Repeated analysis needs --repeat-root")
    args.ci_config = (args.ci_config or args.repo / "predictive_repeats.json").expanduser().resolve()
    roots = validate_paths(args)
    protect_inputs(args.repo, roots, None if args.preflight else args.out_root)
    missing_modules = [name for name in ("numpy", "scipy", "sklearn", "torch", "xgboost")
                       if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise MissingEvidence(f"Use the existing idp_ns environment. Missing modules: {missing_modules}")
    evidence = Evidence()
    helpers, manifest = {}, None
    if args.scope in ("both", "original"):
        helpers["original"] = load_helpers(args.repo, "original")
    if args.scope in ("both", "repeats"):
        manifest = repeat_manifest(args.repeat_root, evidence)
        helpers["repeat"] = load_helpers(args.repeat_root / "source_snapshot", "repeat", manifest["identity"]["source_sha256"])
    for h in helpers.values():
        statistic_contract(h)
        for name in h.hashes:
            evidence.record(h.source / name)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "xgboost")}
    expected_names = (["original"] if "original" in helpers else []) + (
        [f"repeat_{int(r['repeat']):03d}" for r in manifest["identity"]["repeats"]] if manifest else [])
    identity = {"version": VERSION, "script_sha256": file_hash(Path(__file__)), "scope": args.scope,
                "data_root": str(args.data_root), "xgboost_root": str(args.xgboost_root),
                "repeat_root": str(args.repeat_root), "repo": str(args.repo), "feature_contract": FEATURE_CONTRACT,
                "source_sha256": {k: h.hashes for k, h in helpers.items()}, "packages": versions,
                "ci_config": str(args.ci_config), "ci_config_sha256": evidence.record(args.ci_config) if "original" in helpers else None,
                "reference_repeat_identity": manifest["identity_sha256"] if manifest else None,
                "threads_override": args.threads, "expected_analyses": expected_names}
    owner = args.out_root / OWNER
    if owner.is_file() and read_json(owner).get("identity_sha256") != digest(identity):
        raise EvidenceError("Inputs, sources, settings or software differ from this output's manifest. Use a new --out-root.")
    if args.summarize_only:
        if not owner.is_file():
            raise MissingEvidence(f"No completed/started PO run found: {owner}")
        with output_lock(args.out_root):
            result = summarize(args.out_root, helpers, expected_names)
            bundle = make_bundle(args.out_root)
        return 0 if result["complete"] else 2
    selected = select_repeats(args.repeat_ids, manifest["identity"]["repeats"]) if manifest else []
    def jobs():
        if "original" in helpers:
            yield original_plan(args, helpers["original"], evidence), helpers["original"]
        for rep in selected:
            yield repeat_plan(args, helpers["repeat"], evidence, manifest, rep), helpers["repeat"]
    if args.preflight:
        public = []
        for plan, helper in jobs():
            context = prepare_context(plan, args, helper, evidence)
            public.append({"analysis": plan["name"], "state": "ready", "datasets": context["manifest"]["datasets"],
                           "matched_reference": {s: str(p) for s, p in plan["gnn_paths"].items()},
                           "maximum_XGBoost_fits": 4, "bootstrap_count": plan["bootstrap"],
                           "bootstrap_seed": plan["bootstrap_seed"], "alignment": context["manifest"]["reference_prediction_alignment"]})
            del context
        evidence.check_unchanged()
        print(json.dumps({"tasks": public, "new_simulations": 0, "GNN_fits": 0, "GNN_inference": 0,
                          "preflight_output_writes": 0, "maximum_XGBoost_fits": 4 * len(public)}, indent=2), flush=True)
        return 0
    started, fitted = time.monotonic(), 0
    with output_lock(args.out_root):
        atomic_json(owner, {"identity": identity, "identity_sha256": digest(identity)})
        for label, helper in helpers.items():
            copy_sources(helper, args.out_root / "sources" / label)
        shutil.copy2(Path(__file__), args.out_root / "sources" / "run_xgboost_po.py")
        atomic_json(args.out_root / "feature_contract.json", FEATURE_CONTRACT)
        for plan, helper in jobs():
            print(f"\n[analysis] {plan['name']}: existing PO data, Steps 4/8/9", flush=True)
            context = prepare_context(plan, args, helper, evidence)
            fitted += fit_context(context, args, helper)
            del context
        evidence.check_unchanged()
        atomic_json(args.out_root / "source_evidence.json", list(evidence.small.values()))
        result = summarize(args.out_root, helpers, expected_names)
        atomic_json(args.out_root / "last_execution.json", {"new_fits_this_execution": fitted,
                    "seconds": time.monotonic() - started, "requested_repeat_ids": args.repeat_ids,
                    "complete": result["complete"], "pending": result["pending"]})
        bundle = make_bundle(args.out_root)
    print(f"\n{'COMPLETE' if result['complete'] else 'PARTIAL'}: {args.out_root}", flush=True)
    print(f"New XGBoost fits this execution: {fitted}; no GNN refits, GNN inference or simulations.", flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, FileNotFoundError, ImportError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
