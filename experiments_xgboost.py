#!/usr/bin/env python3
"""Ground-truth XGBoost evaluation with canonical and shift count ablations.

Reads existing experiments_pbc7.py PT datasets. Never generates simulations,
modifies input data, imports a GNN trainer, or trains another learner.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

VERSION = "1.4.0"
MODES = ("ground_truth",)
DAILY_FEATURES = ("patient_burden", "patient_admissions",
                  "patient_discharges", "recent_events")
HISTORY = HORIZON = 7
METRICS = ("auroc", "accuracy", "macro_f1", "sensitivity", "specificity")
SPLITS = ("train", "validation", "test")
ROWS = (
    ("step4_baseline", "Step4 Baseline"),
    ("step5_no_admission_turnover", "Step5 No Admissions or Turnover"),
    ("step5_no_edge_weights", "Step5 No Edge Weights"),
    ("step8_distribution_shift", "Step8 Distribution Shift"),
    ("step8_no_admission_turnover", "Step8 Shift: No Admissions or Turnover"),
    ("step9_baseline_to_shift_test", "Step9 Baseline to Shift Test"),
    ("step9_no_admission_turnover", "Step9 Transfer: No Admissions or Turnover"),
)
ABLATION_REFERENCES = {
    "step5_no_admission_turnover": "step4_baseline",
    "step8_no_admission_turnover": "step8_distribution_shift",
    "step9_no_admission_turnover": "step9_baseline_to_shift_test",
}
TRANSFER_SOURCES = {
    "step9_baseline_to_shift_test": "step4_baseline",
    "step9_no_admission_turnover": "step5_no_admission_turnover",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def output_lock(output):
    # POSIX lock releases automatically on normal exit, interruption or process death.
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".xgboost.lock").open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this output directory.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Tasks:
    def __init__(self, output, identity):
        self.output = output
        self.fingerprint = digest(identity)
        path = output / "00_provenance/run_identity.json"
        if path.exists() and read_json(path).get("fingerprint") != self.fingerprint:
            raise RuntimeError("Inputs, settings, code or software changed. Use a new --output directory.")
        atomic_json(path, {"fingerprint": self.fingerprint, **identity})

    def run(self, key, parameters, callback):
        directory = self.output / key
        directory.mkdir(parents=True, exist_ok=True)
        record_path = directory / "_task.json"
        fingerprint = digest({"run": self.fingerprint, "key": key, "parameters": parameters})
        if record_path.exists():
            old = read_json(record_path)
            if old["fingerprint"] != fingerprint:
                raise RuntimeError(f"Changed task inputs: {key}. Use a new output directory.")
            if old["status"] == "complete":
                for name, expected in old["artifacts"].items():
                    path = directory / name
                    if not path.is_file() or file_hash(path) != expected:
                        raise RuntimeError(f"Completed artifact changed or disappeared: {path}")
                print(f"[resume] {key}", flush=True)
                return directory
        print(f"[run] {key}", flush=True)
        started = time.monotonic()
        record = {"key": key, "fingerprint": fingerprint, "status": "running"}
        atomic_json(record_path, record)
        try:
            callback(directory)
            artifacts = {p.relative_to(directory).as_posix(): file_hash(p)
                         for p in sorted(directory.rglob("*"))
                         if p.is_file() and p.name != "_task.json" and not p.name.endswith(".tmp")}
            if not artifacts:
                raise RuntimeError(f"Task produced no artifacts: {key}")
            record.update(status="complete", seconds=time.monotonic() - started, artifacts=artifacts)
            atomic_json(record_path, record)
        except BaseException as exc:
            record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            atomic_json(record_path, record)
            raise
        return directory


def has_pt(path):
    return path.is_dir() and next(path.glob("*.pt"), None) is not None


def valid_archive_path(path):
    for part in path.parts:
        if re.fullmatch(r"T\d+", part) and part != "T7":
            return False
        if re.fullmatch(r"h\d+", part) and part != "h7":
            return False
    return True


def choose_folder(candidates, description):
    seen = set()
    for path in candidates:
        path = Path(path).resolve()
        if path not in seen and has_pt(path):
            return path
        seen.add(path)
    searched = "\n".join(f"  {p}" for p in sorted(seen)) or "  No matching live or archived directories."
    raise FileNotFoundError(f"Missing saved PT dataset: {description}. No data will be regenerated.\nSearched:\n{searched}")


def archive_candidates(archive, tag, split):
    root = archive / tag
    if not root.exists():
        return []
    return sorted((p for p in root.rglob(split) if valid_archive_path(p.relative_to(root))),
                  key=lambda p: (len(p.parts), str(p)))


def original_metrics_path(archive, tag):
    candidates = []
    for p in (archive / tag).rglob("metrics_summary.json"):
        if not valid_archive_path(p.relative_to(archive / tag)):
            continue
        value = read_json(p)
        config = value.get("config", {})
        if int(config.get("T", HISTORY)) == HISTORY and value.get("task", "endogenous_importation_majority_h7") == "endogenous_importation_majority_h7":
            candidates.append(p)
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple original runs for {tag}: {candidates}. Select one results root.")
    return candidates[0] if candidates else None


def discover(data_root, driver):
    plans = []
    for mode in MODES:
        work = data_root / f"TRACK_{mode}" / "work"
        archive = work / "repro_artifacts_steps_1_9"
        if not work.is_dir():
            raise FileNotFoundError(f"Expected original track directory: {work}")
        baseline_train = choose_folder([work / driver.BASELINE_TRAIN_REL] + archive_candidates(
            archive, "step4_baseline", "train_folder"), f"{mode}: baseline train")
        baseline_test = choose_folder([work / driver.BASELINE_TEST_REL] + archive_candidates(
            archive, "step4_baseline", "test_folder"), f"{mode}: frozen canonical test")
        shifted_train = choose_folder([work / driver.STEP8_TRAIN_REL] + archive_candidates(
            archive, "step8_distribution_shift", "train_folder"), f"{mode}: shifted train")
        shifted_test = choose_folder([work / driver.STEP8_TEST_REL] + archive_candidates(
            archive, "step8_distribution_shift", "test_folder") + archive_candidates(
            archive, "step9_baseline_to_shift_test", "test_folder"), f"{mode}: shifted test")
        for tag, label in ROWS:
            count_ablation = tag in ABLATION_REFERENCES
            reference_tag = ABLATION_REFERENCES.get(tag, tag)
            # Each ablation uses the corresponding experiment's data/split.
            original = original_metrics_path(archive, reference_tag)
            train_transform = None
            if reference_tag in {"step4_baseline", "step9_baseline_to_shift_test"}:
                train = baseline_train
            elif reference_tag == "step8_distribution_shift":
                train = shifted_train
            elif tag == "step5_no_edge_weights":
                live = [work / "step5_ablation" / "no_edge_weights"]
                try:
                    train = choose_folder(live + archive_candidates(archive, tag, "train_folder"), f"{mode}: {tag}")
                except FileNotFoundError:
                    # Resetting edge weights cannot change XGBoost's inputs.
                    train = baseline_train
                    train_transform = "original_edge_weight_reset"
            else:
                raise ValueError(f"Unsupported experiment: {tag}")
            test = shifted_test if tag.startswith(("step8_", "step9_")) else baseline_test
            original_record = read_json(original) if original else {}
            original_cfg = original_record.get("config", {})
            step = int(original_cfg.get("sliding_step", driver.CONFIG["MODEL"]["sliding_step"]))
            if step != 1:
                raise RuntimeError(f"{mode}/{tag}: original sliding step is {step}, expected the submitted value 1.")
            plans.append({"mode": mode, "tag": tag, "label": label,
                          "train_folder": str(train), "test_folder": str(test),
                          "train_transform": train_transform,
                          "exclude_admission_turnover": count_ablation,
                          "reference_step": reference_tag,
                          "transfer_source_step": TRANSFER_SOURCES.get(tag),
                          "original_gnn_comparison_available": not count_ablation,
                          "split_seed": int(original_cfg.get("split_seed", 0)),
                          "original_metrics": str(original) if original else None})
    return plans


def input_manifest(folder):
    return [{"name": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(folder.glob("*.pt"), key=lambda p: natural_key(p.name))]


def scalar(data, field):
    value = getattr(data, field, None)
    if value is None:
        raise ValueError(f"Missing required original field: {field}")
    return float(value.item() if hasattr(value, "item") else value)


def daily_values(x, names, previous, mode):
    """Only node status and patient membership enter these four daily values."""
    import numpy as np
    if mode != "ground_truth":
        raise ValueError("This package runs the ground-truth track only.")
    if x.ndim != 2 or x.shape[1] != 6 or len(names) != len(x):
        raise ValueError("Expected the original six node columns and aligned node_names.")
    if len(set(names)) != len(names) or not np.isfinite(x).all():
        raise ValueError("Duplicate patient identifiers or nonfinite original node values.")
    patient = x[:, 0] == 0
    current = {str(names[i]): float(x[i, 1]) for i in np.flatnonzero(patient)}
    states = x[patient, 1]
    if not np.isin(x[:, 1], [0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected ground-truth state encoding.")
    burden = int(np.isin(states, [2, 4]).sum())
    events = float(x[patient, 4:6].sum())
    admissions = len(current.keys() - previous.keys()) if previous is not None else float("nan")
    discharges = len(previous.keys() - current.keys()) if previous is not None else float("nan")
    return np.asarray([burden, admissions, discharges, events], dtype=np.float32), current


def extract_dataset(folder, mode, manifest, destination):
    import numpy as np
    import torch
    # Group insertion order and day order match TemporalGraphDataset._scan_folder.
    groups = {}
    hashes = []
    for index, info in enumerate(manifest):
        path = folder / info["name"]
        if path.stat().st_size != info["bytes"] or path.stat().st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"Input changed during extraction: {path}")
        payload = path.read_bytes()
        data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        sid = str(getattr(data, "sim_id", ""))
        if not sid:
            raise ValueError(f"Missing sim_id: {path}")
        day = int(scalar(data, "day"))
        names = getattr(data, "node_names", None)
        if names is None:
            raise ValueError(f"node_names are needed to count admissions and discharges: {path}")
        label = scalar(data, "y_h7_endog_majority")
        if label not in (0, 1):
            raise ValueError(f"Invalid original label: {path}")
        endogenous = scalar(data, "y_h7_trans_res") + scalar(data, "y_h7_select_res")
        imported = scalar(data, "y_h7_import_res")
        groups.setdefault(sid, []).append({"day": day, "file": path.name,
            "x": data.x.detach().cpu().numpy().copy(), "names": list(map(str, names)),
            "y": int(label), "endogenous": endogenous, "imported": imported})
        hashes.append({**info, "sha256": hashlib.sha256(payload).hexdigest()})
        if (index + 1) % 500 == 0:
            print(f"  extracted {index + 1}/{len(manifest)} daily records: {folder.name}", flush=True)
    features, samples, daily_rows = [], [], []
    for sid, days in groups.items():
        days.sort(key=lambda row: row["day"])
        ds = [row["day"] for row in days]
        if len(set(ds)) != len(ds) or any(b != a + 1 for a, b in zip(ds, ds[1:])):
            raise RuntimeError(f"Noncontiguous or duplicate days in {folder}, trajectory {sid}")
        previous = None
        values = []
        for row in days:
            value, previous = daily_values(row.pop("x"), row.pop("names"), previous, mode)
            values.append(value)
            daily_rows.append({"sim_id": sid, "day": row["day"], "file": row["file"],
                               **dict(zip(DAILY_FEATURES, [None if np.isnan(v) else float(v) for v in value]))})
        for start in range(len(days) - HISTORY + 1):
            last = days[start + HISTORY - 1]
            total = last["endogenous"] + last["imported"]
            positive_tie = total > 0 and last["endogenous"] == last["imported"]
            features.append(np.stack(values[start:start + HISTORY]).reshape(-1))
            samples.append({"sim_id": sid, "day": last["day"], "start_day": days[start]["day"],
                "last_file": last["file"], "y": last["y"], "positive_event_tie": positive_tie,
                "no_future_events": total == 0, "full_future_in_folder": last["day"] + HORIZON <= ds[-1],
                "future_endogenous_audit_only": last["endogenous"],
                "future_imported_audit_only": last["imported"]})
    if not samples:
        raise ValueError(f"No seven-day windows: {folder}")
    matrix = np.asarray(features, dtype=np.float32)
    with (destination / "features.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, X=matrix, y=np.asarray([row["y"] for row in samples], dtype=np.int64))
    os.replace(destination / "features.npz.tmp", destination / "features.npz")
    atomic_json(destination / "samples.json", samples)
    atomic_json(destination / "source_files.json", hashes)
    write_csv(destination / "daily_features.csv", daily_rows)
    atomic_json(destination / "feature_audit.json", {
        "mode": mode, "source": str(folder), "daily_feature_names": DAILY_FEATURES,
        "column_order": [f"lag_{lag}_{name}" for lag in range(HISTORY - 1, -1, -1) for name in DAILY_FEATURES],
        "shape": list(matrix.shape), "topology_used": False, "node_ids_as_features": False,
        "future_labels_as_features": False, "source_manifest_sha256": digest(hashes),
        "observation_rule": "Original GT resistant status and daily CR/IR event flags",
        "first_snapshot_flows": "Unavailable admissions/discharges are NaN, not zero or initial census",
        "window_rule": "Original complete history windows; no change to saved future labels or eligibility",
        "windows": len(samples), "positive_event_ties": sum(r["positive_event_tie"] for r in samples),
        "event_free_future_windows": sum(r["no_future_events"] for r in samples),
        "incomplete_future_windows": sum(not r["full_future_in_folder"] for r in samples),
    })


def load_dataset(directory):
    import numpy as np
    with np.load(directory / "features.npz", allow_pickle=False) as values:
        return values["X"], values["y"], read_json(directory / "samples.json")


def apply_feature_selection(data, plan):
    """Remove both count features from every history day in any partition."""
    if not plan.get("exclude_admission_turnover"):
        return data
    if plan["tag"] not in ABLATION_REFERENCES:
        raise ValueError("Unexpected row requesting the admissions/turnover ablation.")
    X, y, samples = data
    keep = [day * len(DAILY_FEATURES) + DAILY_FEATURES.index(name)
            for day in range(HISTORY) for name in ("patient_burden", "recent_events")]
    return X[:, keep].copy(), y, samples


def original_split(samples, seed):
    import numpy as np
    # Verbatim rule from train_amr_dygformer.py: 20% of sorted/shuffled sim IDs.
    ids = sorted({row["sim_id"] for row in samples})
    if len(ids) < 2:
        raise ValueError("At least two trajectories are required for the original validation split.")
    np.random.RandomState(seed).shuffle(ids)
    n_validation = max(1, min(len(ids) - 1, int(round(len(ids) * 0.2))))
    validation_ids = set(ids[:n_validation])
    train = np.asarray([i for i, row in enumerate(samples) if row["sim_id"] not in validation_ids])
    validation = np.asarray([i for i, row in enumerate(samples) if row["sim_id"] in validation_ids])
    return train, validation


def verify_original_labels(plan, y_validation, y_test):
    import numpy as np
    path = plan["original_metrics"]
    if path is None:
        return {"status": "original_metrics_not_available", "split_rule": "original source defaults"}
    original = read_json(path)
    checks = {}
    for split, values in (("validation", y_validation), ("test", y_test)):
        reference = original.get(split, {}).get("y_true")
        if reference is None:
            checks[split] = "original_y_true_not_available"
        elif not np.array_equal(np.asarray(reference, dtype=int), values):
            raise RuntimeError(f"{plan['mode']}/{plan['tag']}: {split} labels/order differ from original saved GNN results. Refusing a mismatched comparison.")
        else:
            checks[split] = "exact_label_and_order_match"
    return {"status": "checked", **checks, "original_metrics_sha256": file_hash(path)}


def xgb_parameters(seed, threads):
    # Fixed in advance; no test-set tuning or class rebalancing.
    return {"objective": "binary:logistic", "eval_metric": "logloss", "booster": "gbtree",
            "tree_method": "hist", "grow_policy": "lossguide", "max_depth": 0,
            "max_leaves": 15, "n_estimators": 150, "learning_rate": 0.1,
            "min_child_weight": 1, "gamma": 0.0, "max_bin": 256,
            "subsample": 1.0, "colsample_bytree": 1.0, "reg_alpha": 0.0,
            "reg_lambda": 1.0, "scale_pos_weight": 1.0, "base_score": 0.5,
            "random_state": seed, "n_jobs": threads, "verbosity": 0}


def array_hash(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        h.update(str((value.shape, str(value.dtype))).encode())
        h.update(value.tobytes())
    return h.hexdigest()


class XGBoostBinaryModel:
    """Native XGBoost interface, independent of sklearn estimator-tag versions."""

    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.booster = None

    def fit(self, X, y):
        import xgboost as xgb
        params = dict(self.parameters)
        rounds = params.pop("n_estimators")
        params["seed"] = params.pop("random_state")
        params["nthread"] = params.pop("n_jobs")
        train = xgb.DMatrix(X, label=y, nthread=params["nthread"])
        self.booster = xgb.train(params, train, num_boost_round=rounds)
        return self

    def predict_proba(self, X):
        import numpy as np
        import xgboost as xgb
        data = xgb.DMatrix(X, nthread=self.parameters["n_jobs"])
        p = self.booster.predict(data)
        return np.column_stack([1.0 - p, p])

    def save_model(self, path):
        self.booster.save_model(path)

    def load_model(self, path):
        import xgboost as xgb
        self.booster = xgb.Booster(params={"nthread": self.parameters["n_jobs"]})
        self.booster.load_model(path)


def metrics(y, probability):
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
    y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if not len(y):
        return {"n": 0, "positive": 0, "status": "empty", **{key: None for key in METRICS}}
    prediction = (probability > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"n": int(len(y)), "positive": int(y.sum()),
            "status": "ok" if both else "single_class_auc_undefined",
            "auroc": float(roc_auc_score(y, probability)) if both else None,
            "accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def cluster_interval(samples, y, probability, repetitions, seed):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    ids = sorted({row["sim_id"] for row in samples})
    if len(ids) < 2:
        return {"status": "fewer_than_two_trajectories", "clusters": len(ids)}
    groups = {sid: np.asarray([i for i, row in enumerate(samples) if row["sim_id"] == sid]) for sid in ids}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(repetitions):
        selected = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([groups[sid] for sid in selected])
        if len(np.unique(y[idx])) == 2:
            aucs.append(float(roc_auc_score(y[idx], probability[idx])))
    return {"status": "ok" if aucs else "auc_not_estimable", "clusters": len(ids),
            "unit": "whole_simulation_trajectory", "replicates": repetitions,
            "valid_auc_replicates": len(aucs), "method": "95% percentile trajectory bootstrap; conditional on fitted model",
            "auroc_low": float(np.quantile(aucs, 0.025)) if aucs else None,
            "auroc_high": float(np.quantile(aucs, 0.975)) if aucs else None}


def family_name(sample):
    text = sample["last_file"] + "__" + sample["sim_id"]
    for pattern in (r"shift_(?:train|test)_(?:endog|import)_[a-zA-Z0-9]+",
                    r"(?:endog_high|import_high)_(?:train|test)"):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return "unresolved_family"


def evaluate(destination, split, X, y, samples, model, repetitions, seed):
    import numpy as np
    probability = model.predict_proba(X)[:, 1]
    result = metrics(y, probability)
    keep = np.asarray([not row["positive_event_tie"] for row in samples], dtype=bool)
    result["excluding_positive_event_ties"] = metrics(y[keep], probability[keep])
    result["positive_event_ties"] = int((~keep).sum())
    if split == "test":
        result["uncertainty"] = cluster_interval(samples, y, probability, repetitions, seed)
    write_csv(destination / f"{split}_predictions.csv", [
        {**row, "probability": float(p), "prediction": int(p > 0.5)}
        for row, p in zip(samples, probability)])
    strata = []
    if split == "test":
        for field, values in (("family", [family_name(row) for row in samples]),
                              ("trajectory", [row["sim_id"] for row in samples])):
            for value in sorted(set(values)):
                selected = np.asarray([v == value for v in values])
                strata.append({"grouping": field, "group": value,
                               "positive_event_ties": sum(row["positive_event_tie"] for row, flag in zip(samples, selected) if flag),
                               "event_free_windows": sum(row["no_future_events"] for row, flag in zip(samples, selected) if flag),
                               "incomplete_future_windows": sum(not row["full_future_in_folder"] for row, flag in zip(samples, selected) if flag),
                               **metrics(y[selected], probability[selected])})
        write_csv(destination / "test_within_regime.csv", strata)
    return result


def run_experiment(destination, plan, train_data, test_data, params, repetitions,
                   fit_fingerprint, reuse_model, baseline_model, parity):
    import numpy as np
    X, y, samples = train_data
    Xt, yt, test_samples = test_data
    train_idx, val_idx = original_split(samples, plan["split_seed"])
    train_ids = {samples[i]["sim_id"] for i in train_idx}
    val_ids = {samples[i]["sim_id"] for i in val_idx}
    test_ids = {row["sim_id"] for row in test_samples}
    if train_ids & val_ids or (train_ids | val_ids) & test_ids:
        raise RuntimeError("Trajectory overlap across train/validation/external test partitions.")
    model = XGBoostBinaryModel(**params)
    model_path = destination / "model.json"
    ready = destination / "model_ready.json"
    source = baseline_model if plan["tag"].startswith("step9_") else reuse_model
    fit_status = "trained"
    if ready.exists():
        saved = read_json(ready)
        if saved["fingerprint"] != fit_fingerprint or file_hash(model_path) != saved["model_sha256"]:
            raise RuntimeError("Interrupted model checkpoint does not match current inputs.")
        model.load_model(model_path)
        fit_status = saved["fit_status"]
    elif source is not None:
        source = Path(source)
        model.load_model(source)
        shutil.copy2(source, model_path)
        fit_status = "zero_shot_baseline_model" if plan["tag"].startswith("step9_") else "reused_identical_training_inputs"
    else:
        if plan["tag"].startswith("step9_"):
            raise RuntimeError("Step 9 requires the corresponding completed canonical model (with or without counts).")
        if len(np.unique(y[train_idx])) != 2:
            raise ValueError(f"{plan['tag']}: original training partition has only one class; no alternative learner or relabelling is substituted.")
        model.fit(X[train_idx], y[train_idx])
        model.save_model(destination / "model.tmp.json")
        os.replace(destination / "model.tmp.json", model_path)
    if not ready.exists():
        atomic_json(ready, {"fingerprint": fit_fingerprint, "model_sha256": file_hash(model_path),
                            "fit_status": fit_status, "source_model": str(source) if source else None})
    if model.booster.num_features() != X.shape[1] or Xt.shape[1] != X.shape[1]:
        raise RuntimeError("Model and train/test feature counts differ; refusing an incompatible transfer.")
    split_rows = []
    validation_samples = [samples[i] for i in val_idx]
    for name, indices in (("train", train_idx), ("validation", val_idx)):
        split_rows.extend({"split": name, "sim_id": samples[i]["sim_id"], "day": samples[i]["day"],
                           "y": int(y[i])} for i in indices)
    split_rows.extend({"split": "test", "sim_id": row["sim_id"], "day": row["day"], "y": int(row["y"])}
                      for row in test_samples)
    write_csv(destination / "split_manifest.csv", split_rows)
    result = {"mode": plan["mode"], "step": plan["tag"], "label": plan["label"],
              "training_source": plan["train_folder"], "training_transform": plan.get("train_transform"),
              "reference_step": plan.get("reference_step", plan["tag"]),
              "transfer_source_step": plan.get("transfer_source_step"),
              "excluded_features": ["patient_admissions", "patient_discharges"] if plan.get("exclude_admission_turnover") else [],
              "ablation_applied_to": ["train", "validation", "test"] if plan.get("exclude_admission_turnover") else [],
              "fit_status": fit_status, "n_models": 1, "n_training_windows": int(len(train_idx)),
              "n_training_trajectories": len(train_ids), "original_label_parity": parity,
              "feature_width": int(X.shape[1]), "feature_count_per_day": int(X.shape[1] // HISTORY),
              "training_data_hash": array_hash(X[train_idx], y[train_idx]),
              "validation_data_hash": array_hash(X[val_idx], y[val_idx]),
              "test_data_hash": array_hash(Xt, yt), "parameters": params,
              "threshold": 0.5, "threshold_equality": "negative class, matching original argmax"}
    result["validation"] = evaluate(destination, "validation", X[val_idx], y[val_idx],
                                    validation_samples, model, repetitions, params["random_state"])
    result["test"] = evaluate(destination, "test", Xt, yt, test_samples, model, repetitions, params["random_state"])
    result["train"] = evaluate(destination, "train", X[train_idx], y[train_idx],
                               [samples[i] for i in train_idx], model, repetitions, params["random_state"])
    atomic_json(destination / "metrics.json", result)


def format_value(value):
    return "--" if value is None else f"{float(value):.3f}"


def export_tables(destination, completed, plans):
    destination.mkdir(parents=True, exist_ok=True)
    if len(completed) != len(plans):
        raise ValueError("Every experiment plan must have exactly one result directory.")
    rows, strata, ties, comparisons = [], [], [], []
    documents = []
    for plan, directory in zip(plans, completed):
        result = read_json(directory / "metrics.json")
        row = {"track": plan["mode"], "step": plan["tag"], "experiment": plan["label"],
               "fit_status": result["fit_status"], "n_models": 1, "input_features": result["feature_width"]}
        for split in SPLITS:
            row.update({f"{split}_{metric}": result[split][metric] for metric in METRICS})
            row[f"{split}_windows"] = result[split]["n"]
            row[f"{split}_positive"] = result[split]["positive"]
            ties.append({"track": plan["mode"], "step": plan["tag"], "split": split,
                         "excluded_ties": result[split]["positive_event_ties"],
                         **result[split]["excluding_positive_event_ties"]})
        interval = result["test"].get("uncertainty", {})
        row.update(test_auroc_ci_low=interval.get("auroc_low"), test_auroc_ci_high=interval.get("auroc_high"))
        rows.append(row)
        with (directory / "test_within_regime.csv").open(newline="", encoding="utf-8") as stream:
            strata.extend({"track": plan["mode"], "step": plan["tag"], **r} for r in csv.DictReader(stream))
        original_path = plan.get("original_metrics")
        if original_path and plan.get("original_gnn_comparison_available", True):
            original = read_json(original_path)
            for split in SPLITS:
                gm = original.get(split, {}).get("metrics", {})
                comparisons.append({"track": plan["mode"], "step": plan["tag"], "split": split,
                    "xgboost_auroc": result[split]["auroc"], "original_gnn_auroc": gm.get("roc_auc"),
                    "xgboost_accuracy": result[split]["accuracy"], "original_gnn_accuracy": gm.get("accuracy"),
                    "xgboost_macro_f1": result[split]["macro_f1"], "original_gnn_macro_f1": gm.get("f1_macro"),
                    "xgboost_sensitivity": result[split]["sensitivity"], "original_gnn_sensitivity": gm.get("sensitivity"),
                    "xgboost_specificity": result[split]["specificity"], "original_gnn_specificity": gm.get("specificity"),
                    "original_gnn_metrics_available": bool(gm),
                    "label_parity": result["original_label_parity"].get(split, "unverified")})
    write_csv(destination / "all_table_rows.csv", rows)
    write_csv(destination / "within_regime_results.csv", strata)
    write_csv(destination / "positive_event_tie_sensitivity.csv", ties)
    write_csv(destination / "comparison_with_saved_gnn.csv", comparisons)
    for mode in MODES:
        subset = [r for r in rows if r["track"] == mode]
        write_csv(destination / f"xgboost_table_{mode}.csv", subset)
        title = "Ground truth"
        lines = [rf"\section*{{XGBoost: {title}}}", r"\begin{center}\small",
                 r"\noindent Train denotes the fitted training partition; Validation is the held-out partition labelled Train in the original GNN tables.\par\smallskip",
                 r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrrrrrrrrr}\toprule",
                 r"Experiment & \multicolumn{5}{c}{Train (fitted partition)} & \multicolumn{5}{c}{Validation (GNN table: Train)} & \multicolumn{5}{c}{Test} & Models \\",
                 r" & AUROC & Acc & F1 & Sens & Spec & AUROC & Acc & F1 & Sens & Spec & AUROC & Acc & F1 & Sens & Spec & \\ \midrule"]
        for row in subset:
            vals = [format_value(row[f"{split}_{metric}"]) for split in SPLITS for metric in METRICS]
            lines.append(row["experiment"] + " & " + " & ".join(vals) + r" & 1 \\")
        lines += [r"\bottomrule\end{tabular}}\end{center}",
                  r"\noindent F1 is the unweighted mean across both classes; Sens is sensitivity and Spec is specificity. Models denotes one fitted model evaluated per row, with reuse described below. Undefined metrics are shown as dashes. Test AUROC confidence intervals are retained in the CSV exports.",
                  r"\par\smallskip Step 9 uses the corresponding canonical model without retraining: Step 4 for full features, or the Step 5 count-ablated model for the count-ablated transfer. Edge-weight removal does not change XGBoost inputs; that row reuses the baseline model.",
                  r"\par\smallskip Count ablations remove admission and turnover counts from training, validation and testing, retaining 14 inputs: seven days of burden and recent events. Standard rows retain 28 inputs: seven days of all four summaries, without contact topology. The Step 8 ablation trains on the original shifted training set; the Step 9 ablation never trains on shifted data.",
                  r"\par\smallskip The same saved labels and original trajectory split are used; event-free and truncated future windows remain in the primary table for comparability. Steps 6 and 7 are omitted."]
        content = "\n".join(lines) + "\n"
        (destination / f"xgboost_table_{mode}.tex").write_text(content, encoding="utf-8")
        documents.append(content)
    (destination / "xgboost_tables_standalone.tex").write_text(
        "\\documentclass[10pt,a4paper,landscape]{article}\n\\usepackage[margin=1.5cm]{geometry}\n"
        "\\usepackage{booktabs,graphicx}\n\\begin{document}\n" + "\n\\clearpage\n".join(documents) +
        "\n\\end{document}\n", encoding="utf-8")
    atomic_json(destination / "run_summary.json", {
        "rows": len(rows), "tracks": list(MODES),
        "skipped_steps": [6, 7],
        "new_models_trained": sum(row["fit_status"] == "trained" for row in rows),
        "zero_shot_rows": sum(row["fit_status"] == "zero_shot_baseline_model" for row in rows),
        "identical_input_reuses": sum(row["fit_status"] == "reused_identical_training_inputs" for row in rows),
        "new_simulations": 0, "gnn_training_runs": 0, "learners": ["XGBoost"],
    })



def software_versions():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Original results directory containing TRACK_ground_truth/work")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parent), help="Folder containing the original experiments_pbc7.py")
    parser.add_argument("--output", default="xgboost_gt_full_metrics_results")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--plan", action="store_true", help="Locate every dataset and show the run plan without loading graphs or fitting")
    parser.add_argument("--fresh", action="store_true",
                        help="Require a new or empty output directory so all source models are fitted from scratch")
    args = parser.parse_args(argv)
    root, repo, output = [Path(p).expanduser().resolve() for p in (args.data_root, args.repo, args.output)]
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Original data and new outputs must be separate, non-nested directories.")
    if args.fresh and output.exists() and any(output.iterdir()):
        raise FileExistsError("--fresh requires a new or empty --output directory. Choose a new name; existing results are never deleted.")
    if args.threads < 1 or args.bootstrap < 1:
        raise ValueError("--threads and --bootstrap must be positive.")
    driver_path = repo / "experiments_pbc7.py"
    if not driver_path.is_file():
        raise FileNotFoundError("Place this add-on beside experiments_pbc7.py, or pass --repo ORIGINAL_REPO.")
    driver = load_module(driver_path, "_original_experiments_pbc7")
    if driver.CONFIG["MODEL"]["task"] != "endogenous_importation_majority_h7" or int(driver.CONFIG["MODEL"]["T"]) != 7:
        raise ValueError("This add-on is scoped to the submitted seven-day endogenous/importation task.")
    plans = discover(root, driver)
    params = xgb_parameters(args.seed, args.threads)
    if args.plan:
        print(json.dumps({"version": VERSION, "new_learner": "XGBoost", "new_simulations": 0,
                          "skipped_steps": [6, 7],
                          "gnn_training_runs": 0, "daily_features": DAILY_FEATURES,
                          "history_days": HISTORY, "input_width": 28, "parameters": params,
                          "count_ablation_input_width": 14,
                          "table_rows": plans}, indent=2))
        return 0
    try:
        import xgboost
        import torch
    except ImportError as exc:
        raise RuntimeError("Use the original conda environment with XGBoost installed. See README_XGBOOST.md.") from exc
    torch.set_num_threads(args.threads)
    unique = sorted({(p["mode"], p[k]) for p in plans for k in ("train_folder", "test_folder")})
    manifests = {folder: input_manifest(Path(folder)) for _, folder in unique}
    source_names = ("experiments_pbc7.py", "temporal_graph_dataset.py", "train_amr_dygformer.py", "tasks.py", "convert_to_pt.py",
                    "ablate_edge_weights.py")
    sources = {name: file_hash(repo / name) for name in source_names if (repo / name).is_file()}
    identity = {"version": VERSION, "addon_sha256": file_hash(Path(__file__)), "source_sha256": sources,
                "data_root": str(root), "plans": plans, "input_inventory": {k: digest(v) for k, v in manifests.items()},
                "original_metric_sha256": {p["original_metrics"]: file_hash(p["original_metrics"]) for p in plans if p["original_metrics"]},
                "parameters": params, "bootstrap": args.bootstrap, "software": software_versions(),
                "xgboost_version": xgboost.__version__}
    with output_lock(output):
        tasks = Tasks(output, identity)

        def snapshot(destination):
            shutil.copy2(__file__, destination / "experiments_xgboost.py")
            for name in sources:
                shutil.copy2(repo / name, destination / name)
            atomic_json(destination / "dataset_plan.json", plans)
            atomic_json(destination / "feature_contract.json", {
                "daily_feature_names": DAILY_FEATURES, "history_days": HISTORY, "window_features": 28,
                "count_ablation": {"steps": list(ABLATION_REFERENCES), "window_features": 14,
                    "retained_daily_features": ["patient_burden", "recent_events"],
                    "removed_daily_features": ["patient_admissions", "patient_discharges"],
                    "partitions": ["train", "validation", "test"],
                    "sources": "original canonical or shifted datasets according to the corresponding experiment",
                    "transfer_source": "canonical count-ablated model; no training on shifted data"},
                "feature_types": "Patient burden; admissions; discharges/turnover; recent events",
                "ground_truth_events": "Sum of original daily new_cr_acq_today and new_ir_inf_today flags over patients; can count two events for one patient",
                "admission_pressure": "Observed daily admission volume; no simulator importation-probability parameter",
                "first_snapshot_flows": "NaN when previous census/status is unavailable; no initial-census-as-admissions substitution",
                "temporal_order": "Oldest to newest day, four columns per day; no future information in predictors",
                "original_table_train_heading": "Validation partition in original driver",
                "preserved": "Saved labels, all original history windows, split seed and trajectory split, frozen external tests, threshold 0.5",
            })
        tasks.run("01_inputs", {"source_sha256": sources}, snapshot)
        cache_dirs = {}
        for mode, folder in unique:
            key = f"02_features/{mode}/{digest(folder)[:16]}"
            cache_dirs[(mode, folder)] = tasks.run(key, {"folder": folder, "manifest": digest(manifests[folder])},
                lambda dest, ff=folder, mm=mode: extract_dataset(Path(ff), mm, manifests[ff], dest))
        cache = {}
        def get_data(mode, folder):
            key = (mode, folder)
            if key not in cache:
                cache[key] = load_dataset(cache_dirs[key])
            return cache[key]
        completed, fit_registry, canonical_models, canonical_fingerprints = [], {}, {}, {}
        for plan in plans:
            mode = plan["mode"]
            train_data = apply_feature_selection(get_data(mode, plan["train_folder"]), plan)
            if plan.get("train_transform"):
                print(f"[source] {mode}/{plan['tag']}: original Step 5 transform from saved baseline ({plan['train_transform']})", flush=True)
            test_data = apply_feature_selection(get_data(mode, plan["test_folder"]), plan)
            X, y, samples = train_data
            train_idx, val_idx = original_split(samples, plan["split_seed"])
            parity = verify_original_labels(plan, y[val_idx], test_data[1])
            fit_fp = digest({"mode": mode, "data": array_hash(X, y, train_idx, val_idx), "parameters": params})
            if plan["tag"] == "step5_no_edge_weights":
                if fit_fp != canonical_fingerprints["step4_baseline"]:
                    raise RuntimeError(f"{mode}/{plan['tag']}: expected unchanged XGBoost inputs, but they differ from baseline. Check the supplied datasets.")
            transfer_source = plan.get("transfer_source_step")
            if transfer_source and fit_fp != canonical_fingerprints.get(transfer_source):
                raise RuntimeError(f"{plan['tag']}: canonical training inputs/split differ from source model {transfer_source}.")
            key = f"03_experiments/{mode}/{plan['tag']}"
            directory = tasks.run(key, {"plan": plan, "fit_fingerprint": fit_fp, "parity": parity},
                lambda dest, pp=plan, td=train_data, ed=test_data, fp=fit_fp, check=parity:
                    run_experiment(dest, pp, td, ed, params, args.bootstrap, fp,
                                   fit_registry.get(fp), canonical_models.get(pp.get("transfer_source_step")), check))
            completed.append(directory)
            if not plan["tag"].startswith("step9_"):
                fit_registry.setdefault(fit_fp, directory / "model.json")
            if plan["tag"] in TRANSFER_SOURCES.values():
                canonical_models[plan["tag"]] = directory / "model.json"
                canonical_fingerprints[plan["tag"]] = fit_fp
        tasks.run("04_tables", {"rows": len(plans)}, lambda dest: export_tables(dest, completed, plans))
        for folder, manifest in manifests.items():
            if input_manifest(Path(folder)) != manifest:
                raise RuntimeError(f"Original input directory changed during the run: {folder}")
    print(f"Complete: {output / '04_tables'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
