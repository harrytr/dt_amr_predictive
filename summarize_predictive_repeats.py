#!/usr/bin/env python3
"""Summarize independent predictive repeats without requiring torch/GPU access."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import roc_auc_score

from predictive_repeats import (METRICS, SPLITS, STEPS, atomic_json, digest, file_hash,
                                read_json, write_csv, run_lock)


def binary_metrics(y, probability):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0 or len(y) != len(probability):
        raise ValueError("Expected nonempty aligned labels and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid binary labels or probabilities")
    prediction = probability > 0.5  # Same argmax tie rule as the original GNN/XGB.
    tp = int(np.sum((y == 1) & prediction))
    tn = int(np.sum((y == 0) & ~prediction))
    fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction))
    f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {"n_windows": len(y), "positive": int(y.sum()), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "auroc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            "accuracy": (tp + tn) / len(y), "macro_f1": (f1_pos + f1_neg) / 2,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None}


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    y = np.asarray([int(r["y"]) for r in rows])
    p = np.asarray([float(r["probability"]) for r in rows])
    metrics = binary_metrics(y, p)
    if any(int(row["prediction"]) != int(prob > 0.5) for row, prob in zip(rows, p)):
        raise ValueError(f"Threshold or probability mismatch in {path}")
    if len({(r["sim_id"], r["day"]) for r in rows}) != len(rows):
        raise ValueError(f"Duplicate trajectory/day predictions in {path}")
    return rows, y, p, metrics


def mean_interval(values):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    n = len(finite)
    mean = float(finite.mean()) if n else None
    sd = float(finite.std(ddof=1)) if n >= 2 else None
    # Undefined metrics are not silently removed from the CI's estimand.
    estimable = n >= 2 and n == len(values)
    half = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if estimable else None
    return {"mean": mean, "sd": sd, "ci95_low": mean - half if half is not None else None,
            "ci95_high": mean + half if half is not None else None, "n_valid": n,
            "n_available": len(values), "all_metrics_defined": n == len(values),
            "ci_method": "two-sided Student t interval for the mean of independent joint repeats; untruncated"}


def trajectory_bootstrap(rows, y, p, repetitions, seed):
    # Whole trajectories, stratified by the pre-specified simulator family.
    # Overlapping temporal windows are always kept together.
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["sim_id"], []).append(i)
    families = {}
    for sid in sorted(groups):
        family = sid.split("__sim_", 1)[0]
        families.setdefault(family, []).append(sid)
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRICS}
    if len(groups) >= 2:
        for _ in range(repetitions):
            selected = []
            for family in sorted(families):
                ids = families[family]
                selected.extend(rng.choice(ids, len(ids), replace=True).tolist())
            idx = np.asarray([i for sid in selected for i in groups[sid]], dtype=int)
            m = binary_metrics(y[idx], p[idx])
            for key in METRICS:
                if m[key] is not None:
                    values[key].append(m[key])
    return {key: {"low": float(np.quantile(v, .025)) if v else None,
                  "high": float(np.quantile(v, .975)) if v else None,
                  "valid_bootstraps": len(v), "requested_bootstraps": repetitions,
                  "n_trajectories": len(groups), "n_families": len(families),
                  "method": "95% percentile whole-trajectory bootstrap within simulator family; conditional on fitted model"}
            for key, v in values.items()}


def verify_saved_metrics(folder, model, split, calculated):
    if model == "GNN":
        saved = read_json(folder / "metrics_summary.json")[split]["metrics"]
        names = {"auroc": "roc_auc", "macro_f1": "f1_macro"}
    else:
        saved = read_json(folder / "metrics.json")[split]
        names = {}
    for key in METRICS:
        old, current = saved.get(names.get(key, key)), calculated[key]
        if old is None and current is None:
            continue
        if old is None or current is None or not np.isclose(float(old), current, rtol=1e-7, atol=1e-10):
            raise ValueError(f"Stored metric/prediction mismatch: {folder} {split} {key}: {old} vs {current}")


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def export_latex(path, grouped_rows, expected_repeats, completed_repeats):
    lookup = {}
    for row in grouped_rows:
        lookup.setdefault((row["mode"], row["step"], row["model"], row["split"]), {})[row["metric"]] = row
    def cell(item):
        if item["mean"] is None:
            return "--"
        point = f"{item['mean']:.3f}"
        if item["ci95_low"] is not None:
            return point + f" ({item['ci95_low']:.3f}, {item['ci95_high']:.3f})"
        return point
    lines = [r"% Requires: \usepackage{booktabs,longtable,pdflscape}",
             r"\begin{landscape}", r"\begingroup\scriptsize",
             r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.12}",
             r"\begin{longtable}{@{}lllccccc c@{}}",
             r"\caption{Predictive performance across independent joint repetitions at $T=7$ and $H=7$. "
             + f"Completed repetitions: {completed_repeats}/{expected_repeats}. "
             + r"Values are means (95\% confidence intervals across repetitions).}\label{tab:predictive-repeats}\\",
             r"\toprule Experiment & Model & Split & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity & $n$ \\",
             r"\midrule\endfirsthead", r"\toprule Experiment & Model & Split & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity & $n$ \\",
             r"\midrule\endhead", r"\bottomrule\endfoot"]
    for mode in ("ground_truth", "partial_observation"):
        if not any(key[0] == mode for key in lookup):
            continue
        lines += [r"\multicolumn{9}{@{}l}{\textbf{" + tex_escape(mode.replace("_", " ").title()) + r"}}\\"]
        for step in STEPS:
            title = {STEPS[0]: "Canonical", STEPS[1]: "Shifted train/test", STEPS[2]: "Canonical-to-shift"}[step]
            for model in ("GNN", "XGB all summaries", "XGB without counts"):
                for split in SPLITS:
                    items = lookup.get((mode, step, model, split))
                    if not items:
                        continue
                    n = min(item["n_valid"] for item in items.values())
                    lines.append(" & ".join([title, model, split.title(), *[cell(items[k]) for k in METRICS], str(n)]) + r" \\")
                if (mode, step, model, "test") in lookup:
                    lines.append(r"\addlinespace[2pt]")
    lines += [r"\end{longtable}",
              r"\noindent Train denotes evaluation on the fitted training subset; validation denotes the held-out trajectories. "
              r"Step 9 reuses the corresponding Step 4 model and canonical train/validation sets without retraining. "
              r"The $n$ column is the minimum number of repeats with defined metrics in that row, not the number of windows. "
              r"Intervals use Student's $t$ distribution across independent joint repeats and are not truncated to $[0,1]$. "
              r"Whole-trajectory conditional test intervals are exported separately. "
              r"A partial run is provisional. Undefined metrics are not replaced or redrawn.",
              r"\endgroup", r"\end{landscape}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(output):
    output = Path(output).resolve()
    manifest = read_json(output / "run_manifest.json")
    identity = manifest["identity"]
    cfg = identity["config"]
    destination = output / "summary"
    destination.mkdir(exist_ok=True)
    per_repeat, conditional, predictions = [], [], {}
    completed = 0
    for rep in identity["repeats"]:
        r = rep["repeat"]
        root = output / f"repeat_{r:03d}"
        ready = root / "repeat_complete.json"
        if not ready.exists():
            continue
        if read_json(ready)["identity_sha256"] != manifest["identity_sha256"]:
            raise ValueError(f"Repeat identity mismatch: {root}")
        completed += 1
        tasks = [(mode, step, "GNN", root / "gnn" / mode / step) for mode in cfg["tracks"] for step in STEPS]
        if cfg["include_xgboost"]:
            tasks += [("ground_truth", step, f"XGB {label}", root / "xgb" / step / variant)
                      for step in STEPS for variant, label in (("all_summaries", "all summaries"), ("without_counts", "without counts"))]
        for mode, step, model, folder in tasks:
            task_marker = read_json(folder / "complete.json")
            for name, sha in task_marker["files"].items():
                if not (folder / name).is_file() or file_hash(folder / name) != sha:
                    raise ValueError(f"Completed result changed or is missing: {folder / name}")
            for split in SPLITS:
                path = folder / f"{split}_predictions.csv"
                rows, y, p, point = read_predictions(path)
                verify_saved_metrics(folder, model, split, point)
                # Pairing is checked by identity and label, not by coincident counts.
                keys = {(row["sim_id"], int(row["day"])): int(row["y"]) for row in rows}
                pair_key = (r, mode, step, split)
                if pair_key in predictions and predictions[pair_key] != keys:
                    raise ValueError(f"GNN/XGB pairing mismatch: {folder} {split}")
                predictions[pair_key] = keys
                meta = {"repeat": r, "mode": mode, "step": step, "model": model, "split": split,
                        "model_seed": rep["model_seed"], "parameter_seed": rep["parameter_seed"],
                        **{f"multiplier_{k}": v for k, v in rep["parameter_multipliers"].items()},
                        "n_trajectories": len({row["sim_id"] for row in rows})}
                per_repeat.append({**meta, **point})
                if split == "test":
                    # Reusing the same seed preserves cluster selections across
                    # matched learners, though these are marginal intervals.
                    seed = cfg["bootstrap_seed"] + r
                    fingerprint = digest({"prediction_sha256": file_hash(path), "repetitions": cfg["bootstrap_replicates"],
                                          "seed": seed, "method_version": 1})
                    cache = folder / "trajectory_intervals.json"
                    saved = read_json(cache) if cache.is_file() else {}
                    if saved.get("fingerprint") != fingerprint:
                        saved = {"fingerprint": fingerprint, "metrics": trajectory_bootstrap(rows, y, p, cfg["bootstrap_replicates"], seed)}
                        atomic_json(cache, saved)
                    conditional.extend({**meta, "metric": key, "point": point[key], **item}
                                       for key, item in saved["metrics"].items())
    grouped = {}
    for row in per_repeat:
        key = tuple(row[k] for k in ("mode", "step", "model", "split"))
        grouped.setdefault(key, []).append(row)
    summary = []
    expected = len(identity["repeats"])
    for key, rows in grouped.items():
        for metric in METRICS:
            summary.append({**dict(zip(("mode", "step", "model", "split"), key)), "metric": metric,
                            "n_requested": expected, "run_complete": completed == expected,
                            **mean_interval([row[metric] for row in rows])})
    pairs = []
    for mode in cfg["tracks"]:
        for step in STEPS:
            for split in SPLITS:
                gnn = {r["repeat"]: r for r in grouped.get((mode, step, "GNN", split), [])}
                for model in ("XGB all summaries", "XGB without counts"):
                    xgb = {r["repeat"]: r for r in grouped.get((mode, step, model, split), [])}
                    common = sorted(set(gnn) & set(xgb))
                    if not common:
                        continue
                    for metric in METRICS:
                        differences = [gnn[r][metric] - xgb[r][metric]
                                       if gnn[r][metric] is not None and xgb[r][metric] is not None else None for r in common]
                        pairs.append({"mode": mode, "step": step, "split": split, "comparison": f"GNN minus {model}",
                                      "metric": metric, "n_requested": expected, "run_complete": completed == expected,
                                      **mean_interval(differences)})
    write_csv(destination / "per_repeat_metrics.csv", per_repeat)
    write_csv(destination / "mean_metrics_ci95.csv", summary)
    write_csv(destination / "paired_gnn_minus_xgb_ci95.csv", pairs)
    write_csv(destination / "conditional_test_trajectory_ci95.csv", conditional)
    export_latex(destination / "combined_table.tex", summary, expected, completed)
    atomic_json(destination / "summary.json", {"completed_repeats": completed, "requested_repeats": expected,
                "status": "complete" if completed == expected else "partial",
                "summary": summary, "paired_differences": pairs,
                "interpretation": "Joint simulator-seed/model-initialization/parameter-draw variation. Components are not separately identified. Means equally weight repetitions. Five repeats give imprecise uncertainty estimates; inspect all individual repeats.",
                "parameter_rationale": cfg["parameter_rationale"],
                "scope": "Predictive Steps 4, 8, 9 only. No policy experiments, ablation sweep or architecture retuning."})
    print(f"[summary] {completed}/{expected} complete repeats: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Repeat results directory containing run_manifest.json")
    args = parser.parse_args()
    with run_lock(Path(args.output).resolve()):
        summarize(args.output)


if __name__ == "__main__":
    main()
