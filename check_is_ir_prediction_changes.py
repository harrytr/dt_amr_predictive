#!/usr/bin/env python3
"""Compare existing GT predictions with the IS-to-IR label-audit bounds.

Python standard library only; no model loading, training or dataset mutation.
Inputs may be the original audit/model directories or the supplied ZIP bundles.
Default run from the AMR directory: python check_is_ir_prediction_changes.py

This script is specific to the original predictive Steps 4, 8 and 9, T=7,
stride=1, and the supplied temporal_graph_dataset.py evaluation ordering.
Saved summaries contain positional predictions, not window IDs. Their order
is reconstructed from natural-sorted filenames and trajectory/day grouping,
then checked against EVERY old audit label, sample count, saved accuracy and
confusion matrix. These checks cannot independently prove identity if a
different dataset with coincidentally identical labels was substituted.

Accuracy bounds conservatively allow unresolved window labels to vary
independently. Overlapping windows share events, so the extremes need not be
jointly attainable. These are not confidence intervals. The script accepts
the event-reconstruction bounds from the audit; it does not rederive them.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


STEPS = (
    (4, "step4_baseline", "canonical"),
    (8, "step8_distribution_shift", "shifted"),
    (9, "step9_baseline_to_shift_test", "shifted"),
)
FILE_PATTERN = re.compile(r"^(.*)_t(\d+)(?:_L\d+)?\.pt$")


def natural_key(value):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", value)]


def basename(value):
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def read_unique(root, suffix):
    """Read exactly one matching file, without extracting ZIP contents."""
    root = Path(root)
    if root.is_file() and zipfile.is_zipfile(root):
        with zipfile.ZipFile(root) as archive:
            matches = [n for n in archive.namelist()
                       if not n.startswith("__MACOSX/")
                       and (n == suffix or n.endswith("/" + suffix))]
            if len(matches) != 1:
                raise ValueError(f"{root}: expected one {suffix}; found {len(matches)}")
            raw = archive.read(matches[0])
            location = f"{root.resolve()}::{matches[0]}"
    elif root.is_dir():
        matches = [p for p in root.rglob(suffix.rsplit("/", 1)[-1])
                   if p.is_file() and (p.as_posix().endswith("/" + suffix)
                                       or p.as_posix() == suffix)]
        if len(matches) != 1:
            raise ValueError(f"{root}: expected one {suffix}; found {len(matches)}")
        raw = matches[0].read_bytes()
        location = str(matches[0].resolve())
    else:
        raise ValueError(f"Input is not an existing directory or ZIP: {root}")
    return raw, {"location": location, "sha256": hashlib.sha256(raw).hexdigest()}


def binary(value, name):
    if value not in (0, 1, "0", "1"):
        raise ValueError(f"{name}: expected a binary value; got {value!r}")
    return int(value)


def ordered_windows(rows, dataset):
    """Reproduce trajectory insertion order from natural-sorted filenames."""
    selected = [dict(r) for r in rows if r["dataset"] == dataset]
    if not selected:
        raise ValueError(f"No audit rows for dataset {dataset}")
    selected.sort(key=lambda r: natural_key(basename(r["source_file"])))
    groups = {}
    prefixes = defaultdict(set)
    source_names = set()
    for row in selected:
        name = basename(row["source_file"])
        match = FILE_PATTERN.fullmatch(name)
        if not match or int(match[2]) != int(row["day"]):
            raise ValueError(f"Unrecognised filename or inconsistent day: {name}")
        if name in source_names:
            raise ValueError(f"Duplicate source filename: {name}")
        source_names.add(name)
        for field in ("old_label", "new_label_lower", "new_label_upper"):
            row[field] = binary(row[field], field)
        old, lower, upper = (row[k] for k in
                             ("old_label", "new_label_lower", "new_label_upper"))
        if not old <= lower <= upper:
            raise ValueError(f"Non-monotone label bounds: {name}")
        if row["horizon_status"] not in ("complete", "truncated"):
            raise ValueError(f"Unknown horizon status: {name}")
        future_days = int(row["future_days"])
        if not 0 <= future_days <= 7 or (row["horizon_status"] == "complete") != (future_days == 7):
            raise ValueError(f"Inconsistent seven-day horizon: {name}")
        if int(row["definitely_changed"]) != int(lower != old):
            raise ValueError(f"Inconsistent definite-change flag: {name}")
        if int(row["possibly_changed"]) != int(upper != old):
            raise ValueError(f"Inconsistent possible-change flag: {name}")
        row["day"] = int(row["day"])
        groups.setdefault(row["trajectory"], []).append(row)
        prefixes[row["trajectory"]].add(match[1])
    ordered = []
    for trajectory, group in groups.items():
        if len(prefixes[trajectory]) != 1:
            raise ValueError(f"Multiple filename prefixes in trajectory: {trajectory}")
        group.sort(key=lambda r: r["day"])
        days = [r["day"] for r in group]
        if days != list(range(7, days[-1] + 1)):
            raise ValueError(f"Expected consecutive windows starting at day 7: {trajectory}")
        ordered.extend(group)
    return ordered


def match_predictions(document, rows, step):
    if document.get("task") != "endogenous_importation_majority_h7":
        raise ValueError(f"Step {step}: unexpected prediction task")
    config = document["config"]
    if config.get("T") != 7 or config.get("sliding_step") != 1:
        raise ValueError(f"Step {step}: requires T=7 and sliding_step=1")
    if "TRACK_ground_truth" not in str(config.get("test_folder", "")):
        raise ValueError(f"Step {step}: this script requires the original GT summaries")
    test = document["test"]
    truth = [binary(x, "y_true") for x in test["y_true"]]
    predictions = [binary(x, "y_pred") for x in test["y_pred"]]
    probabilities = test["probabilities"]
    if not len(rows) == len(truth) == len(predictions) == len(probabilities) == test["n_samples"]:
        raise ValueError(f"Step {step}: audit/prediction sample counts differ")
    if truth != [r["old_label"] for r in rows]:
        mismatches = sum(y != r["old_label"] for y, r in zip(truth, rows))
        raise ValueError(f"Step {step}: {mismatches} old labels differ; refusing positional matching")
    matrix = [[0, 0], [0, 0]]
    paired = []
    for index, (row, y, prediction, probability) in enumerate(zip(rows, truth, predictions, probabilities)):
        if len(probability) != 2 or not all(math.isfinite(p) and 0 <= p <= 1 for p in probability):
            raise ValueError(f"Step {step}: invalid two-class probability at row {index}")
        if not math.isclose(sum(probability), 1.0, abs_tol=1e-5):
            raise ValueError(f"Step {step}: probabilities do not sum to one")
        if prediction != int(probability[1] > probability[0]):
            raise ValueError(f"Step {step}: prediction does not match original argmax rule")
        matrix[y][prediction] += 1
        definite = row["new_label_lower"] != y
        unresolved = row["new_label_lower"] != row["new_label_upper"]
        group = "confirmed_flip" if definite else "unresolved_flip" if unresolved else "unchanged"
        paired.append({
            "step": step, "dataset": row["dataset"], "prediction_index": index,
            "trajectory": row["trajectory"], "day": row["day"],
            "source_file": row["source_file"], "horizon_status": row["horizon_status"],
            "old_label": y, "new_label_lower": row["new_label_lower"],
            "new_label_upper": row["new_label_upper"], "predicted_label": prediction,
            "endogenous_probability": probability[1], "change_status": group,
            "originally_correct": int(y == prediction),
            "accuracy_count_change_if_flipped": (1 if prediction == 1 else -1) if group != "unchanged" else 0,
        })
    if matrix != test["confusion_matrix"]:
        raise ValueError(f"Step {step}: saved confusion matrix does not match predictions")
    accuracy = (matrix[0][0] + matrix[1][1]) / len(rows)
    if not math.isclose(accuracy, test["metrics"]["accuracy"], abs_tol=1e-10):
        raise ValueError(f"Step {step}: saved accuracy does not match predictions")
    return paired


def summarise(rows, step, scope):
    n = len(rows)
    if not n:
        raise ValueError(f"Step {step}: empty {scope} subset")
    old_correct = sum(r["originally_correct"] for r in rows)
    counts = {}
    for label, status in (("confirmed", "confirmed_flip"), ("unresolved", "unresolved_flip")):
        group = [r for r in rows if r["change_status"] == status]
        counts[label + "_flips"] = len(group)
        counts[label + "_errors_corrected"] = sum(r["predicted_label"] == 1 for r in group)
        counts[label + "_correct_predictions_lost"] = sum(r["predicted_label"] == 0 for r in group)
    definite_delta = counts["confirmed_errors_corrected"] - counts["confirmed_correct_predictions_lost"]
    minimum_delta = definite_delta - counts["unresolved_correct_predictions_lost"]
    maximum_delta = definite_delta + counts["unresolved_errors_corrected"]
    return {
        "step": step, "horizon_status": scope, "windows": n,
        "original_correct": old_correct, "original_accuracy_percent": 100 * old_correct / n,
        **counts, "confirmed_net_correct_change": definite_delta,
        "accuracy_confirmed_flips_only_percent": 100 * (old_correct + definite_delta) / n,
        "net_correct_change_lower": minimum_delta, "net_correct_change_upper": maximum_delta,
        "revised_accuracy_lower_percent": 100 * (old_correct + minimum_delta) / n,
        "revised_accuracy_upper_percent": 100 * (old_correct + maximum_delta) / n,
        "accuracy_change_lower_percentage_points": 100 * minimum_delta / n,
        "accuracy_change_upper_percentage_points": 100 * maximum_delta / n,
    }


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audit", default="IS_IR_audit_original_GT_refined", help="Audit directory or IS_IR_review_bundle.zip")
    parser.add_argument("--models", default="experiments_results/TRACK_ground_truth/work/repro_artifacts_steps_1_9", help="Archived model-summary directory or IS_IR_model_settings.zip")
    parser.add_argument("--output", default="IS_IR_prediction_changes", help="New output directory; existing paths are refused")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose another --output")
    raw, audit_source = read_unique(args.audit, "windows.csv")
    audit_rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    raw_meta, meta_source = read_unique(args.audit, "audit.json")
    audit_meta = json.loads(raw_meta)
    audit_config = audit_meta["configuration"]
    if any(audit_config.get(k) != v for k, v in
           (("history", 7), ("horizon", 7), ("stride", 1), ("format", "pt"), ("pt_track", "ground_truth"))):
        raise ValueError("Requires the refined GT tensor audit with history=7, horizon=7, stride=1")
    ordered = {dataset: ordered_windows(audit_rows, dataset) for dataset in ("canonical", "shifted")}
    summary, details, sources = [], [], [audit_source, meta_source]
    for step, folder, dataset in STEPS:
        raw_model, model_source = read_unique(args.models, f"{folder}/T7/metrics_summary.json")
        document = json.loads(raw_model)
        paired = match_predictions(document, ordered[dataset], step)
        details.extend(paired)
        sources.append(model_source)
        for scope in ("all", "complete", "truncated"):
            subset = paired if scope == "all" else [r for r in paired if r["horizon_status"] == scope]
            summary.append(summarise(subset, step, scope))
    report = {
        "sources": sources,
        "alignment": "Supplied dataset order reconstructed from natural-sorted filenames, trajectory grouping and numeric days; every original label, sample count, confusion matrix and accuracy matched. No per-window IDs are stored in these model summaries.",
        "interpretation": [
            "Uses saved predictions at their original decision rule; no retraining or threshold tuning.",
            "Confirmed-only accuracy applies definite flips and leaves unresolved labels unchanged; it is a scenario, not an exact revised accuracy.",
            "Unresolved counts describe what would happen IF those windows flipped, not confirmed changes.",
            "Accuracy bounds allow unresolved labels to vary independently; shared events across overlapping windows may make extremes unattainable.",
            "Bounds are conditional on the supplied event audit and are not confidence intervals.",
            "Steps 8 and 9 share test windows; do not sum their label-change counts as distinct windows.",
            "Only the original GT cohorts are checked, not partial observation or repeated runs. AUROC is not assessed by this script.",
        ],
        "summary": summary,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "window_predictions.csv", details)
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for row in summary:
        if row["horizon_status"] != "all":
            continue
        print(f"Step {row['step']}: confirmed flips correct {row['confirmed_errors_corrected']} errors "
              f"and lose {row['confirmed_correct_predictions_lost']} correct predictions.")
        print(f"  Unresolved flips could correct {row['unresolved_errors_corrected']} errors "
              f"or lose {row['unresolved_correct_predictions_lost']} correct predictions.")
        print(f"  Original accuracy {row['original_accuracy_percent']:.4f}%; revised accuracy bounds "
              f"{row['revised_accuracy_lower_percent']:.4f}–{row['revised_accuracy_upper_percent']:.4f}%")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
        print(f"CHECK STOPPED: {error}", file=sys.stderr)
        sys.exit(1)
