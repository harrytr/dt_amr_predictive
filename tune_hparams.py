#!/usr/bin/env python3
"""
tune_hparams.py

Lightweight hyperparameter tuner for train_amr_dygformer.py.

Purpose
- Runs a compact, validation-only hyperparameter search around a restricted
  neighbourhood of the default GraphSAGE + Transformer configuration.
- Intended for pre-training calibration before the main experimental driver,
  specifically for Step 4 baseline training and Step 8 distribution-shift
  training.
- Uses the training script's internal trajectory-level train/validation split;
  external test data are never used for model selection.

Search strategy
- Stage 1: random screen of N candidate configurations at a short epoch budget.
- Stage 2: rerun the top-K candidates at a fuller epoch budget.
- Selection: ranks by validation metric from metrics_summary.json
  (AUROC preferred, then F1 macro, then balanced accuracy, then accuracy).

Outputs
- trial_results.csv
- trial_results.json
- best_config.json
- best_config.txt
- search_summary.txt
- Per-trial folders with metrics summaries and checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# CLI helpers
# =============================================================================


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"yes", "true", "t", "y", "1"}:
        return True
    if s in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# =============================================================================
# Search space and ranking
# =============================================================================


@dataclass(frozen=True)
class TrialConfig:
    hidden: int
    sage_layers: int
    transformer_layers: int
    heads: int
    dropout: float
    lr: float
    batch_size: int


@dataclass
class TrialResult:
    stage: str
    trial_index: int
    trial_name: str
    status: str
    config: TrialConfig
    epochs_requested: int
    out_dir: str
    command: List[str]
    best_epoch: Optional[int] = None
    stopped_early: Optional[bool] = None
    epochs_completed: Optional[int] = None
    selected_metric_name: Optional[str] = None
    selected_metric_value: Optional[float] = None
    val_loss_best: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_f1_macro: Optional[float] = None
    val_balanced_accuracy: Optional[float] = None
    val_roc_auc: Optional[float] = None
    val_roc_auc_macro_ovr: Optional[float] = None
    error_message: Optional[str] = None


DEFAULT_SEARCH_SPACE: Dict[str, Sequence[Any]] = {
    "hidden": [32, 64],
    "sage_layers": [2, 3],
    "transformer_layers": [1, 2],
    "heads": [2, 4],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-5, 3e-5, 1e-4],
    "batch_size": [8, 16],
}


def _all_candidate_configs(search_space: Dict[str, Sequence[Any]]) -> List[TrialConfig]:
    keys = [
        "hidden",
        "sage_layers",
        "transformer_layers",
        "heads",
        "dropout",
        "lr",
        "batch_size",
    ]
    grid = itertools.product(*(search_space[k] for k in keys))
    out: List[TrialConfig] = []
    for values in grid:
        cfg = TrialConfig(**dict(zip(keys, values)))
        out.append(cfg)
    return out


def _pick_random_configs(
    search_space: Dict[str, Sequence[Any]],
    n_trials: int,
    rng: random.Random,
) -> List[TrialConfig]:
    all_cfgs = _all_candidate_configs(search_space)
    n_keep = min(max(1, int(n_trials)), len(all_cfgs))
    chosen = rng.sample(all_cfgs, n_keep)
    chosen.sort(key=lambda c: (
        c.hidden,
        c.sage_layers,
        c.transformer_layers,
        c.heads,
        c.dropout,
        c.lr,
        c.batch_size,
    ))
    return chosen


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _extract_selected_metric(summary: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    validation = summary.get("validation", {}) if isinstance(summary, dict) else {}
    metrics = validation.get("metrics", {}) if isinstance(validation, dict) else {}

    metric_priority = [
        "roc_auc_macro_ovr",
        "roc_auc",
        "f1_macro",
        "balanced_accuracy",
        "accuracy",
    ]

    for name in metric_priority:
        value = _safe_float(metrics.get(name))
        if value is not None:
            return name, value

    return "unavailable", None


def _extract_trial_result_from_summary(
    summary: Dict[str, Any],
    stage: str,
    trial_index: int,
    trial_name: str,
    cfg: TrialConfig,
    epochs_requested: int,
    out_dir: str,
    command: List[str],
) -> TrialResult:
    validation = summary.get("validation", {}) if isinstance(summary, dict) else {}
    metrics = validation.get("metrics", {}) if isinstance(validation, dict) else {}
    training = summary.get("training", {}) if isinstance(summary, dict) else {}

    selected_metric_name, selected_metric_value = _extract_selected_metric(summary)

    return TrialResult(
        stage=stage,
        trial_index=trial_index,
        trial_name=trial_name,
        status="ok",
        config=cfg,
        epochs_requested=int(epochs_requested),
        out_dir=str(out_dir),
        command=list(command),
        best_epoch=training.get("best_epoch"),
        stopped_early=training.get("stopped_early"),
        epochs_completed=training.get("epochs_completed"),
        selected_metric_name=selected_metric_name,
        selected_metric_value=selected_metric_value,
        val_loss_best=_safe_float(training.get("best_val_loss")),
        val_accuracy=_safe_float(metrics.get("accuracy")),
        val_f1_macro=_safe_float(metrics.get("f1_macro")),
        val_balanced_accuracy=_safe_float(metrics.get("balanced_accuracy")),
        val_roc_auc=_safe_float(metrics.get("roc_auc")),
        val_roc_auc_macro_ovr=_safe_float(metrics.get("roc_auc_macro_ovr")),
        error_message=None,
    )


def _trial_sort_key(res: TrialResult) -> Tuple[float, float, float, float, float]:
    selected = -1e18 if res.selected_metric_value is None else float(res.selected_metric_value)
    f1 = -1e18 if res.val_f1_macro is None else float(res.val_f1_macro)
    bal = -1e18 if res.val_balanced_accuracy is None else float(res.val_balanced_accuracy)
    acc = -1e18 if res.val_accuracy is None else float(res.val_accuracy)
    best_epoch_score = 1e18 if res.best_epoch is None else -float(res.best_epoch)
    return (selected, f1, bal, acc, best_epoch_score)


# =============================================================================
# Files and formatting
# =============================================================================


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _result_to_flat_row(res: TrialResult) -> Dict[str, Any]:
    cfg = asdict(res.config)
    row: Dict[str, Any] = {
        "stage": res.stage,
        "trial_index": res.trial_index,
        "trial_name": res.trial_name,
        "status": res.status,
        "epochs_requested": res.epochs_requested,
        "out_dir": res.out_dir,
        "best_epoch": res.best_epoch,
        "stopped_early": res.stopped_early,
        "epochs_completed": res.epochs_completed,
        "selected_metric_name": res.selected_metric_name,
        "selected_metric_value": res.selected_metric_value,
        "val_loss_best": res.val_loss_best,
        "val_accuracy": res.val_accuracy,
        "val_f1_macro": res.val_f1_macro,
        "val_balanced_accuracy": res.val_balanced_accuracy,
        "val_roc_auc": res.val_roc_auc,
        "val_roc_auc_macro_ovr": res.val_roc_auc_macro_ovr,
        "error_message": res.error_message,
        "command": " ".join(res.command),
    }
    row.update(cfg)
    return row


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# =============================================================================
# Trial execution
# =============================================================================


def _stream_subprocess_output(
    *,
    proc: subprocess.Popen,
    stdout_log_path: Path,
    stderr_log_path: Path,
) -> Tuple[int, str, str]:
    stdout_chunks: List[str] = []
    stderr_chunks: List[str] = []

    stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _reader(pipe: Any, sink, chunks: List[str], mirror) -> None:
        try:
            for line in iter(pipe.readline, ''):
                if line == '':
                    break
                sink.write(line)
                sink.flush()
                chunks.append(line)
                mirror.write(line)
                mirror.flush()
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    with stdout_log_path.open('w', encoding='utf-8') as stdout_f, stderr_log_path.open('w', encoding='utf-8') as stderr_f:
        stdout_thread = threading.Thread(
            target=_reader,
            args=(proc.stdout, stdout_f, stdout_chunks, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_reader,
            args=(proc.stderr, stderr_f, stderr_chunks, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        return_code = proc.wait()
        stdout_thread.join()
        stderr_thread.join()

    return return_code, ''.join(stdout_chunks), ''.join(stderr_chunks)


def _build_train_command(
    python_exe: str,
    train_script: str,
    args: argparse.Namespace,
    cfg: TrialConfig,
    epochs: int,
    out_dir: Path,
) -> List[str]:
    cmd: List[str] = [
        python_exe,
        train_script,
        "--data_folder",
        str(args.data_folder),
        "--task",
        str(args.task),
        "--T",
        str(args.T),
        "--sliding_step",
        str(args.sliding_step),
        "--hidden",
        str(cfg.hidden),
        "--heads",
        str(cfg.heads),
        "--dropout",
        str(cfg.dropout),
        "--transformer_layers",
        str(cfg.transformer_layers),
        "--sage_layers",
        str(cfg.sage_layers),
        "--batch_size",
        str(cfg.batch_size),
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(cfg.lr),
        "--max_neighbors",
        str(args.max_neighbors),
        "--neighbor_sampling",
        str(bool(args.neighbor_sampling)).lower(),
        "--num_neighbors",
        str(args.num_neighbors),
        "--seed_count",
        str(args.seed_count),
        "--seed_strategy",
        str(args.seed_strategy),
        "--seed_batch_size",
        str(args.seed_batch_size),
        "--max_sub_batches",
        str(args.max_sub_batches),
        "--attn_top_k",
        str(args.attn_top_k),
        "--attn_rank_by",
        str(args.attn_rank_by),
        "--emit_translational_figures",
        str(bool(args.emit_translational_figures)).lower(),
        "--fullgraph_attribution_pass",
        str(bool(args.fullgraph_attribution_pass)).lower(),
        "--translational_top_k",
        str(args.translational_top_k),
        "--split_seed",
        str(args.split_seed),
        "--train_model",
        "true",
        "--require_pt_metadata",
        str(bool(args.require_pt_metadata)).lower(),
        "--fail_on_noncontiguous",
        str(bool(args.fail_on_noncontiguous)).lower(),
        "--out_dir",
        str(out_dir),
        "--early_stopping",
        str(bool(args.early_stopping)).lower(),
        "--patience",
        str(args.patience),
        "--min_delta",
        str(args.min_delta),
        "--save_best_only",
        str(bool(args.save_best_only)).lower(),
        "--lr_scheduler_on_plateau",
        str(bool(args.lr_scheduler_on_plateau)).lower(),
        "--lr_scheduler_factor",
        str(args.lr_scheduler_factor),
        "--lr_scheduler_patience",
        str(args.lr_scheduler_patience),
        "--lr_scheduler_min_lr",
        str(args.lr_scheduler_min_lr),
    ]

    if args.test_folder:
        raise ValueError(
            "tune_hparams.py does not accept --test_folder. "
            "Hyperparameter tuning must remain validation-only to avoid frozen-test leakage."
        )
    if args.use_cls:
        cmd.append("--use_cls")
    if args.use_task_hparams:
        cmd.append("--use_task_hparams")

    return cmd


def _run_single_trial(
    python_exe: str,
    train_script: str,
    args: argparse.Namespace,
    cfg: TrialConfig,
    trial_index: int,
    stage: str,
    epochs: int,
    out_root: Path,
) -> TrialResult:
    trial_name = f"{stage}_trial_{trial_index:03d}"
    trial_out_dir = out_root / trial_name
    trial_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = _build_train_command(
        python_exe=python_exe,
        train_script=train_script,
        args=args,
        cfg=cfg,
        epochs=epochs,
        out_dir=trial_out_dir,
    )

    print(
        f"TUNE_TRIAL_START stage={stage} trial={trial_index} "
        f"hidden={cfg.hidden} sage_layers={cfg.sage_layers} "
        f"transformer_layers={cfg.transformer_layers} heads={cfg.heads} "
        f"dropout={cfg.dropout} lr={cfg.lr} batch_size={cfg.batch_size} epochs={epochs}",
        flush=True,
    )

    stdout_path = trial_out_dir / "stdout.log"
    stderr_path = trial_out_dir / "stderr.log"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    return_code, stdout_text, stderr_text = _stream_subprocess_output(
        proc=proc,
        stdout_log_path=stdout_path,
        stderr_log_path=stderr_path,
    )

    summary_path = trial_out_dir / "metrics_summary.json"
    if return_code != 0:
        err_tail = (stderr_text or stdout_text or f"Return code {return_code}")[-8000:]
        print(
            f"TUNE_TRIAL_FAILED stage={stage} trial={trial_index} return_code={return_code}",
            flush=True,
        )
        if err_tail:
            print(err_tail, file=sys.stderr, flush=True)
        return TrialResult(
            stage=stage,
            trial_index=trial_index,
            trial_name=trial_name,
            status="failed",
            config=cfg,
            epochs_requested=int(epochs),
            out_dir=str(trial_out_dir),
            command=cmd,
            error_message=err_tail,
        )

    if not summary_path.exists():
        return TrialResult(
            stage=stage,
            trial_index=trial_index,
            trial_name=trial_name,
            status="failed",
            config=cfg,
            epochs_requested=int(epochs),
            out_dir=str(trial_out_dir),
            command=cmd,
            error_message="Training completed but metrics_summary.json was not found.",
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        return TrialResult(
            stage=stage,
            trial_index=trial_index,
            trial_name=trial_name,
            status="failed",
            config=cfg,
            epochs_requested=int(epochs),
            out_dir=str(trial_out_dir),
            command=cmd,
            error_message=f"Failed to parse metrics_summary.json: {e}",
        )

    result = _extract_trial_result_from_summary(
        summary=summary,
        stage=stage,
        trial_index=trial_index,
        trial_name=trial_name,
        cfg=cfg,
        epochs_requested=epochs,
        out_dir=str(trial_out_dir),
        command=cmd,
    )

    sel = "NA" if result.selected_metric_value is None else f"{result.selected_metric_value:.6f}"
    print(
        f"TUNE_TRIAL_DONE stage={stage} trial={trial_index} "
        f"metric={result.selected_metric_name} value={sel} best_epoch={result.best_epoch}",
        flush=True,
    )
    return result


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--test_folder", type=str, default=None)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--T", type=int, default=7)
    parser.add_argument("--sliding_step", type=int, default=1)
    parser.add_argument("--search_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--train_script", type=str, default="train_amr_dygformer.py")
    parser.add_argument("--python_executable", type=str, default=sys.executable)

    parser.add_argument("--n_trials_quick", type=int, default=10)
    parser.add_argument("--n_finalists", type=int, default=3)
    parser.add_argument("--quick_epochs", type=int, default=12)
    parser.add_argument("--full_epochs", type=int, default=35)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=0)

    parser.add_argument("--max_neighbors", type=int, default=20)
    parser.add_argument("--neighbor_sampling", type=str2bool, default=True)
    parser.add_argument("--num_neighbors", type=str, default="15,10")
    parser.add_argument("--seed_count", type=int, default=256)
    parser.add_argument("--seed_strategy", type=str, default="random")
    parser.add_argument("--seed_batch_size", type=int, default=64)
    parser.add_argument("--max_sub_batches", type=int, default=4)

    parser.add_argument("--attn_top_k", type=int, default=50)
    parser.add_argument("--attn_rank_by", type=str, default="abs_diff", choices=["abs_diff", "mean"])
    parser.add_argument("--emit_translational_figures", type=str2bool, default=False)
    parser.add_argument("--fullgraph_attribution_pass", type=str2bool, default=True)
    parser.add_argument("--translational_top_k", type=int, default=20)
    parser.add_argument("--use_task_hparams", action="store_true")
    parser.add_argument("--use_cls", action="store_true")
    parser.add_argument("--require_pt_metadata", type=str2bool, default=True)
    parser.add_argument("--fail_on_noncontiguous", type=str2bool, default=True)

    parser.add_argument("--early_stopping", type=str2bool, default=True)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_best_only", type=str2bool, default=True)
    parser.add_argument("--lr_scheduler_on_plateau", type=str2bool, default=True)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6)

    args = parser.parse_args()

    if args.test_folder is not None and str(args.test_folder).strip() != "":
        raise ValueError(
            "--test_folder is not allowed in tune_hparams.py. "
            "Use validation-only tuning and reserve the external test set for the final locked model evaluation."
        )

    rng = random.Random(int(args.random_seed))
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = (Path.cwd() / train_script_path).resolve()
    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    metadata = {
        "search_name": args.search_name,
        "data_folder": str(Path(args.data_folder).resolve()),
        "test_folder": None if args.test_folder is None else str(Path(args.test_folder).resolve()),
        "task": args.task,
        "T": int(args.T),
        "sliding_step": int(args.sliding_step),
        "n_trials_quick": int(args.n_trials_quick),
        "n_finalists": int(args.n_finalists),
        "quick_epochs": int(args.quick_epochs),
        "full_epochs": int(args.full_epochs),
        "random_seed": int(args.random_seed),
        "split_seed": int(args.split_seed),
        "train_script": str(train_script_path),
        "python_executable": args.python_executable,
    }
    _write_json(out_root / "search_metadata.json", metadata)

    search_space = DEFAULT_SEARCH_SPACE
    quick_candidates = _pick_random_configs(
        search_space=search_space,
        n_trials=args.n_trials_quick,
        rng=rng,
    )

    _write_json(
        out_root / "quick_candidates.json",
        {"candidates": [asdict(c) for c in quick_candidates]},
    )

    all_results: List[TrialResult] = []

    print(
        f"TUNE_META name={args.search_name} quick_trials={len(quick_candidates)} "
        f"finalists={args.n_finalists} quick_epochs={args.quick_epochs} full_epochs={args.full_epochs}",
        flush=True,
    )

    # ---------------------------------------------------------------------
    # Stage 1: quick screen
    # ---------------------------------------------------------------------
    for idx, cfg in enumerate(quick_candidates, start=1):
        res = _run_single_trial(
            python_exe=args.python_executable,
            train_script=str(train_script_path),
            args=args,
            cfg=cfg,
            trial_index=idx,
            stage="quick",
            epochs=int(args.quick_epochs),
            out_root=out_root,
        )
        all_results.append(res)

    ok_quick = [r for r in all_results if r.stage == "quick" and r.status == "ok" and r.selected_metric_value is not None]
    ok_quick.sort(key=_trial_sort_key, reverse=True)

    if len(ok_quick) == 0:
        _write_text(
            out_root / "search_summary.txt",
            "No successful quick trials produced a usable validation metric.\n",
        )
        _write_json(out_root / "trial_results.json", [_result_to_flat_row(r) for r in all_results])
        _write_csv(
            out_root / "trial_results.csv",
            [_result_to_flat_row(r) for r in all_results],
            fieldnames=list(_result_to_flat_row(all_results[0]).keys()) if all_results else ["stage"],
        )
        return 1

    n_finalists = min(max(1, int(args.n_finalists)), len(ok_quick))
    finalists_cfgs = [r.config for r in ok_quick[:n_finalists]]
    _write_json(out_root / "finalists_from_quick.json", {"finalists": [asdict(c) for c in finalists_cfgs]})

    print(f"TUNE_FINALISTS count={n_finalists}", flush=True)

    # ---------------------------------------------------------------------
    # Stage 2: finalists at fuller budget
    # ---------------------------------------------------------------------
    full_results: List[TrialResult] = []
    for idx, cfg in enumerate(finalists_cfgs, start=1):
        res = _run_single_trial(
            python_exe=args.python_executable,
            train_script=str(train_script_path),
            args=args,
            cfg=cfg,
            trial_index=idx,
            stage="final",
            epochs=int(args.full_epochs),
            out_root=out_root,
        )
        all_results.append(res)
        full_results.append(res)

    ok_final = [r for r in full_results if r.status == "ok" and r.selected_metric_value is not None]
    ok_final.sort(key=_trial_sort_key, reverse=True)

    if len(ok_final) == 0:
        best_result = ok_quick[0]
        best_stage = "quick"
    else:
        best_result = ok_final[0]
        best_stage = "final"

    best_cfg = asdict(best_result.config)
    best_payload = {
        "search_name": args.search_name,
        "selected_from_stage": best_stage,
        "selected_metric_name": best_result.selected_metric_name,
        "selected_metric_value": best_result.selected_metric_value,
        "best_epoch": best_result.best_epoch,
        "stopped_early": best_result.stopped_early,
        "epochs_completed": best_result.epochs_completed,
        "config": best_cfg,
        "train_script": str(train_script_path),
        "data_folder": str(Path(args.data_folder).resolve()),
        "test_folder": None if args.test_folder is None else str(Path(args.test_folder).resolve()),
        "task": args.task,
        "T": int(args.T),
        "sliding_step": int(args.sliding_step),
        "split_seed": int(args.split_seed),
    }
    _write_json(out_root / "best_config.json", best_payload)

    best_txt_lines = [
        f"search_name: {args.search_name}",
        f"selected_from_stage: {best_stage}",
        f"selected_metric_name: {best_result.selected_metric_name}",
        f"selected_metric_value: {best_result.selected_metric_value}",
        f"best_epoch: {best_result.best_epoch}",
        f"stopped_early: {best_result.stopped_early}",
        f"epochs_completed: {best_result.epochs_completed}",
        "config:",
    ]
    for k, v in best_cfg.items():
        best_txt_lines.append(f"  {k}: {v}")
    _write_text(out_root / "best_config.txt", "\n".join(best_txt_lines) + "\n")

    flat_rows = [_result_to_flat_row(r) for r in all_results]
    _write_json(out_root / "trial_results.json", flat_rows)
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
    else:
        fieldnames = ["stage"]
    _write_csv(out_root / "trial_results.csv", flat_rows, fieldnames)

    summary_lines = [
        f"search_name: {args.search_name}",
        f"data_folder: {Path(args.data_folder).resolve()}",
        f"task: {args.task}",
        f"T: {args.T}",
        f"sliding_step: {args.sliding_step}",
        f"quick_trials_requested: {args.n_trials_quick}",
        f"quick_trials_successful: {len(ok_quick)}",
        f"finalists_requested: {args.n_finalists}",
        f"final_trials_successful: {len(ok_final)}",
        f"selected_from_stage: {best_stage}",
        f"selected_metric_name: {best_result.selected_metric_name}",
        f"selected_metric_value: {best_result.selected_metric_value}",
        f"best_epoch: {best_result.best_epoch}",
        f"stopped_early: {best_result.stopped_early}",
        "selected_config:",
    ]
    for k, v in best_cfg.items():
        summary_lines.append(f"  {k}: {v}")
    _write_text(out_root / "search_summary.txt", "\n".join(summary_lines) + "\n")

    print(
        f"TUNE_BEST stage={best_stage} metric={best_result.selected_metric_name} "
        f"value={best_result.selected_metric_value}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
