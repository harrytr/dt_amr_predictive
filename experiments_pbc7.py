#!/usr/bin/env python3
"""
experiments_pbc7.py

python experiments_pbc7.py \
  --run_both_state_modes \
  --emit_latex \
  --run_all_T \
  --archive_train_test_folders \
  --keep_step_train_graphml \
  --tune_step4 \
  --tune_step8


Experiment orchestration pipeline for the AMR digital-twin prediction framework.

This driver coordinates multi-step experimental runs across one or both state
modes, including data generation, graph conversion, model training, evaluation,
figure production, and LaTeX export. It supports running complete pipelines from
scratch as well as resuming from intermediate steps.

Main capabilities
- Runs predefined experiment stages over canonical, ablation, observation-shift,
  and epidemiological-shift settings.
- Supports single-track or dual-track execution across state modes.
- Preserves and reuses a frozen external test benchmark where required by the
  experimental design.
- Optionally archives train/test folders and selected intermediate artefacts.
- Produces structured figures, summary outputs, and LaTeX-ready report material.
- Supports lightweight hyperparameter tuning for Step 4 baseline training and
  Step 8 distribution-shift training through ``tune_hparams.py``.

Execution control
- Supports step-wise execution with configurable start and stop points.
- Supports repeated runs across multiple temporal horizons.
- Supports optional emission of LaTeX summaries and export bundles for reporting.
- Supports controlled retention of GraphML artefacts for downstream visualisation
  and figure generation.
- Supports evaluation-only repair mode via ``--no_train``. In this mode the
  script reuses existing datasets and trained checkpoints, skips all data
  generation/conversion stages (including Steps 1--2), and only reruns the
  required evaluation/export logic.
- Supports training-without-regeneration mode via ``--no_simulation``. In this
  mode the script reuses all existing datasets across Steps 1--7, skips every
  simulation/data-generation/conversion stage, and still retrains/evaluates the
  requested model runs on the existing prepared datasets.
- Supports optional pre-training calibration for Step 4 and Step 8 using a
  validation-only lightweight search. Step 4 tuned settings can be propagated to
  Steps 5--7 to preserve a common calibrated baseline architecture, Step 8 keeps
  its own tuned configuration, and Step 9 evaluates the Step 4 trained model on
  the Step 8 shifted test set.

Outputs
Depending on the selected options, the script can generate:
- training and evaluation artefacts
- archived train/test graph folders
- figures and comparison panels
- metrics summaries
- LaTeX tables and figure blocks
- reproducibility-oriented experiment folders
- tuning outputs for Step 4 / Step 8 when tuning is enabled

Examples

1) Run both state modes from scratch through Step 9:
   python experiments_pbc3.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders

2) Resume both state modes from Step 6.2 through Step 9:
   python experiments_pbc3.py --run_both_state_modes --start 6.2 --stop 9 --emit_latex --run_all_T

3) Reuse all existing datasets and retrain only the models from Step 4 through Step 9:
   python experiments_pbc3.py --run_both_state_modes --start 4 --stop 9 --no_simulation --emit_latex --run_all_T

4) Reuse all existing datasets and checkpoints to rerun evaluation/export only:
   python experiments_pbc3.py --run_both_state_modes --start 4 --stop 9 --no_train --emit_latex --run_all_T

5) Run Step 4 baseline with lightweight tuning:
   python experiments_pbc3.py --run_both_state_modes --start 4 --stop 4 --tune_step4

6) Run Step 8 distribution shift with lightweight tuning:
   python experiments_pbc3.py --run_both_state_modes --start 8 --stop 8 --tune_step8
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable
# =============================================================================
# CONFIG (EDIT HERE)
# =============================================================================

# Canonical surveillance policy used by the baseline benchmark and by
# downstream perturbation steps unless that step explicitly varies one of these
# fields.  Step 6 changes only screen_result_delay_days or screen_every_k_days
# relative to this policy.
CANONICAL_SURVEILLANCE_EXTRA_ARGS: List[str] = [
    "--screen_every_k_days", "7",
    "--screen_result_delay_days", "2",
    "--screen_on_admission", "1",
    "--persist_observations", "1",
]

CONFIG: Dict[str, Any] = {
    "SIM": {
        "num_regions": 1,
        "num_wards": 10,
        "num_patients": 200,
        "num_staff": 300,
        "export_yaml": True,
        "export_gif": False,
        "override_staff_wards_per_staff": True,
        "staff_wards_per_staff": 2,
        "enable_superspreader": False,
        "superspreader_staff": "",
        "superspreader_state": "IR",
        "superspreader_staff_contacts": 50,
        "superspreader_patient_frac_mult": 3.0,
        "superspreader_patient_min_add": 10,
        "superspreader_edge_weight_mult": 1.5,
        "superspreader_start_day": 1,
        "superspreader_end_day": 9999,
        "enable_admit_import_seasonality": False,
        "admit_import_seasonality": "none",
        "admit_import_period_days": 7,
        "admit_import_pmax_cs": 1.0,
        "admit_import_pmax_cr": 1.0,
        "admit_import_pmax_is": 1.0,
        "admit_import_pmax_ir": 1.0,
        "admit_import_amp": 0.5,
        "admit_import_phase_day": 0,
        "admit_import_high_start_day": 1,
        "admit_import_high_end_day": 180,
        "admit_import_high_mult": 1.5,
        "admit_import_low_mult": 1.0,
        "admit_import_shock_min_days": 7,
        "admit_import_shock_max_days": 30,
        "admit_import_shock_mult_min": 1.5,
        "admit_import_shock_mult_max": 5.0,
        "beta_res_mult": 1.0,
        "beta_state_mult_cs": 0.5,
        "beta_state_mult_cr": 0.5,
        "beta_state_mult_is": 1.0,
        "beta_state_mult_ir": 1.0,
        "crossward_hand_hygiene_enabled": 1,
        "crossward_hand_hygiene_mult": 0.55,
        "passive_screen_capacity_per_week": 0,
        "active_symptom_screening": 1,
        "admission_screen_symptomatic_only": 1,
        "isolation_trigger_policy": "resistant_only",
    },
    "CONVERT": {
        "horizons": "7",
        "workers": None,
    },
    "MODEL": {
        "use_task_hparams": False,
        "train_model": True,
        "task": "endogenous_importation_majority_h7",
        # "task": "early_outbreak_warning_h14",
        "pred_horizon": 7,
        "early_outbreak_fixed_threshold": 0.55,
        "T": 7,
        "T_list": "7",
        "sliding_step": 1,
        "hidden": 64,
        "heads": 2,
        "dropout": 0.3,
        "transformer_layers": 2,
        "sage_layers": 2,
        "use_cls": False,
        "batch_size": 16,
        "epochs": 50,
        "lr": 1e-4,
        "max_neighbors": 20,
        "neighbor_sampling": True,
        "num_neighbors": "15,10",
        "seed_count": 256,
        "seed_strategy": "random",
        "seed_batch_size": 64,
        "max_sub_batches": 4,
        "attn_top_k": 10,
        "attn_rank_by": "abs_diff",
        "fullgraph_attribution_pass": True,
        "emit_translational_figures": True,
        "translational_top_k": 20,
        "early_stopping": True,
        "patience": 7,
        "min_delta": 1e-4,
        "save_best_only": True,
        "lr_scheduler_on_plateau": True,
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 3,
        "lr_scheduler_min_lr": 1e-6,
    },
    "TEST": {
        "test_frac_per_class": 0.5,
        "min_per_class": 10,
        "seed": 1337,
        "balance_tolerance": 1,
        "require_balanced_test": False,
    },
    "STEP1": {
        "n_sims_per_trajectory": 10,
        "num_days": 60,
        "outbreak_expected_label_min_frac": 0.3,
        "outbreak_require_two_class_pooled_baseline": True,
        "trajectories": {
            "endog_high_train": {
                "seed_base": 4100,
                "p_admit_import_cs": 0.005,
                "p_admit_import_cr": 0.005,
                "p_admit_import_is": 0.005,
                "p_admit_import_ir": 0.005,
                "daily_discharge_frac": 0.02,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
            },
            "import_high_train": {
                "seed_base": 5100,
                "p_admit_import_cs": 0.6,
                "p_admit_import_cr": 0.6,
                "p_admit_import_is": 0.6,
                "p_admit_import_ir": 0.6,
                "daily_discharge_frac": 0.25,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
            },
            "endog_high_test": {
                "seed_base": 6100,
                "p_admit_import_cs": 0.005,
                "p_admit_import_cr": 0.005,
                "p_admit_import_is": 0.005,
                "p_admit_import_ir": 0.005,
                "daily_discharge_frac": 0.02,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
            },
            "import_high_test": {
                "seed_base": 7100,
                "p_admit_import_cs": 0.6,
                "p_admit_import_cr": 0.6,
                "p_admit_import_is": 0.6,
                "p_admit_import_ir": 0.6,
                "daily_discharge_frac": 0.25,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
            },
        },
    },
     "STEP8": {
        "num_days": 360,
        "n_sims_per_trajectory": 12,
        "max_majority_frac": 0.70,
        "continue_to_step9_on_balance_failure": True,
        "train_trajectories": {
            "shift_train_endog_a": {
                "seed_base": 8100,
                "p_admit_import_cs": 0.010,
                "p_admit_import_cr": 0.010,
                "p_admit_import_is": 0.010,
                "p_admit_import_ir": 0.010,
                "daily_discharge_frac": 0.04,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "30",
                    "--superspreader_end_day", "240",
                    "--superspreader_patient_frac_mult", "4.0",
                    "--superspreader_patient_min_add", "15",
                    "--superspreader_staff_contacts", "80",
                    "--superspreader_edge_weight_mult", "2.5",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", "360",
                    "--admit_import_high_start_day", "60",
                    "--admit_import_high_end_day", "180",
                    "--admit_import_high_mult", "1.2",
                    "--admit_import_low_mult", "0.8",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
                "shift_train_import_a": {
                "seed_base": 9100,
                "p_admit_import_cs": 0.200,
                "p_admit_import_cr": 0.200,
                "p_admit_import_is": 0.200,
                "p_admit_import_ir": 0.200,
                "daily_discharge_frac": 0.12,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s1",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "90",
                    "--superspreader_end_day", "330",
                    "--superspreader_patient_frac_mult", "3.5",
                    "--superspreader_patient_min_add", "12",
                    "--superspreader_staff_contacts", "70",
                    "--superspreader_edge_weight_mult", "2.0",
                    "--admit_import_seasonality", "sinusoid",
                    "--admit_import_period_days", "360",
                    "--admit_import_amp", "0.30",
                    "--admit_import_phase_day", "30",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
        },
        "test_trajectories": {
            "shift_test_endog_a": {
                "seed_base": 10100,
                "p_admit_import_cs": 0.012,
                "p_admit_import_cr": 0.012,
                "p_admit_import_is": 0.012,
                "p_admit_import_ir": 0.012,
                "daily_discharge_frac": 0.05,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "45",
                    "--superspreader_end_day", "270",
                    "--superspreader_patient_frac_mult", "4.5",
                    "--superspreader_patient_min_add", "18",
                    "--superspreader_staff_contacts", "90",
                    "--superspreader_edge_weight_mult", "2.6",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", "360",
                    "--admit_import_high_start_day", "75",
                    "--admit_import_high_end_day", "210",
                    "--admit_import_high_mult", "1.3",
                    "--admit_import_low_mult", "0.8",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
            "shift_test_import_a": {
                "seed_base": 11100,
                "p_admit_import_cs": 0.140,
                "p_admit_import_cr": 0.140,
                "p_admit_import_is": 0.140,
                "p_admit_import_ir": 0.140,
                "daily_discharge_frac": 0.10,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s1",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "120",
                    "--superspreader_end_day", "345",
                    "--superspreader_patient_frac_mult", "4.0",
                    "--superspreader_patient_min_add", "15",
                    "--superspreader_staff_contacts", "80",
                    "--superspreader_edge_weight_mult", "2.2",
                    "--admit_import_seasonality", "sinusoid",
                    "--admit_import_period_days", "360",
                    "--admit_import_amp", "0.35",
                    "--admit_import_phase_day", "45",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
        },
    },
}
# =============================================================================

DEFAULT_RESULTS_PARENT = Path("experiments_results")
DEFAULT_OVERLEAF_DIRNAME = "overleaf_package"

TRAINING_OUT_REL = Path("training_outputs")


def _machine_cpu_count() -> int:
    try:
        n = int(os.cpu_count() or 1)
    except Exception:
        n = 1
    return max(1, n)


def _half_machine_core_budget() -> int:
    env_budget = str(os.environ.get("DT_TRAIN_CPU_BUDGET", "")).strip()
    if env_budget != "":
        try:
            return max(1, int(env_budget))
        except Exception:
            pass
    return max(1, _machine_cpu_count() // 2)


def _resolve_train_runtime_budgets(neighbor_sampling: bool) -> Tuple[int, int, int]:
    if bool(neighbor_sampling):
        return 0, 1, 1

    thread_budget = _half_machine_core_budget()
    num_workers = thread_budget
    torch_num_threads = thread_budget
    torch_num_interop_threads = max(1, min(4, thread_budget // 4 if thread_budget >= 4 else 1))
    return num_workers, torch_num_threads, torch_num_interop_threads
  
  

def _task_family(task_name: str) -> str:
    t = str(task_name).strip().lower()
    if t.startswith("early_outbreak_warning_h"):
        return "early_outbreak_warning"
    if t.startswith("endogenous_importation_") or t.startswith("endogenous_transmission_"):
        return "mechanism_split"
    return "default"


def _build_task_aligned_step1_trajectories(task_name: str, current_step1_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    family = _task_family(task_name)
    num_days = int(current_step1_cfg.get("num_days", 90))

    if family == "early_outbreak_warning":
        return {
            "outbreak_low_train": {
                "seed_base": 4100,
                "p_admit_import_cs": 0.0,
                "p_admit_import_cr": 0.0,
                "p_admit_import_is": 0.0,
                "p_admit_import_ir": 0.0,
                "daily_discharge_frac": 0.0,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--admit_import_seasonality", "none",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
            "outbreak_high_train": {
                "seed_base": 5100,
                "p_admit_import_cs": 0.55,
                "p_admit_import_cr": 0.55,
                "p_admit_import_is": 0.55,
                "p_admit_import_ir": 0.55,
                "daily_discharge_frac": 0.45,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "1",
                    "--superspreader_end_day", str(num_days),
                    "--superspreader_patient_frac_mult", "8.0",
                    "--superspreader_patient_min_add", "40",
                    "--superspreader_staff_contacts", "180",
                    "--superspreader_edge_weight_mult", "4.0",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", str(num_days),
                    "--admit_import_high_start_day", "1",
                    "--admit_import_high_end_day", str(num_days),
                    "--admit_import_high_mult", "3.0",
                    "--admit_import_low_mult", "1.5",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
            "outbreak_low_test": {
                "seed_base": 6100,
                "p_admit_import_cs": 0.0,
                "p_admit_import_cr": 0.0,
                "p_admit_import_is": 0.0,
                "p_admit_import_ir": 0.0,
                "daily_discharge_frac": 0.0,
                "daily_discharge_min_per_ward": 0,
                "extra_sim_args": [
                    "--admit_import_seasonality", "none",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
            "outbreak_high_test": {
                "seed_base": 7100,
                "p_admit_import_cs": 0.45,
                "p_admit_import_cr": 0.45,
                "p_admit_import_is": 0.45,
                "p_admit_import_ir": 0.45,
                "daily_discharge_frac": 0.35,
                "daily_discharge_min_per_ward": 1,
                "extra_sim_args": [
                    "--superspreader_staff", "s0",
                    "--superspreader_state", "IR",
                    "--superspreader_start_day", "1",
                    "--superspreader_end_day", str(num_days),
                    "--superspreader_patient_frac_mult", "7.0",
                    "--superspreader_patient_min_add", "35",
                    "--superspreader_staff_contacts", "140",
                    "--superspreader_edge_weight_mult", "3.5",
                    "--admit_import_seasonality", "piecewise",
                    "--admit_import_period_days", str(num_days),
                    "--admit_import_high_start_day", "1",
                    "--admit_import_high_end_day", str(num_days),
                    "--admit_import_high_mult", "2.5",
                    "--admit_import_low_mult", "1.25",
                    "--admit_import_pmax_cs", "1.0",
                    "--admit_import_pmax_cr", "1.0",
                    "--admit_import_pmax_is", "1.0",
                    "--admit_import_pmax_ir", "1.0",
                    *CANONICAL_SURVEILLANCE_EXTRA_ARGS,
                ],
            },
        }

    return {
        "endog_high_train": {
            "seed_base": 4100,
            "p_admit_import_cs": 0.005,
            "p_admit_import_cr": 0.005,
            "p_admit_import_is": 0.005,
            "p_admit_import_ir": 0.005,
            "daily_discharge_frac": 0.02,
            "daily_discharge_min_per_ward": 0,
            "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
        },
        "import_high_train": {
            "seed_base": 5100,
            "p_admit_import_cs": 0.60,
            "p_admit_import_cr": 0.60,
            "p_admit_import_is": 0.60,
            "p_admit_import_ir": 0.60,
            "daily_discharge_frac": 0.25,
            "daily_discharge_min_per_ward": 1,
            "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
        },
        "endog_high_test": {
            "seed_base": 6100,
            "p_admit_import_cs": 0.005,
            "p_admit_import_cr": 0.005,
            "p_admit_import_is": 0.005,
            "p_admit_import_ir": 0.005,
            "daily_discharge_frac": 0.02,
            "daily_discharge_min_per_ward": 0,
            "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
        },
        "import_high_test": {
            "seed_base": 7100,
            "p_admit_import_cs": 0.60,
            "p_admit_import_cr": 0.60,
            "p_admit_import_is": 0.60,
            "p_admit_import_ir": 0.60,
            "daily_discharge_frac": 0.25,
            "daily_discharge_min_per_ward": 1,
            "extra_sim_args": list(CANONICAL_SURVEILLANCE_EXTRA_ARGS),
        },
    }

def _resolve_canonical_trajectory_sets(task_name: str, current_step1_cfg: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], List[str]]:
    trajectories = _build_task_aligned_step1_trajectories(task_name, current_step1_cfg)
    names = list(trajectories.keys())
    train_names = [name for name in names if name.endswith("_train")]
    test_names = [name for name in names if name.endswith("_test")]
    return trajectories, names, train_names, test_names


CONFIG["STEP1"]["trajectories"], CANONICAL_TRAJECTORY_NAMES, CANONICAL_TRAIN_TRAJECTORIES, CANONICAL_TEST_TRAJECTORIES = _resolve_canonical_trajectory_sets(
    str(CONFIG["MODEL"].get("task", "")).strip(),
    CONFIG["STEP1"],
)

BASELINE_TRAIN_REL = Path("synthetic_amr_graphs_train")
BASELINE_TEST_REL = Path("synthetic_amr_graphs_test_frozen")
LIVE_TEST_REL = Path("synthetic_amr_graphs_test")

STEP5_ABLATION_REL = Path("step5_ablation")
STEP8_TRAIN_REL = Path("synthetic_step8_shift_train")
STEP8_TEST_REL = Path("synthetic_step8_shift_test")


def _step_dataset_globs(task_name: str, stage: str) -> List[str]:
    family = _task_family(task_name)
    stage_key = str(stage).strip().lower()

    if family == "mechanism_split":
        mapping = {
            "delay": [
                "synthetic_endog_import_step6_delay*_pt_flat",
                "synthetic_endog_import_step6_delay*_pt",
            ],
            "frequency": [
                "synthetic_endog_import_step6_freq*_pt_flat",
                "synthetic_endog_import_step6_freq*_pt",
            ],
            "sweep": [
                "synthetic_endog_import_step7c_sweep_pt_flat",
                "synthetic_endog_import_step7c_sweep_pt",
            ],
        }
        return list(mapping.get(stage_key, []))

    if family == "early_outbreak_warning":
        mapping = {
            "delay": [
                "synthetic_outbreak_step6_delay*_pt_flat",
                "synthetic_outbreak_step6_delay*_pt",
                "synthetic_*outbreak*step6_delay*_pt_flat",
                "synthetic_*outbreak*step6_delay*_pt",
                "synthetic_*step6_delay*_pt_flat",
                "synthetic_*step6_delay*_pt",
            ],
            "frequency": [
                "synthetic_outbreak_step6_freq*_pt_flat",
                "synthetic_outbreak_step6_freq*_pt",
                "synthetic_*outbreak*step6_freq*_pt_flat",
                "synthetic_*outbreak*step6_freq*_pt",
                "synthetic_*step6_freq*_pt_flat",
                "synthetic_*step6_freq*_pt",
            ],
            "sweep": [
                "synthetic_outbreak_step7*_sweep*_pt_flat",
                "synthetic_outbreak_step7*_sweep*_pt",
                "synthetic_*outbreak*step7*_sweep*_pt_flat",
                "synthetic_*outbreak*step7*_sweep*_pt",
                "synthetic_*step7*_sweep*_pt_flat",
                "synthetic_*step7*_sweep*_pt",
            ],
        }
        return list(mapping.get(stage_key, []))

    fallback = {
        "delay": ["synthetic_*step6_delay*_pt_flat", "synthetic_*step6_delay*_pt"],
        "frequency": ["synthetic_*step6_freq*_pt_flat", "synthetic_*step6_freq*_pt"],
        "sweep": ["synthetic_*step7*_sweep*_pt_flat", "synthetic_*step7*_sweep*_pt"],
    }
    return list(fallback.get(stage_key, []))


DATASET_FIGURES_REL = Path("_dataset_graph_figures")


# =============================================================================
# Report writer
# =============================================================================

class Report:
    def __init__(self, path: Path, dry: bool):
        self.path = path
        self.dry = bool(dry)
        self._enabled = not self.dry
        self._console_enabled = os.environ.get("DT_DISABLE_CONSOLE_PROGRESS", "0") != "1"

        self._active_kind: Optional[str] = None
        self._console_last_render_len: int = 0

        self._step_title: Optional[str] = None
        self._step_current: int = 0
        self._step_total: int = 1
        self._step_extra: str = ""
        self._step_started_at: Optional[float] = None

        self._subtask_title: Optional[str] = None
        self._subtask_current: int = 0
        self._subtask_total: int = 1
        self._subtask_extra: str = ""
        self._subtask_started_at: Optional[float] = None

        if self._enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existed = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", encoding="utf-8") as f:
                if existed:
                    f.write("\n" + "#" * 80 + "\n")
                f.write(f"SESSION START UTC: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}\n")

    def _log(self, line: str = "") -> None:
        if not self._enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(line) + "\n")

    def write(self, line: str = "") -> None:
        self._log(str(line))

    def _progress_bar(self, current: int, total: int, width: int = 32) -> str:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        filled = int(round(width * current / total))
        return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"

    def _fmt_elapsed(self, started_at: Optional[float]) -> str:
        if started_at is None:
            return "00:00:00"
        elapsed = max(0, int(time.time() - started_at))
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _render_line(self, msg: str, kind: str) -> None:
      if not self._console_enabled:
          return

      is_tty = sys.stdout.isatty()
  
      if not is_tty:
          print(msg, flush=True)
          self._active_kind = None
          self._console_last_render_len = 0
          return
  
      if self._active_kind is not None and self._active_kind != kind:
          sys.stdout.write("\n")
          sys.stdout.flush()
          self._console_last_render_len = 0
  
      clear = "\r\033[2K"
      pad = ""
      if len(msg) < self._console_last_render_len:
          pad = " " * (self._console_last_render_len - len(msg))
  
      sys.stdout.write(clear + msg + pad)
      sys.stdout.flush()
      self._active_kind = kind
      self._console_last_render_len = len(msg)

    def _close_active_line(self) -> None:
        if not self._console_enabled:
            self._active_kind = None
            self._console_last_render_len = 0
            return
        if self._active_kind is not None:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._active_kind = None
            self._console_last_render_len = 0

    def _render_step_line(self) -> None:
        if self._step_title is None:
            return
        msg = (
            f"{self._step_title} "
            f"{self._progress_bar(self._step_current, self._step_total)} "
            f"| {self._fmt_elapsed(self._step_started_at)}"
        )
        if self._step_extra:
            msg += f" | {self._step_extra}"
        self._render_line(msg, kind="step")

    def _render_subtask_line(self) -> None:
        if self._subtask_title is None:
            return
        msg = (
            f"  {self._subtask_title} "
            f"{self._progress_bar(self._subtask_current, self._subtask_total)} "
            f"| {self._fmt_elapsed(self._subtask_started_at)}"
        )
        if self._subtask_extra:
            msg += f" | {self._subtask_extra}"
        self._render_line(msg, kind="subtask")

    def start_step_console(self, title: str, total: int = 1, extra: str = "") -> None:
        self._step_title = str(title).strip()
        self._step_current = 0
        self._step_total = max(1, int(total))
        self._step_extra = str(extra).strip()
        self._step_started_at = time.time()
        self._render_step_line()

    def update_step_console(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        extra: Optional[str] = None,
    ) -> None:
        if self._step_title is None:
            return
        if current is not None:
            self._step_current = max(0, int(current))
        if total is not None:
            self._step_total = max(1, int(total))
        if extra is not None:
            self._step_extra = str(extra).strip()
        self._render_step_line()

    def tick_step_console(self, extra: Optional[str] = None) -> None:
        if extra is not None:
            self._step_extra = str(extra).strip()
        self._render_step_line()

    def finish_step_console(self, force_complete: bool = True) -> None:
        if self._step_title is None:
            return
        if force_complete:
            self._step_current = self._step_total
            self._render_step_line()
        self._close_active_line()
        self._step_title = None
        self._step_current = 0
        self._step_total = 1
        self._step_extra = ""
        self._step_started_at = None

    def start_subtask_console(self, title: str, total: int = 1, extra: str = "") -> None:
        self._subtask_title = str(title).strip()
        self._subtask_current = 0
        self._subtask_total = max(1, int(total))
        self._subtask_extra = str(extra).strip()
        self._subtask_started_at = time.time()
        self._render_subtask_line()

    def update_subtask_console(
        self,
        current: Optional[int] = None,
        total: Optional[int] = None,
        extra: Optional[str] = None,
    ) -> None:
        if self._subtask_title is None:
            return
        if current is not None:
            self._subtask_current = max(0, int(current))
        if total is not None:
            self._subtask_total = max(1, int(total))
        if extra is not None:
            self._subtask_extra = str(extra).strip()
        self._render_subtask_line()

    def tick_subtask_console(self, extra: Optional[str] = None) -> None:
        if self._subtask_title is None:
            return
        if extra is not None:
            self._subtask_extra = str(extra).strip()
        self._render_subtask_line()

    def finish_subtask_console(self, force_complete: bool = True) -> None:
        if self._subtask_title is None:
            return
        if force_complete:
            self._subtask_current = self._subtask_total
            self._render_subtask_line()
        self._close_active_line()
        self._subtask_title = None
        self._subtask_current = 0
        self._subtask_total = 1
        self._subtask_extra = ""
        self._subtask_started_at = None
        if self._step_title is not None:
            self._render_step_line()

    def section(self, title: str, total: Optional[int] = None) -> None:
        title_s = str(title)
        self.write("")
        self.write("=" * 80)
        self.write(title_s)
        self.write("=" * 80)

        if title_s.startswith("STEP "):
            if self._step_title is not None:
                self.finish_step_console(force_complete=True)
            self.start_step_console(title_s, total=1 if total is None else int(total))

    def kv(self, k: str, v: Any) -> None:
        self.write(f"{k}: {v}")

# =============================================================================
# Progress helpers
# =============================================================================

def _progress_bar(current: int, total: int, width: int = 32) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


def _report_progress(
    report: Report,
    *,
    prefix: str,
    current: int,
    total: int,
    extra: str = "",
    target: str = "step",
) -> None:
    msg = f"{prefix} {_progress_bar(current, total)}"
    if extra:
        msg += f" | {extra}"
    report.write(msg)

    if target == "step":
        report.update_step_console(current=current, total=total, extra=f"{prefix} | {extra}" if extra else prefix)
    elif target == "subtask":
        report.update_subtask_console(current=current, total=total, extra=f"{prefix} | {extra}" if extra else prefix)

def _format_duration_hms(total_seconds: float) -> str:
    secs = max(0, int(round(float(total_seconds))))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _planned_parallel_step_keys(start_v: float, stop_v: float) -> List[str]:
    step_bounds: List[Tuple[str, float, float]] = [
        ("1", 1.0, 1.0),
        ("2", 2.0, 2.0),
        ("3", 3.0, 3.0),
        ("4", 4.0, 4.0),
        ("5", 5.0, 5.0),
        ("6", 6.1, 6.2),
        ("7", 7.0, 7.0),
        ("8", 8.0, 8.0),
        ("9", 9.0, 9.0),
    ]
    planned: List[str] = []
    for step_key, lo, hi in step_bounds:
        if start_v <= hi and stop_v >= lo:
            planned.append(step_key)
    return planned


def _parallel_progress_bar(current: int, total: int, width: int = 28) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _parallel_stage_bar(stage_counter: int, status: str, width: int = 10) -> str:
    status_s = str(status).strip().lower()
    if status_s in {"done", "failed"}:
        fill = width if status_s == "done" else max(1, width // 2)
        ch = "#" if status_s == "done" else "!"
        return "[" + ch * fill + "-" * (width - fill) + "]"

    phase = max(0, int(stage_counter)) % max(1, width)
    fill = min(width, phase + 1)
    return "[" + "=" * fill + "." * (width - fill) + "]"


def _write_track_progress_state(progress_path: Path, payload: Dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(progress_path)


def _read_track_progress_state(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


_PARALLEL_CONSOLE_LINE_COUNT = 0


def _render_parallel_tracks_console(
    states: List[Dict[str, Any]],
    *,
    initialized: bool,
) -> bool:
    global _PARALLEL_CONSOLE_LINE_COUNT

    def _state_line(state: Dict[str, Any]) -> str:
        track_name = str(state.get("track_name", "track")).strip() or "track"
        current = int(state.get("completed_steps", 0) or 0)
        total = int(state.get("total_steps", 1) or 1)
        status = str(state.get("status", "pending")).strip() or "pending"
        elapsed = _format_duration_hms(float(state.get("elapsed_sec", 0.0) or 0.0))
        active_step = str(state.get("active_step", "")).strip()
        stage_counter = int(state.get("stage_counter", 0) or 0)
        stage_bar = _parallel_stage_bar(stage_counter, status)
        extra = f" | {active_step}" if active_step else ""
        return (
            f"{track_name:<26} {_parallel_progress_bar(current, total)} {current}/{total} "
            f"{stage_bar} | stage={status} | {elapsed}{extra}"
        )

    if not sys.stdout.isatty() or str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        if not initialized:
            for state in states:
                print(_state_line(state), flush=True)
            return True
        return initialized

    line_count = max(1, len(states))
    rendered_lines = [_state_line(state) for state in states]
    target_line_count = max(1, max(int(_PARALLEL_CONSOLE_LINE_COUNT or 0), line_count))

    if not initialized:
        _PARALLEL_CONSOLE_LINE_COUNT = target_line_count
        sys.stdout.write("\033[?25l")
        for idx in range(target_line_count):
            line = rendered_lines[idx] if idx < len(rendered_lines) else ""
            sys.stdout.write("\r\033[2K" + line)
            if idx < target_line_count - 1:
                sys.stdout.write("\n")
        sys.stdout.flush()
        return True

    if target_line_count > 1:
        sys.stdout.write(f"\033[{target_line_count - 1}F")
    else:
        sys.stdout.write("\r")

    for idx in range(target_line_count):
        line = rendered_lines[idx] if idx < len(rendered_lines) else ""
        sys.stdout.write("\r\033[2K" + line)
        if idx < target_line_count - 1:
            sys.stdout.write("\n")

    sys.stdout.flush()
    _PARALLEL_CONSOLE_LINE_COUNT = target_line_count
    return True

def _to_work_dir_relative_path(path_value: Path | str, work_dir: Path) -> str:
    path_obj = Path(path_value)
    if not path_obj.is_absolute():
        return path_obj.as_posix()
    try:
        return path_obj.relative_to(work_dir).as_posix()
    except Exception:
        return path_obj.as_posix()

# =============================================================================
# Shell helpers
# =============================================================================
def _subtask_label_from_cmd(cmd: List[str]) -> str:
    if len(cmd) >= 2 and str(cmd[1]).endswith(".py"):
        return Path(str(cmd[1])).name
    return Path(str(cmd[0])).name



def _normalize_python_script_cmd_for_cwd(cmd: List[str], cwd: Optional[Path]) -> List[str]:
    normalized = [str(x) for x in cmd]
    if len(normalized) < 2 or not normalized[1].endswith(".py"):
        return normalized

    script_name = Path(normalized[1]).name

    if cwd is not None:
        staged_script = cwd / script_name
        if staged_script.is_file():
            normalized[1] = staged_script.name
            return normalized

    repo_script = Path(__file__).resolve().parent / script_name
    if repo_script.is_file():
        normalized[1] = str(repo_script)

    return normalized


def run_cmd(
    cmd: List[str],
    dry: bool,
    cwd: Optional[Path] = None,
    env: Optional[dict] = None,
    report: Optional[Report] = None,
    *,
    stream_output: bool = False,
) -> None:
    cwd_path = cwd.resolve() if cwd is not None else None
    cmd_norm = _normalize_python_script_cmd_for_cwd(cmd, cwd_path)

    msg = "RUN: " + " ".join(str(x) for x in cmd_norm)
    if cwd_path is not None:
        msg += f" | CWD: {cwd_path}"
    if report is not None:
        report.write(msg)
    else:
        print("\n" + msg)

    if dry:
        if report is not None:
            report.write("  [DRY_RUN] skipped")
        else:
            print("  [DRY_RUN] skipped")
        return

    merged_env = os.environ.copy()
    merged_env["PYTHONUTF8"] = "1"
    merged_env["PYTHONIOENCODING"] = "utf-8"
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    label = _subtask_label_from_cmd(cmd)
    if report is not None:
        report.start_subtask_console(f"run_cmd:{label}", total=1, extra="starting")

    stdout_text = ""
    stderr_text = ""

    try:
        if stream_output:
            proc = subprocess.Popen(
                cmd_norm,
                cwd=str(cwd_path) if cwd_path else None,
                env=merged_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            merged_lines: List[str] = []
            tick = 0
            assert proc.stdout is not None
            while True:
                line = proc.stdout.readline()
                if line:
                    merged_lines.append(line)
                    line_to_write = line.rstrip("\n")
                    if report is not None:
                        report.write(line_to_write)
                    else:
                        print(line_to_write, flush=True)
                    continue
                rc = proc.poll()
                if rc is not None:
                    break
                if report is not None:
                    report.tick_subtask_console(extra=f"running {'.' * ((tick % 3) + 1)}")
                time.sleep(0.2)
                tick += 1
            return_code = proc.wait()
            stdout_text = "".join(merged_lines)
            stderr_text = ""
        else:
            with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_f, tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8"
            ) as stderr_f:
                proc = subprocess.Popen(
                    cmd_norm,
                    cwd=str(cwd_path) if cwd_path else None,
                    env=merged_env,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    text=True,
                )

                tick = 0
                while proc.poll() is None:
                    if report is not None:
                        report.tick_subtask_console(extra=f"running {'.' * ((tick % 3) + 1)}")
                    time.sleep(0.5)
                    tick += 1

                return_code = proc.wait()

                stdout_f.seek(0)
                stderr_f.seek(0)
                stdout_text = stdout_f.read() or ""
                stderr_text = stderr_f.read() or ""
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None and not stream_output:
        if stdout_text.strip():
            report.write("--- stdout begin ---")
            for line in stdout_text.splitlines():
                report.write(line)
            report.write("--- stdout end ---")
        if stderr_text.strip():
            report.write("--- stderr begin ---")
            for line in stderr_text.splitlines():
                report.write(line)
            report.write("--- stderr end ---")

    if return_code != 0:
        stdout_tail = stdout_text.strip()[-4000:]
        stderr_tail = stderr_text.strip()[-4000:]
        err_lines = [
            f"Command failed with exit code {return_code}",
            f"CMD: {' '.join(str(x) for x in cmd_norm)}",
            f"CWD: {cwd_path if cwd_path is not None else Path.cwd()}",
        ]
        if stdout_tail:
            err_lines += ["--- stdout (tail) ---", stdout_tail]
        if stderr_tail:
            err_lines += ["--- stderr (tail) ---", stderr_tail]
        err_msg = "\n".join(err_lines)
        if report is not None:
            report.write(err_msg)
        raise subprocess.CalledProcessError(return_code, cmd_norm, output=stdout_text, stderr=stderr_text)



def _extract_unrecognized_cli_tokens(output_text: str) -> List[str]:
    text = str(output_text or "")
    matches = re.findall(r"unrecognized arguments?:\s*(.+)", text)
    if not matches:
        return []

    tokens: List[str] = []
    for match in matches:
        for token in shlex.split(match):
            tok = str(token).strip()
            if tok.startswith("--"):
                tokens.append(tok)

    seen = set()
    ordered: List[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered



def _strip_cli_flags_and_values(cmd: List[str], flags_to_remove: List[str]) -> List[str]:
    if not flags_to_remove:
        return list(cmd)

    remove_set = {str(flag) for flag in flags_to_remove}
    stripped: List[str] = []
    i = 0
    while i < len(cmd):
        token = str(cmd[i])
        if token in remove_set:
            i += 1
            if i < len(cmd) and not str(cmd[i]).startswith("--"):
                i += 1
            continue
        stripped.append(cmd[i])
        i += 1
    return stripped



def _run_cmd_with_unrecognized_arg_fallback(
    cmd: List[str],
    *,
    dry: bool,
    cwd: Optional[Path],
    report: Optional[Report],
    stream_output: bool = False,
    fallback_context: str = "",
) -> None:
    attempt_cmd = list(cmd)
    removed_flags: List[str] = []

    while True:
        try:
            run_cmd(attempt_cmd, dry=dry, cwd=cwd, report=report, stream_output=stream_output)
            if removed_flags and report is not None:
                report.write(
                    f"CLI_FALLBACK_SUCCESS context={fallback_context or 'n/a'} removed_flags={removed_flags}"
                )
            return
        except subprocess.CalledProcessError as exc:
            combined_output = "\n".join(
                [
                    str(getattr(exc, "output", "") or ""),
                    str(getattr(exc, "stderr", "") or ""),
                ]
            )
            unsupported_flags = _extract_unrecognized_cli_tokens(combined_output)
            unsupported_flags = [flag for flag in unsupported_flags if flag in attempt_cmd]
            if not unsupported_flags:
                raise

            next_cmd = _strip_cli_flags_and_values(attempt_cmd, unsupported_flags)
            if next_cmd == attempt_cmd:
                raise

            removed_flags.extend([flag for flag in unsupported_flags if flag not in removed_flags])
            if report is not None:
                report.write(
                    f"CLI_FALLBACK_RETRY context={fallback_context or 'n/a'} removing_unsupported={unsupported_flags}"
                )
            attempt_cmd = next_cmd

def require_file(p: Path) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p.resolve()}")


def ensure_dir(p: Path, dry: bool) -> None:
    if dry:
        return
    p.mkdir(parents=True, exist_ok=True)


def _safe_tag(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "tag"


def _copy_with_collision_guard(src: Path, dst_dir: Path, *, source_hint: Optional[Path] = None) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)

    desired_name = src.name
    dst = dst_dir / desired_name
    if not dst.exists():
        shutil.copy2(src, dst)
        return desired_name

    hint = source_hint.name if source_hint is not None else src.parent.name
    hint = _safe_tag(hint)
    alt_name = f"{hint}__{src.name}"
    alt_dst = dst_dir / alt_name
    if not alt_dst.exists():
        shutil.copy2(src, alt_dst)
        return alt_name

    raise RuntimeError(
        f"Filename collision while copying into {dst_dir}: '{src.name}' and '{alt_name}' already exist. "
        f"Source file was: {src}"
    )


def _copy_graphml_tree(
    *,
    src_root: Path,
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_keep",
) -> int:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would keep GraphML tree {src_root} -> {dst_root} ({label})")
        return 0

    if not src_root.exists():
        raise FileNotFoundError(f"Missing GraphML source root: {src_root.resolve()}")

    graphml_files = sorted([p for p in src_root.rglob("*.graphml") if p.is_file()])
    if not graphml_files:
        if report is not None:
            report.write(f"GRAPHML_KEEP skipped empty source={src_root} label={label}")
        return 0

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    if report is not None:
        report.start_subtask_console(f"graphml_keep:{label}", total=len(graphml_files), extra=str(src_root))

    copied = 0
    manifest_path = dst_root / "manifest.csv"
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kept_file", "relative_source_path"])
            for src_file in graphml_files:
                rel = src_file.relative_to(src_root)
                dst_file = dst_root / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied += 1
                w.writerow([dst_file.relative_to(dst_root).as_posix(), rel.as_posix()])
                if report is not None:
                    report.update_subtask_console(current=copied, total=len(graphml_files), extra=str(src_root))
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"GRAPHML_KEEP copied={copied} src={src_root} dst={dst_root} label={label}")
    return copied


def keep_graphml_roots(
    *,
    src_roots: List[Path],
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_keep",
) -> int:
    copied_total = 0
    seen = set()
    ordered_roots: List[Path] = []
    for root in src_roots:
        rp = root.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        ordered_roots.append(root)

    for src_root in ordered_roots:
        target = dst_root / _safe_tag(src_root.name)
        copied_total += _copy_graphml_tree(
            src_root=src_root,
            dst_root=target,
            dry=dry,
            report=report,
            label=f"{label}::{src_root.name}",
        )

    if report is not None:
        report.write(f"GRAPHML_KEEP_TOTAL copied={copied_total} dst={dst_root} label={label}")
    return copied_total



def _kept_graphml_split_dir(graphml_keep_root: Path, step_name: str, split_name: str) -> Path:
    step_tag = _safe_tag(step_name)
    split_tag = str(split_name).strip().lower()
    if split_tag not in {"train", "test"}:
        raise ValueError(f"split_name must be 'train' or 'test', got: {split_name}")
    return graphml_keep_root / step_tag / split_tag



def keep_step_graphml_split(
    *,
    graphml_keep_root: Path,
    step_name: str,
    train_src_roots: List[Path],
    test_src_roots: List[Path],
    dry: bool,
    report: Optional[Report] = None,
    pool_train: bool = False,
    pool_test: bool = False,
) -> None:
    train_dst = _kept_graphml_split_dir(graphml_keep_root, step_name, "train")
    test_dst = _kept_graphml_split_dir(graphml_keep_root, step_name, "test")

    if pool_train:
        _pool_into_existing_keep_root(
            src_roots=train_src_roots,
            dst_root=train_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::train",
        )
    else:
        keep_graphml_roots(
            src_roots=train_src_roots,
            dst_root=train_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::train",
        )

    if pool_test:
        _pool_into_existing_keep_root(
            src_roots=test_src_roots,
            dst_root=test_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::test",
        )
    else:
        keep_graphml_roots(
            src_roots=test_src_roots,
            dst_root=test_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::test",
        )


def _has_graphml_files(folder: Path) -> bool:
    return folder.exists() and any(folder.rglob("*.graphml"))


def _purge_graphml_under(
    root: Path,
    *,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "graphml_cleanup",
) -> int:

    if os.environ.get("DT_KEEP_GRAPHML", "0") == "1":
        if report is not None:
            report.write(f"GRAPHML_PURGE skipped root={root} label={label} because DT_KEEP_GRAPHML=1")
        return 0

    files = sorted([p for p in root.rglob("*.graphml") if p.is_file()])
    n = len(files)

    if dry:
        if report is not None and n > 0:
            report.write(f"[DRY_RUN] would delete {n} .graphml files under {root} ({label})")
        return n

    if n == 0:
        return 0

    if report is not None:
        report.start_subtask_console(f"_purge_graphml_under:{label}", total=n, extra=str(root))

    deleted = 0
    try:
        for p in files:
            try:
                p.unlink()
                deleted += 1
                if report is not None:
                    report.update_subtask_console(current=deleted, total=n, extra=str(root))
            except FileNotFoundError:
                continue
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None and deleted > 0:
        report.write(f"GRAPHML_PURGE deleted={deleted} root={root} label={label}")
    return deleted


def _has_direct_graphml_files(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.graphml"))


def _is_dataset_graph_root(folder: Path) -> bool:
    """
    A dataset graph root is a directory that represents a whole dataset and should
    be summarised once with graph_folder_figures.py before GraphML purge.

    Accepted shapes:
      - a folder containing sim_* subfolders with GraphML beneath it
      - a non-sim_* folder containing GraphML files directly
    """
    if not folder.exists() or not folder.is_dir():
        return False
    if not _has_graphml_files(folder):
        return False

    try:
        child_dirs = [p for p in folder.iterdir() if p.is_dir()]
    except FileNotFoundError:
        return False

    if any(p.name.startswith("sim_") for p in child_dirs):
        return True

    if _has_direct_graphml_files(folder) and not folder.name.startswith("sim_"):
        return True

    return False


def _find_dataset_graph_roots(search_root: Path) -> List[Path]:
    """
    Find dataset-level graph roots that should be summarised before purge.
    Keeps deepest valid roots to avoid duplicate summaries.
    """
    candidates: List[Path] = []

    if _is_dataset_graph_root(search_root):
        candidates.append(search_root)

    for p in search_root.rglob("*"):
        if p.is_dir() and _is_dataset_graph_root(p):
            candidates.append(p)

    uniq = sorted(set(candidates), key=lambda x: (-len(x.parts), str(x)))
    kept: List[Path] = []
    for p in uniq:
        if any(k in p.parents for k in kept):
            continue
        kept.append(p)

    return sorted(kept, key=lambda x: str(x))


def _rel_to_workdir_safe(path: Path, work_dir: Path) -> Path:
    try:
        return path.relative_to(work_dir)
    except Exception:
        return Path(path.name)


def archive_single_dataset_figures(
    *,
    py: str,
    graph_dir: Path,
    dst_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    title: str = "Graph dataset summary",
    label: str = "dataset",
    enable_graph_folder_figures: bool = False,
) -> None:
    if not enable_graph_folder_figures:
        msg = f"SKIP dataset_figures: graph_folder_figures disabled by CLI for {graph_dir}"
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    if not _has_graphml_files(graph_dir):
        msg = f"SKIP dataset_figures: no .graphml found under {graph_dir}"
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    cmd = [
        py,
        "graph_folder_figures.py",
        "--graph_dir", str(graph_dir),
        "--out_dir", str(dst_dir),
        "--identity", str(identity),
        "--title", str(title),
        "--label", str(label),
    ]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def archive_dataset_graph_figures_before_purge(
    *,
    py: str,
    work_dir: Path,
    search_root: Path,
    archive_root: Path,
    stage_tag: str,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> List[Path]:
    if not enable_graph_folder_figures:
        if report is not None:
            report.write(f"DATASET_FIGURES disabled by CLI for stage={stage_tag}")
        return []

    roots = _find_dataset_graph_roots(search_root)

    if not roots:
        if report is not None:
            report.write(f"DATASET_FIGURES none found under {search_root} for stage={stage_tag}")
        return []

    out_base = archive_root / DATASET_FIGURES_REL / _safe_tag(stage_tag)

    if report is not None:
        report.start_subtask_console(f"archive_dataset_graph_figures:{stage_tag}", total=len(roots), extra=str(search_root))

    try:
        for idx, ds_root in enumerate(roots, start=1):
            rel = _rel_to_workdir_safe(ds_root, work_dir)
            ds_tag = _safe_tag(rel.as_posix().replace("/", "__"))
            ds_title = f"{stage_tag}: {rel.as_posix()}"
            ds_out = out_base / ds_tag

            if report is not None:
                report.update_subtask_console(
                    current=idx,
                    total=len(roots),
                    extra=rel.as_posix(),
                )

            archive_single_dataset_figures(
                py=py,
                graph_dir=ds_root,
                dst_dir=ds_out,
                dry=dry,
                report=report,
                cwd=cwd,
                identity=identity,
                title=ds_title,
                label=ds_tag,
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
    finally:
        if report is not None:
            report.finish_subtask_console(force_complete=True)

    return roots


# =============================================================================
# Archiving
# =============================================================================

def archive_training_outputs(
    dst: Path,
    *,
    work_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    src_dir: Optional[Path] = None,
) -> None:
    ensure_dir(dst, dry)

    if src_dir is None:
        src = work_dir / TRAINING_OUT_REL
    else:
        src = src_dir if src_dir.is_absolute() else (work_dir / src_dir)

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would archive {src} -> {dst}")
        return

    if not src.exists():
        raise FileNotFoundError(f"Missing training outputs dir: {src.resolve()}")

    files: List[Path] = []
    for pattern in ["*.png", "*.csv", "*.pt", "*.json", "*.txt", "*.pdf", "*.jpg", "*.jpeg"]:
        files.extend(sorted(src.rglob(pattern)))

    total = len(files)
    copied_n = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"archive_training_outputs:{dst.name}", total=total, extra=str(src))

    try:
        for f in files:
            rel = f.relative_to(src)
            out_f = dst / rel
            out_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out_f)
            copied_n += 1
            if report is not None:
                report.update_subtask_console(current=copied_n, total=total, extra=str(src))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"ARCHIVE training_outputs ({copied_n} files) -> {dst}")


def _archive_dst_for_training_run(
    *,
    archive_root: Path,
    step_tag: str,
    t_val: int,
    h_val: Optional[int],
    run_all_T: bool,
    run_all_horizons: bool,
) -> Path:
    dst = archive_root / _safe_tag(step_tag)

    if run_all_T:
        dst = dst / f"T{int(t_val)}"

    if run_all_horizons and h_val is not None:
        dst = dst / f"h{int(h_val)}"

    return dst


def archive_dataset_folder(
    dst: Path,
    *,
    work_dir: Path,
    folder: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str = "dataset_folder",
) -> None:
    src_dir = folder if folder.is_absolute() else (work_dir / folder)

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would archive {label} {src_dir} -> {dst}")
        return

    if not src_dir.exists():
        raise FileNotFoundError(f"Missing {label}: {src_dir.resolve()}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in sorted(src_dir.rglob("*")) if p.is_file()]
    total = len(all_files)
    copied = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"archive_dataset_folder:{label}", total=total, extra=str(src_dir))

    manifest_path = dst / "manifest.csv"
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["archived_name", "relative_source_path"])

            for src_file in all_files:
                rel = src_file.relative_to(src_dir)

                if src_file.suffix.lower() in {".json", ".csv", ".txt", ".yaml", ".yml", ".graphml"}:
                    out_path = dst / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, out_path)
                    w.writerow([out_path.relative_to(dst).as_posix(), rel.as_posix()])
                    copied += 1
                    if report is not None:
                        report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
                    continue

                if src_file.suffix.lower() == ".pt":
                    short_name = src_file.name
                    out_path = dst / short_name

                    if len(str(out_path)) >= 235:
                        stem = src_file.stem
                        suffix = src_file.suffix
                        short_stem = stem[:80]
                        short_name = f"{copied:06d}__{short_stem}{suffix}"
                        out_path = dst / short_name

                    shutil.copy2(src_file, out_path)
                    w.writerow([short_name, rel.as_posix()])
                    copied += 1
                    if report is not None:
                        report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
                    continue

                out_path = dst / rel.name
                if out_path.exists():
                    out_path = dst / f"{copied:06d}__{rel.name}"
                shutil.copy2(src_file, out_path)
                w.writerow([out_path.name, rel.as_posix()])
                copied += 1
                if report is not None:
                    report.update_subtask_console(current=copied, total=total, extra=str(src_dir))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"ARCHIVE {label} ({copied} files) -> {dst}")

def archive_dataset_pair_figures(
    *,
    py: str,
    graph_dir: Path,
    compare_dir: Path,
    dst_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    title: str = "Train vs test graph dataset summary",
    label: str = "train",
    compare_label: str = "test",
    enable_graph_folder_figures: bool = False,
) -> None:
    if not enable_graph_folder_figures:
        msg = (
            f"SKIP dataset_figures: graph_folder_figures disabled by CLI, "
            f"graph_dir={graph_dir} compare_dir={compare_dir}"
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    graph_has_graphml = _has_graphml_files(graph_dir)
    compare_has_graphml = _has_graphml_files(compare_dir)

    if not graph_has_graphml or not compare_has_graphml:
        msg = (
            f"SKIP dataset_figures: graph_folder_figures.py requires .graphml files, "
            f"but got graph_dir={graph_dir} compare_dir={compare_dir}"
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)
        return

    cmd = [
        py,
        "graph_folder_figures.py",
        "--graph_dir", str(graph_dir),
        "--compare_dir", str(compare_dir),
        "--out_dir", str(dst_dir),
        "--identity", str(identity),
        "--title", str(title),
        "--label", str(label),
        "--compare_label", str(compare_label),
    ]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


# =============================================================================
# Overleaf export
# =============================================================================

def _slugify_label(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s if s else "fig"


def _latex_escape(s: str) -> str:
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _step_sort_key(step_tag: str) -> Tuple[float, str]:
    s = str(step_tag).strip().lower()
    m = re.match(r"step(\d+)(?:[._-]?(\d+))?", s)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2)) if m.group(2) is not None else 0
        return (major + minor / 10.0, s)
    return (9999.0, s)


def _pretty_step_name(step_tag: str) -> str:
    s = str(step_tag).strip().replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _infer_default_h_from_config() -> int:
    task_h = _infer_horizon_from_task_name(str(CONFIG["MODEL"].get("task", "")).strip())
    if task_h is not None:
        return int(task_h)
    return int(CONFIG["MODEL"].get("pred_horizon", 7))


def _extract_t_h_from_rel_parts(parts: Tuple[str, ...]) -> Tuple[int, int]:
    default_t = int(CONFIG["MODEL"]["T"])
    default_h = _infer_default_h_from_config()

    t_val = default_t
    h_val = default_h

    for part in parts:
        mt = re.fullmatch(r"T(\d+)", part)
        if mt:
            t_val = int(mt.group(1))
            continue

        mh = re.fullmatch(r"h(\d+)", part)
        if mh:
            h_val = int(mh.group(1))
            continue

    return t_val, h_val


def _discover_microgrid_pages(
    *,
    run_root: Path,
    figures_dir: Path,
    figure_rel_map: Optional[Dict[Path, str]] = None,
) -> Dict[Tuple[str, int, int], List[Dict[str, str]]]:
    del figures_dir

    figure_rel_map = figure_rel_map or {}
    pages: Dict[Tuple[str, int, int], List[Dict[str, str]]] = {}

    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_9"
        if not repro_root.exists():
            continue

        for step_dir in sorted(repro_root.iterdir()):
            if not step_dir.is_dir():
                continue

            for candidate_dir in [step_dir] + sorted([p for p in step_dir.rglob("*") if p.is_dir()]):
                cm_test = candidate_dir / "confusion_matrix_test.png"
                cm_train = candidate_dir / "confusion_matrix.png"
                roc_train = candidate_dir / "roc_curve.png"
                roc_test = candidate_dir / "roc_curve_test.png"

                cm = cm_test if cm_test.exists() else cm_train

                if not cm.exists():
                    continue
                if not roc_train.exists():
                    continue
                if not roc_test.exists():
                    continue

                rel_from_repro = candidate_dir.relative_to(repro_root)
                parts = rel_from_repro.parts
                if len(parts) == 0:
                    continue

                step_tag = parts[0]
                t_val, h_val = _extract_t_h_from_rel_parts(parts[1:])
                key = (track_dir.name, t_val, h_val)

                def _to_fig_path(p: Path) -> str:
                    p_resolved = p.resolve()
                    if p_resolved in figure_rel_map:
                        return figure_rel_map[p_resolved]
                    rel = p.relative_to(run_root)
                    return (Path("figures") / rel).as_posix()

                pages.setdefault(key, []).append(
                    {
                        "step_tag": step_tag,
                        "cm": _to_fig_path(cm),
                        "roc_train": _to_fig_path(roc_train),
                        "roc_test": _to_fig_path(roc_test),
                    }
                )

    for key in list(pages.keys()):
        pages[key] = sorted(
            pages[key],
            key=lambda row: _step_sort_key(row["step_tag"]),
        )

    return pages


def _latex_microgrid_for_page(
    *,
    track_name: str,
    t_val: int,
    h_val: int,
    rows: List[Dict[str, str]],
) -> List[str]:
    lines: List[str] = []

    caption = (
        f"Comparison grid for {_latex_escape(track_name)} at "
        f"$T={int(t_val)}$ and $H={int(h_val)}$. "
        f"Each row corresponds to one experiment configuration."
    )
    label = _slugify_label(f"{track_name}-T{t_val}-H{h_val}")

    lines.append(r"\clearpage")
    lines.append(r"\begin{figure}[p]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{3pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.1}")
    lines.append(r"  \small")
    lines.append(r"  \begin{tabular}{m{0.18\textwidth}m{0.26\textwidth}m{0.26\textwidth}m{0.26\textwidth}}")
    lines.append(r"    \textbf{Experiment} & \textbf{Confusion matrix} & \textbf{Train ROC} & \textbf{Test ROC} \\")
    lines.append(r"    \hline")

    for row in rows:
        step_name = _latex_escape(_pretty_step_name(row["step_tag"]))
        cm = row["cm"]
        roc_train = row["roc_train"]
        roc_test = row["roc_test"]

        lines.append(
            "    "
            + step_name
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{cm}}}"
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{roc_train}}}"
            + " & "
            + rf"\includegraphics[width=\linewidth,height=0.18\textheight,keepaspectratio]{{{roc_test}}}"
            + r" \\"
        )

    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{fig:{label}}}")
    lines.append(r"\end{figure}")
    lines.append("")

    return lines


def _mean_sd(values: Iterable[float]) -> Tuple[Optional[float], Optional[float], int]:
    vals: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv != fv:
            continue
        vals.append(fv)

    n = len(vals)
    if n == 0:
        return None, None, 0
    if n == 1:
        return vals[0], 0.0, 1

    mean_v = sum(vals) / n
    var_v = sum((x - mean_v) ** 2 for x in vals) / max(1, n - 1)
    return mean_v, var_v ** 0.5, n


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:
        return None
    return v


def _extract_binary_sens_spec_from_confusion_matrix(cm: Any) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(cm, list) or len(cm) != 2:
        return None, None
    try:
        row0 = list(cm[0])
        row1 = list(cm[1])
        if len(row0) != 2 or len(row1) != 2:
            return None, None
        tn = float(row0[0])
        fp = float(row0[1])
        fn = float(row1[0])
        tp = float(row1[1])
    except Exception:
        return None, None

    sensitivity = None if (tp + fn) <= 0 else tp / (tp + fn)
    specificity = None if (tn + fp) <= 0 else tn / (tn + fp)
    return sensitivity, specificity


def _metric_value_from_split(split: Dict[str, Any], metric_name: str) -> Optional[float]:
    if not isinstance(split, dict):
        return None

    metrics = split.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    if metric_name == "auroc":
        for key in ("roc_auc", "roc_auc_macro_ovr"):
            v = _safe_float(metrics.get(key))
            if v is not None:
                return v
        return None

    if metric_name == "accuracy":
        return _safe_float(metrics.get("accuracy"))

    if metric_name == "f1":
        for key in ("f1_macro", "f1_weighted"):
            v = _safe_float(metrics.get(key))
            if v is not None:
                return v
        return None

    if metric_name == "sensitivity":
        sens, _ = _extract_binary_sens_spec_from_confusion_matrix(split.get("confusion_matrix"))
        return sens

    if metric_name == "specificity":
        _, spec = _extract_binary_sens_spec_from_confusion_matrix(split.get("confusion_matrix"))
        return spec

    return None


def _format_mean_sd_latex(mean_v: Optional[float], sd_v: Optional[float], n: int) -> str:
    if mean_v is None:
        return '--'
    if n <= 1 or sd_v is None:
        return f'{mean_v:.3f}'
    return f'{mean_v:.3f} $\\pm$ {sd_v:.3f}'


def _discover_metrics_tables(run_root: Path) -> Dict[Tuple[str, int, int], List[Dict[str, Any]]]:
    tables: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    metric_names = ["auroc", "accuracy", "f1", "sensitivity", "specificity"]

    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_9"
        if not repro_root.exists():
            continue

        grouped: Dict[Tuple[str, int, int, str], List[Path]] = {}
        for summary_path in sorted(repro_root.rglob("metrics_summary.json")):
            rel_parent = summary_path.parent.relative_to(repro_root)
            parts = rel_parent.parts
            if len(parts) == 0:
                continue
            step_tag = parts[0]
            t_val, h_val = _extract_t_h_from_rel_parts(parts[1:])
            grouped.setdefault((track_dir.name, t_val, h_val, step_tag), []).append(summary_path)

        for (track_name, t_val, h_val, step_tag), summary_paths in grouped.items():
            train_values: Dict[str, List[float]] = {k: [] for k in metric_names}
            test_values: Dict[str, List[float]] = {k: [] for k in metric_names}

            for summary_path in summary_paths:
                try:
                    payload = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                train_split = payload.get("validation", {})
                test_split = payload.get("test", {})

                for metric_name in metric_names:
                    train_v = _metric_value_from_split(train_split, metric_name)
                    test_v = _metric_value_from_split(test_split, metric_name)
                    if train_v is not None:
                        train_values[metric_name].append(train_v)
                    if test_v is not None:
                        test_values[metric_name].append(test_v)

            row = {
                "step_tag": step_tag,
                "n_runs": len(summary_paths),
                "train": {},
                "test": {},
            }

            for metric_name in metric_names:
                train_mean, train_sd, train_n = _mean_sd(train_values[metric_name])
                test_mean, test_sd, test_n = _mean_sd(test_values[metric_name])
                row["train"][metric_name] = {"mean": train_mean, "sd": train_sd, "n": train_n}
                row["test"][metric_name] = {"mean": test_mean, "sd": test_sd, "n": test_n}

            tables.setdefault((track_name, t_val, h_val), []).append(row)

    for key in list(tables.keys()):
        tables[key] = sorted(tables[key], key=lambda row: _step_sort_key(str(row.get("step_tag", ""))))

    return tables


def _metrics_step_display_order(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _rank(step_tag: str) -> Tuple[float, int, str]:
        s = str(step_tag).strip().lower()

        if s == "step4_baseline":
            return (4.0, 0, s)
        if s == "step5_core_node_features_only":
            return (5.0, 0, s)
        if s == "step5_no_edge_weights":
            return (5.0, 1, s)

        m_delay = re.match(r"step6_delay_(.+)$", s)
        if m_delay:
            mm = re.search(r"(\d+)", m_delay.group(1))
            val = int(mm.group(1)) if mm else 999999
            return (6.1, val, s)

        m_freq = re.match(r"step6_freq_(.+)$", s)
        if m_freq:
            mm = re.search(r"(\d+)", m_freq.group(1))
            val = int(mm.group(1)) if mm else 999999
            return (6.2, val, s)

        if s == "step7_sweep":
            return (7.0, 0, s)
        if s == "step8_distribution_shift":
            return (8.0, 0, s)
        if s == "step9_baseline_to_shift_test":
            return (9.0, 0, s)

        base, key = _step_sort_key(s)
        return (base, 999999, key)

    return sorted(rows, key=lambda row: _rank(str(row.get("step_tag", ""))))


def _extract_step6_condition_value(step_tag: str, prefix: str) -> Optional[int]:
    s = str(step_tag).strip().lower()
    if not s.startswith(prefix):
        return None

    suffix = s[len(prefix):]
    matches = re.findall(r"(\d+)", suffix)
    if not matches:
        return None
    return int(matches[-1])


def _step_label_metrics(step_tag: str) -> str:
    s = str(step_tag).strip().lower()
    main = _step_label_main(s)
    if main is not None:
        return main

    if s.startswith("step6_delay_"):
        delay_val = _extract_step6_condition_value(s, "step6_delay_")
        if delay_val is not None:
            return rf"\shortstack[l]{{Step6.1 Delay {delay_val}}}"
        return r"\shortstack[l]{Step6.1 Delay}"

    if s.startswith("step6_freq_"):
        freq_val = _extract_step6_condition_value(s, "step6_freq_")
        if freq_val is not None:
            return rf"\shortstack[l]{{Step6.2 Freq {freq_val}}}"
        return r"\shortstack[l]{Step6.2 Freq}"

    return _latex_escape(_pretty_step_name(step_tag))


def _latex_metrics_table_for_page(
    *,
    track_name: str,
    t_val: int,
    h_val: int,
    rows: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    metric_pairs = [
        ("AUROC", "auroc"),
        ("Acc", "accuracy"),
        ("F1", "f1"),
        ("Sens", "sensitivity"),
        ("Spec", "specificity"),
    ]
    label = _slugify_label(f"{track_name}-metrics-T{t_val}-H{h_val}")
    caption = (
        f"Compact quantitative summary for {_latex_escape(track_name)} at "
        f"$T={int(t_val)}$ and $H={int(h_val)}$. "
        r"Entries report mean performance across archived runs; when more than one run is available, the table shows mean $\pm$ standard deviation."
    )

    ordered_rows = _metrics_step_display_order(rows)

    lines.append(r"\begin{table}[p]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3.0pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.12}")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth,center}")
    lines.append(r"    \begin{tabular}{m{0.22\textwidth}ccccc|ccccc|c}")
    lines.append(r"      \textbf{Experiment} & \multicolumn{5}{c|}{\textbf{Train}} & \multicolumn{5}{c|}{\textbf{Test}} & \textbf{$n$} \\")
    lines.append(r"      & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} & \textbf{Sens} & \textbf{Spec} & \textbf{AUROC} & \textbf{Acc} & \textbf{F1} & \textbf{Sens} & \textbf{Spec} & \\")
    lines.append(r"      \hline")

    for row in ordered_rows:
        step_tag = str(row.get("step_tag", "")).strip()
        label_tex = _step_label_metrics(step_tag)
        cells = [f"      {label_tex}"]
        for _, key in metric_pairs:
            metric = row.get("train", {}).get(key, {})
            cells.append(_format_mean_sd_latex(metric.get("mean"), metric.get("sd"), int(metric.get("n", 0))))
        for _, key in metric_pairs:
            metric = row.get("test", {}).get(key, {})
            cells.append(_format_mean_sd_latex(metric.get("mean"), metric.get("sd"), int(metric.get("n", 0))))
        cells.append(str(int(row.get("n_runs", 0))))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return lines



def _discover_tuning_best_configs(run_root: Path) -> Dict[Tuple[str, int, int], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_9"
        if not repro_root.exists():
            continue

        for step_tag in ["step4_baseline_tuning", "step8_distribution_shift_tuning"]:
            step_root = repro_root / step_tag
            if not step_root.exists() or not step_root.is_dir():
                continue

            best_paths = sorted(step_root.rglob("best_config.json"))
            for best_path in best_paths:
                try:
                    payload = json.loads(best_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                config = payload.get("config", {}) if isinstance(payload, dict) else {}
                if not isinstance(config, dict) or not config:
                    continue

                rel_parts = best_path.parent.relative_to(step_root).parts
                t_val, h_val = _extract_t_h_from_rel_parts(rel_parts)
                row = {
                    "step_tag": step_tag,
                    "selected_from_stage": payload.get("selected_from_stage"),
                    "selected_metric_name": payload.get("selected_metric_name"),
                    "selected_metric_value": payload.get("selected_metric_value"),
                    "best_epoch": payload.get("best_epoch"),
                    "epochs_completed": payload.get("epochs_completed"),
                    "stopped_early": payload.get("stopped_early"),
                    "config": config,
                }
                out.setdefault((track_dir.name, int(t_val), int(h_val)), []).append(row)

    for key in list(out.keys()):
        out[key] = sorted(out[key], key=lambda row: _step_sort_key(str(row.get("step_tag", ""))))
    return out


def _step_label_tuning(step_tag: str) -> str:
    s = str(step_tag).strip().lower()
    if s == "step4_baseline_tuning":
        return "Step4 tuning"
    if s == "step8_distribution_shift_tuning":
        return "Step8 tuning"
    return _latex_escape(_pretty_step_name(step_tag))


def _format_tuning_param_value(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, bool):
        return "true" if v else "false"
    return _latex_escape(v)


def _latex_tuning_table_for_page(
    *,
    track_name: str,
    t_val: int,
    h_val: int,
    rows: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    label = _slugify_label(f"{track_name}-tuning-T{t_val}-H{h_val}")
    caption = (
        f"Best lightweight hyperparameter settings selected for {_latex_escape(track_name)} at "
        f"$T={int(t_val)}$ and $H={int(h_val)}$. "
        r"Selections are reported only for steps where pre-training tuning was enabled."
    )
    ordered_rows = sorted(rows, key=lambda row: _step_sort_key(str(row.get("step_tag", ""))))
    preferred_param_order = [
        "hidden", "sage_layers", "transformer_layers", "heads", "dropout", "lr", "batch_size",
        "epochs", "sliding_step", "neighbor_sampling", "num_neighbors", "seed_count", "seed_strategy",
        "seed_batch_size", "max_sub_batches", "max_neighbors", "attn_top_k", "attn_rank_by",
        "fullgraph_attribution_pass", "emit_translational_figures", "translational_top_k", "use_cls",
        "use_task_hparams",
    ]

    lines.append(r"\begin{table}[p]")
    lines.append(r"  \centering")
    lines.append(r"  \scriptsize")
    lines.append(r"  \setlength{\tabcolsep}{3.5pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.10}")
    lines.append(r"  \begin{adjustbox}{max width=\textwidth,center}")
    lines.append(r"    \begin{tabular}{m{0.17\textwidth}m{0.33\textwidth}m{0.15\textwidth}m{0.12\textwidth}m{0.10\textwidth}}")
    lines.append(r"      \textbf{Tuning step} & \textbf{Best parameters} & \textbf{Validation metric} & \textbf{Best epoch} & \textbf{Stage} \\")
    lines.append(r"      \hline")

    for row in ordered_rows:
        config = row.get("config", {}) if isinstance(row.get("config", {}), dict) else {}
        param_items = []
        for key in preferred_param_order:
            if key in config:
                param_items.append(rf"\texttt{{{_latex_escape(key)}}}={_format_tuning_param_value(config[key])}")
        for key in sorted(config.keys()):
            if key not in preferred_param_order:
                param_items.append(rf"\texttt{{{_latex_escape(key)}}}={_format_tuning_param_value(config[key])}")
        params_tex = r"; ".join(param_items) if param_items else "--"

        metric_name = row.get("selected_metric_name")
        metric_value = _safe_float(row.get("selected_metric_value"))
        if metric_name and metric_value is not None:
            metric_tex = rf"\texttt{{{_latex_escape(metric_name)}}}={metric_value:.4f}"
        elif metric_name:
            metric_tex = rf"\texttt{{{_latex_escape(metric_name)}}}"
        else:
            metric_tex = "--"

        best_epoch = row.get("best_epoch")
        best_epoch_tex = "--" if best_epoch is None else str(best_epoch)
        stage = row.get("selected_from_stage")
        stage_tex = "--" if stage is None else _latex_escape(stage)
        step_label = _step_label_tuning(str(row.get("step_tag", "")))
        lines.append(
            "      "
            + step_label
            + " & "
            + params_tex
            + " & "
            + metric_tex
            + " & "
            + best_epoch_tex
            + " & "
            + stage_tex
            + r" \\")

    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{tab:{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return lines

def _dataset_png_order_key(name: str) -> Tuple[int, str]:
    order = {
        "figure_microgrid": 1,
        "figure_distributions": 2,
        "figure_communities_and_centrality": 3,
        "figure_flow_sankey": 4,
        "figure_timeline_nodes_edges": 5,
        "figure_state_percentages": 6,
        "figure_train_vs_test_shift": 7,
        "figure_train_vs_test_ecdf": 8,
        "figure_timeline_diff_test_minus_train": 9,
    }
    for prefix, rank in order.items():
        if name.startswith(prefix):
            return (rank, name)
    return (999, name)


def _pretty_dataset_page_name(rel_parts: Tuple[str, ...]) -> str:
    s = " / ".join(rel_parts)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.title()


def _discover_dataset_figure_sets(
    run_root: Path,
    figure_rel_map: Optional[Dict[Path, str]] = None,
) -> List[Dict[str, Any]]:
    figure_rel_map = figure_rel_map or {}
    out: List[Dict[str, Any]] = []

    for track_dir in sorted(run_root.glob("TRACK_*")):
        dataset_root = track_dir / "work" / "repro_artifacts_steps_1_9" / DATASET_FIGURES_REL
        if not dataset_root.exists():
            continue

        for ds_dir in sorted([p for p in dataset_root.rglob("*") if p.is_dir()]):
            pngs = sorted(
                [p for p in ds_dir.glob("*.png") if p.is_file()],
                key=lambda p: _dataset_png_order_key(p.name),
            )
            if not pngs:
                continue

            rel = ds_dir.relative_to(dataset_root)
            rel_pngs = [
                figure_rel_map.get(p.resolve(), (Path("figures") / p.relative_to(run_root)).as_posix())
                for p in pngs
            ]

            out.append(
                {
                    "track_name": track_dir.name,
                    "dataset_rel_parts": rel.parts,
                    "dataset_title": _pretty_dataset_page_name(rel.parts),
                    "pngs": rel_pngs,
                }
            )

    return out


def _latex_dataset_figures_for_pages(
    *,
    track_name: str,
    dataset_title: str,
    dataset_rel_parts: Tuple[str, ...],
    pngs: List[str],
    chunk_size: int = 4,
) -> List[str]:
    lines: List[str] = []
    page_chunks = [pngs[i:i + chunk_size] for i in range(0, len(pngs), chunk_size)]

    for page_idx, chunk in enumerate(page_chunks, start=1):
        label = _slugify_label(
            f"{track_name}-{'-'.join(dataset_rel_parts)}-page-{page_idx}"
        )
        caption = (
            f"Dataset-level raw-graph summaries for {_latex_escape(track_name)}: "
            f"{_latex_escape(dataset_title)} (page {page_idx} of {len(page_chunks)})."
        )

        lines.append(r"\clearpage")
        lines.append(r"\begin{figure}[p]")
        lines.append(r"  \centering")

        for idx, fig_path in enumerate(chunk, start=1):
            lines.append(r"  \begin{minipage}[t]{0.48\textwidth}")
            lines.append(r"    \centering")
            lines.append(
                rf"    \includegraphics[width=\linewidth,height=0.34\textheight,keepaspectratio]{{{fig_path}}}"
            )
            lines.append(r"  \end{minipage}")
            if idx % 2 == 1 and idx != len(chunk):
                lines.append(r"  \hfill")
            else:
                lines.append(r"  \vspace{0.7em}")

        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{fig:{label}}}")
        lines.append(r"\end{figure}")
        lines.append("")

    return lines


def _short_overleaf_rel_path(rel: Path) -> Path:
    rel = Path(rel)
    suffix = rel.suffix.lower()
    stem = _safe_tag(rel.stem)[:80] or "figure"
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:12]

    top_parts = list(rel.parts[:3])
    top_slug = _safe_tag("__".join(top_parts))[:60] or "misc"
    filename = f"{top_slug}__{stem}__{digest}{suffix}"
    return Path(top_slug) / filename




def _step_label_main(step_tag: str) -> Optional[str]:
    s = str(step_tag).strip().lower()
    if s == "step4_baseline":
        return "Step4 Baseline"
    if s == "step5_core_node_features_only":
        return r"\shortstack[l]{Step5 Core Node\\Features Only}"
    if s == "step5_no_edge_weights":
        return r"\shortstack[l]{Step5 No Edge\\Weights}"
    if s == "step7_sweep":
        return "Step7 Sweep"
    if s == "step8_distribution_shift":
        return "Step8 Distribution Shift"
    if s == "step9_baseline_to_shift_test":
        return "Step9 Baseline→Shift Test"
    return None


def _step_label_cross_track(step_tag: str, track_name: str) -> Optional[str]:
    s = str(step_tag).strip().lower()
    track_suffix = "Ground truth" if str(track_name).strip() == "TRACK_ground_truth" else "Partial observation"

    delay_val = _extract_step6_condition_value(s, "step6_delay_")
    if delay_val is not None:
        return rf"\shortstack[l]{{Step6 Delay {delay_val}\\{track_suffix}}}"

    freq_val = _extract_step6_condition_value(s, "step6_freq_")
    if freq_val is not None:
        return rf"\shortstack[l]{{Step6 Freq {freq_val}\\{track_suffix}}}"

    return None


def _latex_row_from_page_entry(row: Dict[str, str], label_tex: str, *, image_height: str) -> str:
    cm = row["cm"]
    roc_train = row["roc_train"]
    roc_test = row["roc_test"]
    return (
        f"      {label_tex} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{cm}}} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{roc_train}}} &\n"
        f"      \\includegraphics[width=\\linewidth,height={image_height},keepaspectratio]{{{roc_test}}} \\\\"
    )


def _latex_grid_block(
    *,
    caption: str,
    label: str,
    rows_tex: List[str],
    arraystretch: str,
) -> List[str]:
    lines: List[str] = []
    lines.append(r"\begin{figure}[!]")
    lines.append(r"  \centering")
    lines.append(r"  \setlength{\tabcolsep}{4pt}")
    lines.append(rf"  \renewcommand{{\arraystretch}}{{{arraystretch}}}")
    lines.append(r"  \small")
    lines.append(r"  \begin{adjustbox}{max totalsize={\textwidth}{0.92\textheight},center}")
    lines.append(r"    \begin{tabular}{m{0.18\textwidth}m{0.26\textwidth}m{0.26\textwidth}m{0.26\textwidth}}")
    lines.append(r"      \textbf{Experiment} & \textbf{Confusion matrix} & \textbf{Train ROC} & \textbf{Test ROC} \\")
    lines.append(r"      \hline")
    lines.extend(rows_tex)
    lines.append(r"    \end{tabular}")
    lines.append(r"  \end{adjustbox}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{figure}")
    lines.append("")
    return lines


def _build_main_track_grid_pages(pages: Dict[Tuple[str, int, int], List[Dict[str, str]]]) -> List[str]:
    lines: List[str] = []
    desired_order = [
        "step4_baseline",
        "step5_core_node_features_only",
        "step5_no_edge_weights",
        "step7_sweep",
        "step8_distribution_shift",
        "step9_baseline_to_shift_test",
    ]
    for track_name, t_val, h_val in sorted(pages.keys(), key=lambda x: (x[0], int(x[1]), int(x[2]))):
        page_rows = pages[(track_name, t_val, h_val)]
        row_map = {str(r.get("step_tag", "")).strip().lower(): r for r in page_rows}
        rows_tex: List[str] = []
        for step_tag in desired_order:
            row = row_map.get(step_tag)
            if row is None:
                continue
            label_tex = _step_label_main(step_tag)
            if label_tex is None:
                continue
            rows_tex.append(_latex_row_from_page_entry(row, label_tex, image_height="0.15\\textheight"))
            rows_tex.append("")

        if not rows_tex:
            continue

        track_label_slug = _slugify_label(track_name.replace("TRACK_", "track-"))
        caption = (
            f"Main comparison grid for {_latex_escape(track_name)} at $T={int(t_val)}$ and $H={int(h_val)}$. "
            r"Step~6 delay/frequency ablations are shown separately in Figures\ref{fig:track-step6-delay-cross-track}, "
            r"\ref{fig:track-step6-frequency-cross-track}."
        )
        label = f"fig:{track_label_slug}-main-t{int(t_val)}-h{int(h_val)}"
        lines.extend(
            _latex_grid_block(
                caption=caption,
                label=label,
                rows_tex=rows_tex,
                arraystretch="1.08",
            )
        )
    return lines


def _build_cross_track_step6_grid_pages(
    pages: Dict[Tuple[str, int, int], List[Dict[str, str]]],
    *,
    mode: str,
) -> List[str]:
    lines: List[str] = []
    if mode not in {"delay", "frequency"}:
        raise ValueError(f"Unsupported mode: {mode}")

    prefix = "step6_delay_" if mode == "delay" else "step6_freq_"
    figure_label = "fig:track-step6-delay-cross-track" if mode == "delay" else "fig:track-step6-frequency-cross-track"
    figure_title = "delay" if mode == "delay" else "screening-frequency"
    figure_title_caption = "delay" if mode == "delay" else "screening frequency"

    combos = sorted(pages.keys(), key=lambda x: (int(x[1]), int(x[2]), x[0]))
    grouped_by_th: Dict[Tuple[int, int], List[Tuple[str, List[Dict[str, str]]]]] = {}
    for track_name, t_val, h_val in combos:
        grouped_by_th.setdefault((int(t_val), int(h_val)), []).append((track_name, pages[(track_name, t_val, h_val)]))

    for (t_val, h_val), per_track_rows in grouped_by_th.items():
        rows_candidates: List[Tuple[int, str, Dict[str, str], str]] = []
        for track_name, page_rows in per_track_rows:
            for row in page_rows:
                step_tag = str(row.get("step_tag", "")).strip().lower()
                if not step_tag.startswith(prefix):
                    continue
                label_tex = _step_label_cross_track(step_tag, track_name)
                if label_tex is None:
                    continue
                m = re.search(r"(\d+)", step_tag)
                rank_num = int(m.group(1)) if m else 999999
                track_rank = 0 if track_name == "TRACK_ground_truth" else 1
                rows_candidates.append((rank_num, f"{rank_num:06d}_{track_rank}_{track_name}_{step_tag}", row, label_tex))

        rows_candidates.sort(key=lambda x: (x[0], x[1]))
        rows_tex = []
        for _, _, row, label_tex in rows_candidates:
            rows_tex.append(_latex_row_from_page_entry(row, label_tex, image_height="0.105\\textheight"))
            rows_tex.append("")

        if not rows_tex:
            continue

        caption = (
            f"Cross-track comparison of Step~6 {figure_title_caption} ablations at $T={int(t_val)}$ and $H={int(h_val)}$. "
            "Rows are grouped by intervention level and state-observation regime."
        )
        label = figure_label if (int(t_val), int(h_val)) == sorted(grouped_by_th.keys())[0] else f"{figure_label}-t{int(t_val)}-h{int(h_val)}"
        lines.extend(
            _latex_grid_block(
                caption=caption,
                label=label,
                rows_tex=rows_tex,
                arraystretch="1.06",
            )
        )
    return lines


def _discover_translational_figure_sets(
    run_root: Path,
    figure_rel_map: Optional[Dict[Path, str]] = None,
) -> List[Dict[str, Any]]:
    figure_rel_map = figure_rel_map or {}
    out: List[Dict[str, Any]] = []
    for track_dir in sorted(run_root.glob("TRACK_*")):
        repro_root = track_dir / "work" / "repro_artifacts_steps_1_9"
        if not repro_root.exists():
            continue
        for step_dir in sorted([p for p in repro_root.iterdir() if p.is_dir()]):
            for trans_dir in sorted(step_dir.rglob("translational_figures")):
                #pngs = sorted([p for p in trans_dir.glob("*.png") if p.is_file()])
                pngs = sorted(
                    [
                        p
                        for p in trans_dir.glob("*.png")
                        if p.is_file() and "translational_microgrid_" in p.name
                    ]
                )
                if not pngs:
                    continue
                rel_pngs = [figure_rel_map.get(p.resolve(), (Path("figures") / p.relative_to(run_root)).as_posix()) for p in pngs]
                out.append({
                    "track_name": track_dir.name,
                    "step_tag": step_dir.name,
                    "trans_dir_rel": trans_dir.relative_to(repro_root).parts,
                    "pngs": rel_pngs,
                })
    return out


def _latex_translational_figures_for_pages(*, track_name: str, step_tag: str, pngs: List[str], chunk_size: int = 4) -> List[str]:
    lines: List[str] = []
    chunks = [pngs[i:i + chunk_size] for i in range(0, len(pngs), chunk_size)]
    for page_idx, chunk in enumerate(chunks, start=1):
        label = _slugify_label(f"{track_name}-{step_tag}-translational-{page_idx}")
        caption = (
            f"Translational attribution figures for {_latex_escape(track_name)} / "
            f"{_latex_escape(_pretty_step_name(step_tag))} (page {page_idx} of {len(chunks)})."
        )
        lines.append(r"\clearpage")
        lines.append(r"\begin{figure}[p]")
        lines.append(r"  \centering")
        for idx, fig_path in enumerate(chunk, start=1):
            lines.append(r"  \begin{minipage}[t]{0.48\textwidth}")
            lines.append(r"    \centering")
            lines.append(
                rf"    \includegraphics[width=\linewidth,height=0.34\textheight,keepaspectratio]{{{fig_path}}}"
            )
            lines.append(r"  \end{minipage}")
            if idx % 2 == 1 and idx != len(chunk):
                lines.append(r"  \hfill")
            else:
                lines.append(r"  \vspace{0.7em}")
        lines.append(f"  \caption{{{caption}}}")
        lines.append(f"  \label{{fig:{label}}}")
        lines.append(r"\end{figure}")
        lines.append("")
    return lines


def export_overleaf_package(
    *,
    run_root: Path,
    archive_root: Path,
    out_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> Path:
    out_dir = out_dir if out_dir.is_absolute() else (run_root / out_dir)
    figures_dir = out_dir / "figures"
    latex_path = out_dir / "latex.txt"

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would export Overleaf package to: {out_dir}")
        return latex_path

    if out_dir.exists():
        shutil.rmtree(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".pdf", ".jpg", ".jpeg"}
    copied: List[Path] = []
    figure_rel_map: Dict[Path, str] = {}
    for p in sorted(archive_root.rglob("*")):
        if out_dir in p.parents or p == out_dir:
            continue
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(archive_root)
        short_rel = _short_overleaf_rel_path(rel)
        dst = figures_dir / short_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied.append(dst)
        figure_rel_map[p.resolve()] = (Path("figures") / short_rel).as_posix()

    lines: List[str] = []
    lines.append("% Auto-generated by experiments_pb.py")
    lines.append("% Assumes figures are located under: figures/")
    lines.append("% Suggested preamble additions:")
    lines.append("%   \\usepackage{graphicx}")
    lines.append("%   \\usepackage{float}")
    lines.append("%   \\usepackage{array}")
    lines.append("%   \\usepackage{caption}")
    lines.append("%   \\usepackage{adjustbox}")
    lines.append("")

    pages = _discover_microgrid_pages(
        run_root=run_root,
        figures_dir=figures_dir,
        figure_rel_map=figure_rel_map,
    )
    metrics_tables = _discover_metrics_tables(run_root)
    tuning_tables = _discover_tuning_best_configs(run_root)
    dataset_sets = _discover_dataset_figure_sets(
        run_root=run_root,
        figure_rel_map=figure_rel_map,
    )
    translational_sets = _discover_translational_figure_sets(
        run_root=run_root,
        figure_rel_map=figure_rel_map,
    )

    main_lines = _build_main_track_grid_pages(pages)
    delay_lines = _build_cross_track_step6_grid_pages(pages, mode="delay")
    freq_lines = _build_cross_track_step6_grid_pages(pages, mode="frequency")

    if main_lines:
        lines.extend(main_lines)
        for (track_name, t_val, h_val) in sorted(pages.keys(), key=lambda x: (x[0], int(x[1]), int(x[2]))):
            metric_rows = metrics_tables.get((track_name, t_val, h_val), [])
            if metric_rows:
                lines.extend(
                    _latex_metrics_table_for_page(
                        track_name=track_name,
                        t_val=int(t_val),
                        h_val=int(h_val),
                        rows=metric_rows,
                    )
                )
            else:
                lines.append(
                    f"% No metrics-summary table could be assembled for {track_name} T={int(t_val)} H={int(h_val)}."
                )
            tuning_rows = tuning_tables.get((track_name, int(t_val), int(h_val)), [])
            if tuning_rows:
                lines.extend(
                    _latex_tuning_table_for_page(
                        track_name=track_name,
                        t_val=int(t_val),
                        h_val=int(h_val),
                        rows=tuning_rows,
                    )
                )
            else:
                lines.append(
                    f"% No tuning-summary table could be assembled for {track_name} T={int(t_val)} H={int(h_val)}."
                )
    else:
        lines.append("% No main track comparison grids could be assembled from the archived outputs.")

    if delay_lines:
        lines.extend(delay_lines)
    else:
        lines.append("% No Step 6 delay comparison grid could be assembled from the archived outputs.")

    if freq_lines:
        lines.extend(freq_lines)
    else:
        lines.append("% No Step 6 frequency comparison grid could be assembled from the archived outputs.")

    if not dataset_sets:
        lines.append("% No dataset-figure pages could be assembled from archived raw-graph summaries.")
    else:
        for item in dataset_sets:
            lines.extend(
                _latex_dataset_figures_for_pages(
                    track_name=str(item["track_name"]),
                    dataset_title=str(item["dataset_title"]),
                    dataset_rel_parts=tuple(item["dataset_rel_parts"]),
                    pngs=list(item["pngs"]),
                )
            )

    if not translational_sets:
        lines.append("% No translational attribution figures could be assembled from archived outputs.")
    else:
        for item in translational_sets:
            lines.extend(
                _latex_translational_figures_for_pages(
                    track_name=str(item["track_name"]),
                    step_tag=str(item["step_tag"]),
                    pngs=list(item["pngs"]),
                )
            )

    latex_path.write_text("\n".join(lines), encoding="utf-8")
    if report is not None:
        report.write(f"OVERLEAF package exported -> {out_dir}")
        report.write(f"OVERLEAF latex.txt -> {latex_path}")
        report.write(f"OVERLEAF copied figures: {len(copied)}")
        report.write(f"OVERLEAF main grids: {1 if main_lines else 0}")
        report.write(f"OVERLEAF metrics tables: {len(metrics_tables)}")
        report.write(f"OVERLEAF tuning tables: {len(tuning_tables)}")
        report.write(f"OVERLEAF step6 delay grids: {1 if delay_lines else 0}")
        report.write(f"OVERLEAF step6 frequency grids: {1 if freq_lines else 0}")
        report.write(f"OVERLEAF dataset-figure sets: {len(dataset_sets)}")
        report.write(f"OVERLEAF translational-figure sets: {len(translational_sets)}")
    return latex_path


# =============================================================================
# Task / CLI arg building
# =============================================================================

def step_key_to_float(k: str) -> float:
    if k.endswith("b") and k[:-1].isdigit():
        return float(k[:-1]) + 0.1
    return float(k)


def _parse_int_list(s: str, *, what: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        if not re.fullmatch(r"\d+", p):
            raise ValueError(f"Invalid {what} token '{p}'. Expected comma-separated integers.")
        v = int(p)
        if v <= 0:
            raise ValueError(f"Invalid {what} '{v}'. Must be positive.")
        out.append(v)
    return sorted(set(out))


def _parse_horizons_list(s: str) -> List[int]:
    return _parse_int_list(s, what="horizon")


def _parse_T_list(s: str) -> List[int]:
    return _parse_int_list(s, what="T")


def _task_with_horizon(task: str, H: int) -> str:
    t = str(task).strip()
    if re.search(r"_h\d+$", t):
        return re.sub(r"_h\d+$", f"_h{int(H)}", t)
    return f"{t}_h{int(H)}"


def _infer_horizon_from_task_name(task: str) -> Optional[int]:
    m = re.search(r"_h(\d+)$", str(task).strip())
    if not m:
        return None
    try:
        h = int(m.group(1))
    except Exception:
        return None
    return h if h > 0 else None


def _get_max_T_needed_runtime(T_list_runtime: List[int]) -> int:
    vals = [int(CONFIG.get("MODEL", {}).get("T", 7))]
    vals.extend(int(t) for t in T_list_runtime if int(t) > 0)
    return max(vals)


def _dir_has_pt_files(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and any(folder.rglob("*.pt"))


def _archived_dataset_dirs_for_step(archive_root: Path, step_tag: str, split_name: str) -> List[Path]:
    step_root = archive_root / _safe_tag(step_tag)
    if not step_root.exists() or not step_root.is_dir():
        return []

    candidates: List[Path] = []
    direct = step_root / split_name
    if _dir_has_pt_files(direct):
        candidates.append(direct)

    for p in sorted(step_root.rglob(split_name)):
        if p.is_dir() and _dir_has_pt_files(p):
            candidates.append(p)

    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        uniq.append(c)
    return uniq


def _resolve_existing_dataset_dir(
    *,
    requested_folder: Path,
    work_dir: Path,
    archive_root: Path,
    step_tag: str,
    split_name: str,
    report: Optional[Report] = None,
    preferred_archive_dir: Optional[Path] = None,
    fallback_step_tags: Optional[List[str]] = None,
) -> Path:
    req_abs = requested_folder if requested_folder.is_absolute() else (work_dir / requested_folder)
    if _dir_has_pt_files(req_abs):
        return req_abs

    if preferred_archive_dir is not None:
        pref = preferred_archive_dir / split_name
        if _dir_has_pt_files(pref):
            if report is not None:
                report.write(f"DATASET_RESOLVE split={split_name} step={step_tag} source=preferred_archive path={pref}")
            return pref

    search_tags: List[str] = [str(step_tag)]
    for st in list(fallback_step_tags or []):
        if st not in search_tags:
            search_tags.append(str(st))

    for st in search_tags:
        cands = _archived_dataset_dirs_for_step(archive_root, st, split_name)
        if cands:
            chosen = cands[0]
            if report is not None:
                report.write(
                    f"DATASET_RESOLVE split={split_name} step={step_tag} archive_step={st} source=archive path={chosen}"
                )
            return chosen

    return req_abs


def _discover_archived_step_tags(archive_root: Path, prefix: str) -> List[str]:
    tags: List[str] = []
    if not archive_root.exists():
        return tags
    for p in sorted(archive_root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            tags.append(p.name)
    return tags




def _list_live_condition_train_dirs(root_dir: Path, tag_prefix: str) -> List[Tuple[str, Path]]:
    if not root_dir.exists() or not root_dir.is_dir():
        return []
    out: List[Tuple[str, Path]] = []
    for d in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        if _dir_has_pt_files(d):
            out.append((f"{tag_prefix}{_safe_tag(d.name)}", d))
    return out


def _discover_condition_train_dirs(
    *,
    work_dir: Path,
    archive_root: Path,
    live_root: Optional[Path],
    archive_tag_prefix: str,
    tag_prefix: str,
    report: Optional[Report] = None,
) -> List[Tuple[str, Path]]:
    """
    Discover per-condition train folders either from the live generated grid root
    or from archived train_folder copies for the corresponding step tags.
    Returns a list of (step_tag, absolute_train_folder_path).
    """
    discovered: List[Tuple[str, Path]] = []

    if live_root is not None:
        live_abs = live_root if live_root.is_absolute() else (work_dir / live_root)
        live_rows = _list_live_condition_train_dirs(live_abs, tag_prefix)
        if live_rows:
            if report is not None:
                report.write(
                    f"CONDITION_DISCOVERY source=live root={live_abs} tags={[tag for tag, _ in live_rows]}"
                )
            return [(tag, p.resolve()) for tag, p in live_rows]

    archived_tags = _discover_archived_step_tags(archive_root, archive_tag_prefix)
    for step_tag in archived_tags:
        train_dir = _resolve_existing_dataset_dir(
            requested_folder=Path(step_tag),
            work_dir=work_dir,
            archive_root=archive_root,
            step_tag=step_tag,
            split_name="train_folder",
            report=report,
            fallback_step_tags=[],
        )
        if _dir_has_pt_files(train_dir):
            discovered.append((step_tag, train_dir.resolve()))

    if report is not None:
        src = "archive" if discovered else "none"
        report.write(
            f"CONDITION_DISCOVERY source={src} prefix={archive_tag_prefix} tags={[tag for tag, _ in discovered]}"
        )
    return discovered

def _make_test_folder_checker(
    *,
    work_dir: Path,
    archive_root: Path,
    requested_folder: Path,
    archive_step_tags: List[str],
    tol: int,
    report: Report,
) -> Callable[[str], None]:
    def _checker(task_name: str) -> None:
        preferred = _preferred_label_attr_for_task(task_name)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict label checking: {task_name}")
        test_folder = _resolve_existing_dataset_dir(
            requested_folder=requested_folder,
            work_dir=work_dir,
            archive_root=archive_root,
            step_tag=archive_step_tags[0] if archive_step_tags else "",
            split_name="test_folder",
            report=report,
            fallback_step_tags=archive_step_tags[1:],
        )
        require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))
        report_label_stats(
            report,
            folder=test_folder,
            name=f"TEST CHECK [{task_name}]",
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_two_class_folder(
            test_folder,
            name=f"TEST CHECK [{task_name}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            test_folder,
            name=f"TEST CHECK [{task_name}] (contiguity gate)",
            report=report,
        )
    return _checker


def _build_train_args_from_config(
    model_cfg: Dict[str, Any],
    py: str,
    data_folder: Path,
    *,
    work_dir: Optional[Path] = None,
    task_override: Optional[str] = None,
    out_dir: Optional[str] = None,
    T_override: Optional[int] = None,
    test_folder: Optional[Path] = None,
    train_model_override: Optional[bool] = None,
) -> List[str]:
    task_eff = str(task_override).strip() if task_override is not None else str(model_cfg["task"]).strip()
    T_eff = int(T_override) if T_override is not None else int(model_cfg["T"])

    data_folder_arg = str(data_folder)
    out_dir_arg = str(out_dir) if out_dir is not None else None
    test_folder_arg = str(test_folder) if test_folder is not None else None
    if work_dir is not None:
        data_folder_arg = _to_work_dir_relative_path(data_folder, work_dir)
        if out_dir_arg is not None and str(out_dir_arg).strip() != "":
            out_dir_arg = _to_work_dir_relative_path(Path(out_dir_arg), work_dir)
        if test_folder_arg is not None and str(test_folder_arg).strip() != "":
            test_folder_arg = _to_work_dir_relative_path(Path(test_folder_arg), work_dir)

    args: List[str] = [
        py, "train_amr_dygformer.py",
        "--data_folder", data_folder_arg,
        "--task", task_eff,
        "--T", str(T_eff),
    ]

    if bool(model_cfg.get("use_task_hparams", True)):
        args.append("--use_task_hparams")
    else:
        args += [
            "--sliding_step", str(int(model_cfg["sliding_step"])),
            "--hidden", str(int(model_cfg["hidden"])),
            "--heads", str(int(model_cfg["heads"])),
            "--dropout", str(float(model_cfg["dropout"])),
            "--transformer_layers", str(int(model_cfg["transformer_layers"])),
            "--sage_layers", str(int(model_cfg["sage_layers"])),
            "--batch_size", str(int(model_cfg["batch_size"])),
            "--epochs", str(int(model_cfg["epochs"])),
            "--lr", str(float(model_cfg["lr"])),
        ]
        if bool(model_cfg.get("use_cls", False)):
            args.append("--use_cls")

    ns_flag = "true" if bool(model_cfg.get("neighbor_sampling", False)) else "false"
    args += ["--neighbor_sampling", ns_flag]

    if bool(model_cfg.get("neighbor_sampling", False)):
        args += [
            "--num_neighbors", str(model_cfg["num_neighbors"]),
            "--seed_count", str(int(model_cfg["seed_count"])),
            "--seed_strategy", str(model_cfg["seed_strategy"]),
            "--seed_batch_size", str(int(model_cfg["seed_batch_size"])),
            "--max_sub_batches", str(int(model_cfg["max_sub_batches"])),
        ]
    else:
        args += ["--max_neighbors", str(int(model_cfg["max_neighbors"]))]

    args += [
        "--attn_top_k", str(int(model_cfg["attn_top_k"])),
        "--attn_rank_by", str(model_cfg["attn_rank_by"]),
        "--fullgraph_attribution_pass", "true" if bool(model_cfg.get("fullgraph_attribution_pass", True)) else "false",
        "--emit_translational_figures", "true" if bool(model_cfg.get("emit_translational_figures", True)) else "false",
        "--translational_top_k", str(int(model_cfg.get("translational_top_k", 20))),
    ]
    
    args += [
    "--early_stopping", "true" if bool(model_cfg.get("early_stopping", True)) else "false",
    "--patience", str(int(model_cfg.get("patience", 7))),
    "--min_delta", str(float(model_cfg.get("min_delta", 1e-4))),
    "--save_best_only", "true" if bool(model_cfg.get("save_best_only", True)) else "false",
    "--lr_scheduler_on_plateau", "true" if bool(model_cfg.get("lr_scheduler_on_plateau", True)) else "false",
    "--lr_scheduler_factor", str(float(model_cfg.get("lr_scheduler_factor", 0.5))),
    "--lr_scheduler_patience", str(int(model_cfg.get("lr_scheduler_patience", 3))),
    "--lr_scheduler_min_lr", str(float(model_cfg.get("lr_scheduler_min_lr", 1e-6))),
]

    train_model_effective = bool(model_cfg.get("train_model", True)) if train_model_override is None else bool(train_model_override)
    train_flag = "true" if train_model_effective else "false"
    args += ["--train_model", train_flag]

    if out_dir_arg is not None and str(out_dir_arg).strip() != "":
        args += ["--out_dir", str(out_dir_arg)]

    num_workers_budget, torch_threads_budget, interop_budget = _resolve_train_runtime_budgets(
        neighbor_sampling=bool(model_cfg.get("neighbor_sampling", False))
    )
    args += [
        "--num_workers", str(int(num_workers_budget)),
        "--torch_num_threads", str(int(torch_threads_budget)),
        "--torch_num_interop_threads", str(int(interop_budget)),
    ]

    if test_folder_arg is not None and str(test_folder_arg).strip() != "":
        args += ["--test_folder", str(test_folder_arg)]

    return args


def _load_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_model_cfg_with_tuned_params(base_model_cfg: Dict[str, Any], tuned_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base_model_cfg)
    if not isinstance(tuned_cfg, dict):
        return merged
    allowed_keys = {
        "hidden", "heads", "dropout", "transformer_layers", "sage_layers",
        "batch_size", "epochs", "lr", "neighbor_sampling", "num_neighbors",
        "seed_count", "seed_strategy", "seed_batch_size", "max_sub_batches",
        "max_neighbors", "sliding_step", "use_cls", "use_task_hparams",
        "attn_top_k", "attn_rank_by", "fullgraph_attribution_pass",
        "emit_translational_figures", "translational_top_k",
    }
    for key, value in tuned_cfg.items():
        if key in allowed_keys:
            merged[key] = value
    return merged


def _run_lightweight_tuning(
    *,
    step_tag: str,
    work_dir: Path,
    archive_root: Path,
    py: str,
    dry: bool,
    report: Report,
    data_folder: Path,
    task_name: str,
    t_values: List[int],
    h_values: List[Optional[int]],
    tune_trials_quick: int,
    tune_finalists: int,
    tune_quick_epochs: int,
    tune_full_epochs: int,
    tune_split_seed: int,
) -> Dict[Tuple[int, Optional[int]], Dict[str, Any]]:
    tuned_by_combo: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
    total_jobs = max(1, len(t_values) * len(h_values))
    done = 0

    report.start_subtask_console(f"tune:{_safe_tag(step_tag)}", total=total_jobs, extra="validation-only search")
    try:
        for t_val in t_values:
            for h_val in h_values:
                task_eff = _task_with_horizon(task_name, int(h_val)) if h_val is not None else task_name
                combo_suffix = f"T{int(t_val)}" + (f"_h{int(h_val)}" if h_val is not None else "")
                tuning_root = work_dir / f"tuning_outputs_{_safe_tag(step_tag)}_{combo_suffix}"

                cmd = [
                    py, "tune_hparams.py",
                    "--data_folder", _to_work_dir_relative_path(data_folder, work_dir),
                    "--task", str(task_eff),
                    "--T", str(int(t_val)),
                    "--search_name", f"{_safe_tag(step_tag)}_{combo_suffix}",
                    "--out_dir", _to_work_dir_relative_path(tuning_root, work_dir),
                    "--n_trials_quick", str(int(tune_trials_quick)),
                    "--n_finalists", str(int(tune_finalists)),
                    "--quick_epochs", str(int(tune_quick_epochs)),
                    "--full_epochs", str(int(tune_full_epochs)),
                    "--split_seed", str(int(tune_split_seed)),
                    "--sliding_step", str(int(CONFIG["MODEL"].get("sliding_step", 1))),
                    "--max_neighbors", str(int(CONFIG["MODEL"].get("max_neighbors", 20))),
                    "--neighbor_sampling", "true" if bool(CONFIG["MODEL"].get("neighbor_sampling", False)) else "false",
                    "--num_neighbors", str(CONFIG["MODEL"].get("num_neighbors", "15,10")),
                    "--seed_count", str(int(CONFIG["MODEL"].get("seed_count", 256))),
                    "--seed_strategy", str(CONFIG["MODEL"].get("seed_strategy", "random")),
                    "--seed_batch_size", str(int(CONFIG["MODEL"].get("seed_batch_size", 64))),
                    "--max_sub_batches", str(int(CONFIG["MODEL"].get("max_sub_batches", 4))),
                    "--attn_top_k", str(int(CONFIG["MODEL"].get("attn_top_k", 10))),
                    "--attn_rank_by", str(CONFIG["MODEL"].get("attn_rank_by", "abs_diff")),
                    "--emit_translational_figures", "true" if bool(CONFIG["MODEL"].get("emit_translational_figures", True)) else "false",
                    "--fullgraph_attribution_pass", "true" if bool(CONFIG["MODEL"].get("fullgraph_attribution_pass", True)) else "false",
                    "--translational_top_k", str(int(CONFIG["MODEL"].get("translational_top_k", 20))),
                    "--early_stopping", "true",
                    "--patience", "7",
                    "--min_delta", "1e-4",
                    "--save_best_only", "true",
                    "--lr_scheduler_on_plateau", "true",
                    "--lr_scheduler_factor", "0.5",
                    "--lr_scheduler_patience", "3",
                    "--lr_scheduler_min_lr", "1e-6",
                ]
                if bool(CONFIG["MODEL"].get("use_cls", False)):
                    cmd.append("--use_cls")
                if bool(CONFIG["MODEL"].get("use_task_hparams", False)):
                    cmd.append("--use_task_hparams")
                _run_cmd_with_unrecognized_arg_fallback(
                    cmd,
                    dry=dry,
                    cwd=work_dir,
                    report=report,
                    stream_output=True,
                    fallback_context=f"{step_tag}:{combo_suffix}",
                )

                best_cfg_path = tuning_root / "best_config.json"
                best_cfg_payload = _load_json_file(best_cfg_path)
                if not best_cfg_payload:
                    raise FileNotFoundError(f"Missing or unreadable best_config.json after tuning: {best_cfg_path}")
                best_cfg = best_cfg_payload.get("config", best_cfg_payload)
                if not isinstance(best_cfg, dict) or not best_cfg:
                    raise RuntimeError(f"best_config.json does not contain a usable 'config' payload: {best_cfg_path}")

                tuned_by_combo[(int(t_val), h_val)] = _merge_model_cfg_with_tuned_params(CONFIG["MODEL"], best_cfg)
                archive_dst = _archive_dst_for_training_run(
                    archive_root=archive_root,
                    step_tag=f"{_safe_tag(step_tag)}_tuning",
                    t_val=int(t_val),
                    h_val=h_val,
                    run_all_T=len(t_values) > 1,
                    run_all_horizons=len(h_values) > 1 or any(v is not None for v in h_values),
                )
                archive_training_outputs(archive_dst, work_dir=work_dir, dry=dry, report=report, src_dir=tuning_root)

                done += 1
                report.update_subtask_console(current=done, total=total_jobs, extra=combo_suffix)
    finally:
        report.finish_subtask_console(force_complete=True)

    return tuned_by_combo


def _load_archived_tuned_cfg_map(
    *,
    step_tag: str,
    work_dir: Path,
    archive_root: Path,
    report: Optional[Report],
    t_values: List[int],
    h_values: List[Optional[int]],
) -> Dict[Tuple[int, Optional[int]], Dict[str, Any]]:
    del work_dir

    tuned_by_combo: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
    tuning_step_tag = f"{_safe_tag(step_tag)}_tuning"
    tuning_root = archive_root / tuning_step_tag

    run_all_T_effective = len(t_values) > 1
    run_all_horizons_effective = len(h_values) > 1 or any(v is not None for v in h_values)

    for t_val in t_values:
        for h_val in h_values:
            archive_dst = _archive_dst_for_training_run(
                archive_root=archive_root,
                step_tag=tuning_step_tag,
                t_val=int(t_val),
                h_val=h_val,
                run_all_T=run_all_T_effective,
                run_all_horizons=run_all_horizons_effective,
            )

            candidate_paths: List[Path] = [archive_dst / "best_config.json"]

            fallback_paths: List[Path] = []
            if tuning_root.exists():
                if not run_all_T_effective and not run_all_horizons_effective:
                    fallback_paths.append(tuning_root / "best_config.json")
                elif not run_all_T_effective and h_val is not None:
                    fallback_paths.append(tuning_root / f"h{int(h_val)}" / "best_config.json")
                elif run_all_T_effective and not run_all_horizons_effective:
                    fallback_paths.append(tuning_root / f"T{int(t_val)}" / "best_config.json")
                elif run_all_T_effective and h_val is not None:
                    fallback_paths.append(tuning_root / f"T{int(t_val)}" / f"h{int(h_val)}" / "best_config.json")

            best_cfg_payload: Dict[str, Any] = {}
            chosen_path: Optional[Path] = None
            checked_paths: List[Path] = []

            for candidate in candidate_paths + fallback_paths:
                candidate_resolved = candidate.resolve()
                if candidate_resolved in checked_paths:
                    continue
                checked_paths.append(candidate_resolved)

                best_cfg_payload = _load_json_file(candidate)
                if best_cfg_payload:
                    chosen_path = candidate
                    break

            if not best_cfg_payload:
                if report is not None:
                    report.write(
                        f"ARCHIVED_TUNED_CFG_MISSING step={step_tag} T={int(t_val)} h={h_val} "
                        f"checked={[str(p) for p in (candidate_paths + fallback_paths)]}"
                    )
                continue

            best_cfg = best_cfg_payload.get("config", best_cfg_payload)
            if not isinstance(best_cfg, dict) or not best_cfg:
                if report is not None:
                    report.write(
                        f"ARCHIVED_TUNED_CFG_INVALID step={step_tag} T={int(t_val)} h={h_val} path={chosen_path}"
                    )
                continue

            tuned_by_combo[(int(t_val), h_val)] = _merge_model_cfg_with_tuned_params(CONFIG["MODEL"], best_cfg)

            if report is not None:
                report.write(
                    f"ARCHIVED_TUNED_CFG_FOUND step={step_tag} T={int(t_val)} h={h_val} path={chosen_path}"
                )

    if report is not None:
        report.write(
            f"ARCHIVED_TUNED_CFG_LOAD step={step_tag} combos_loaded={len(tuned_by_combo)} "
            f"requested={len(t_values) * len(h_values)}"
        )

    return tuned_by_combo


def _build_sim_extra_args_global_only(sim_cfg: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts += [
        "--num_regions", str(int(sim_cfg["num_regions"])),
        "--num_patients", str(int(sim_cfg["num_patients"])),
        "--num_staff", str(int(sim_cfg["num_staff"])),
        "--num_wards", str(int(sim_cfg["num_wards"])),
    ]
    if bool(sim_cfg.get("export_yaml", True)):
        parts.append("--export_yaml")
    if not bool(sim_cfg.get("export_gif", True)):
        parts.append("--no_export_gif")
    if bool(sim_cfg.get("override_staff_wards_per_staff", False)):
        parts += ["--staff_wards_per_staff", str(int(sim_cfg["staff_wards_per_staff"]))]

    if bool(sim_cfg.get("enable_superspreader", False)) and str(sim_cfg.get("superspreader_staff", "")).strip():
        parts += [
            "--superspreader_staff", str(sim_cfg["superspreader_staff"]),
            "--superspreader_state", str(sim_cfg["superspreader_state"]),
            "--superspreader_start_day", str(int(sim_cfg["superspreader_start_day"])),
            "--superspreader_end_day", str(int(sim_cfg["superspreader_end_day"])),
            "--superspreader_patient_frac_mult", str(float(sim_cfg["superspreader_patient_frac_mult"])),
            "--superspreader_patient_min_add", str(int(sim_cfg["superspreader_patient_min_add"])),
            "--superspreader_staff_contacts", str(int(sim_cfg["superspreader_staff_contacts"])),
            "--superspreader_edge_weight_mult", str(float(sim_cfg["superspreader_edge_weight_mult"])),
        ]

    if bool(sim_cfg.get("enable_admit_import_seasonality", False)) and str(sim_cfg.get("admit_import_seasonality", "none")) != "none":
        mode = str(sim_cfg["admit_import_seasonality"])
        parts += [
            "--admit_import_seasonality", mode,
            "--admit_import_period_days", str(int(sim_cfg["admit_import_period_days"])),
            "--admit_import_pmax_cs", str(float(sim_cfg["admit_import_pmax_cs"])),
            "--admit_import_pmax_cr", str(float(sim_cfg["admit_import_pmax_cr"])),
            "--admit_import_pmax_is", str(float(sim_cfg.get("admit_import_pmax_is", 1.0))),
            "--admit_import_pmax_ir", str(float(sim_cfg.get("admit_import_pmax_ir", 1.0))),
        ]
        if mode == "sinusoid":
            parts += [
                "--admit_import_amp", str(float(sim_cfg["admit_import_amp"])),
                "--admit_import_phase_day", str(int(sim_cfg["admit_import_phase_day"])),
            ]
        elif mode == "piecewise":
            parts += [
                "--admit_import_high_start_day", str(int(sim_cfg["admit_import_high_start_day"])),
                "--admit_import_high_end_day", str(int(sim_cfg["admit_import_high_end_day"])),
                "--admit_import_high_mult", str(float(sim_cfg["admit_import_high_mult"])),
                "--admit_import_low_mult", str(float(sim_cfg["admit_import_low_mult"])),
            ]
        elif mode == "shock":
            parts += [
                "--admit_import_shock_min_days", str(int(sim_cfg["admit_import_shock_min_days"])),
                "--admit_import_shock_max_days", str(int(sim_cfg["admit_import_shock_max_days"])),
                "--admit_import_shock_mult_min", str(float(sim_cfg["admit_import_shock_mult_min"])),
                "--admit_import_shock_mult_max", str(float(sim_cfg["admit_import_shock_mult_max"])),
            ]

    passthrough_keys = [
        "beta_res_mult",
        "beta_state_mult_cs",
        "beta_state_mult_cr",
        "beta_state_mult_is",
        "beta_state_mult_ir",
        "crossward_hand_hygiene_enabled",
        "crossward_hand_hygiene_mult",
        "passive_screen_capacity_per_week",
        "active_symptom_screening",
        "admission_screen_symptomatic_only",
        "isolation_trigger_policy",
        "no_abx_sensitive_selection_weight",
        "abx_resistant_selection_weight",
    ]
    for key in passthrough_keys:
        if key in sim_cfg and sim_cfg[key] is not None:
            parts += [f"--{key}", str(sim_cfg[key])]
    return " ".join(parts)

def _build_convert_extra_args(convert_cfg: Dict[str, Any]) -> str:
    parts: List[str] = []

    horizons = str(convert_cfg.get("horizons", "")).strip()
    if horizons:
        parts += ["--horizons", horizons]

    workers = convert_cfg.get("workers", None)
    if workers is None:
        workers_int = _machine_cpu_count()
    else:
        workers_int = int(workers)

    if workers_int < 0:
        raise ValueError(f"CONFIG['CONVERT']['workers'] must be >= 0, got {workers_int}")

    parts += ["--workers", str(workers_int)]
    return " ".join(parts)


# =============================================================================
# Code staging
# =============================================================================

def _is_python_package_dir(d: Path) -> bool:
    return d.is_dir() and (d / "__init__.py").is_file()


def _copy_tree_excluding(src: Path, dst: Path, exclude_names: set, dry: bool) -> None:
    if dry:
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in exclude_names:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def stage_code_into_workdir(project_root: Path, work_dir: Path, dry: bool) -> None:
    exclude = {
        ".git", ".github",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".venv", "venv", "env", "envs",
        "dist", "build", "node_modules",
        "training_outputs",
        "repro_artifacts_steps_1_9",
        "experiments_results",
    }

    ensure_dir(work_dir, dry)

    py_files = sorted(project_root.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(f"No top-level *.py files found in {project_root.resolve()}")

    for f in py_files:
        dst = work_dir / f.name
        if not dry:
            shutil.copy2(f, dst)

    for d in sorted(project_root.iterdir()):
        if d.name in exclude:
            continue
        if _is_python_package_dir(d):
            _copy_tree_excluding(d, work_dir / d.name, exclude, dry)


def _make_run_dirs(results_parent: Path, timestamped: bool, dry: bool) -> Dict[str, Path]:
    del timestamped
    run_root = results_parent.resolve()
    ensure_dir(run_root, dry)

    report_filename = str(os.environ.get("DT_REPORT_FILENAME", "run_report.txt")).strip() or "run_report.txt"
    report_path = (run_root / report_filename).resolve()
    return {
        "run_root": run_root,
        "report_path": report_path,
    }


def _make_track_dirs(run_root: Path, track_name: str, dry: bool) -> Dict[str, Path]:
    track_root = (run_root / track_name).resolve()
    work_dir = (track_root / "work").resolve()
    archive_root = (work_dir / "repro_artifacts_steps_1_9").resolve()
    graphml_keep_root = (track_root / "kept_graphml").resolve()

    ensure_dir(track_root, dry)
    ensure_dir(work_dir, dry)
    ensure_dir(archive_root, dry)
    ensure_dir(graphml_keep_root, dry)

    return {
        "track_root": track_root,
        "work_dir": work_dir,
        "archive_root": archive_root,
        "graphml_keep_root": graphml_keep_root,
    }


def _default_pt_out_dir_for_track(run_root: Path, track_name: str) -> Path:
    return (run_root / track_name / "work" / "repro_artifacts_steps_1_9" / "pt_copies").resolve()


def _find_all_dirs(work_dir: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for pat in patterns:
        for candidate in sorted(work_dir.glob(pat)):
            if candidate.is_dir():
                rp = candidate.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
    return out


def _cleanup_transient_pt_files(
    *,
    work_dir: Path,
    pt_out_dir: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> int:
    """
    Fast cleanup of transient PT-producing outputs while preserving resume-critical assets.

    Preserved on purpose:
      - canonical Step 2 *_pt_flat folders
      - synthetic_amr_graphs_train
      - synthetic_amr_graphs_test_frozen
      - synthetic_amr_graphs_test
      - archived outputs under repro_artifacts_steps_1_9

    Removed as whole transient directories when present:
      - pt_out_dir scratch copies
      - training_outputs* folders in work_dir
      - step5 ablation folder
      - step6 delay generated train folders
      - step6 frequency generated train folders
      - step7 sweep generated train folder

    Fallback:
      - if whole-directory removal fails for any transient root, delete only its .pt files
    """
    transient_roots: List[Path] = []

    transient_roots.append(pt_out_dir.resolve())
    transient_roots.extend(p.resolve() for p in work_dir.glob("training_outputs*") if p.is_dir())

    step5_root = work_dir / STEP5_ABLATION_REL
    if step5_root.exists() and step5_root.is_dir():
        transient_roots.append(step5_root.resolve())

    task_name = str(CONFIG.get("MODEL", {}).get("task", "")).strip()
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "delay")))
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "frequency")))
    transient_roots.extend(_find_all_dirs(work_dir, _step_dataset_globs(task_name, "sweep")))

    protected_roots = {
        (work_dir / _trajectory_pt_flat_dir(name)).resolve()
        for name in CANONICAL_TRAJECTORY_NAMES
    }
    protected_roots.update(
        {
            (work_dir / BASELINE_TRAIN_REL).resolve(),
            (work_dir / BASELINE_TEST_REL).resolve(),
            (work_dir / LIVE_TEST_REL).resolve(),
            (work_dir / "repro_artifacts_steps_1_9").resolve(),
        }
    )

    seen_roots = set()
    unique_roots: List[Path] = []
    for root in transient_roots:
        rp = root.resolve()
        if (
            rp.exists()
            and rp.is_dir()
            and rp not in seen_roots
            and rp not in protected_roots
            and not any(protected in rp.parents for protected in protected_roots)
        ):
            seen_roots.add(rp)
            unique_roots.append(rp)

    if dry:
        if report is not None:
            for root in unique_roots:
                report.write(f"[DRY_RUN] PT_CLEANUP_TRANSIENT remove_dir={root}")
        return len(unique_roots)

    removed_dirs = 0
    fallback_deleted_pts = 0

    if unique_roots and report is not None:
        report.start_subtask_console(
            "pt_cleanup_transient",
            total=len(unique_roots),
            extra="remove transient directories",
        )

    try:
        for idx, root in enumerate(unique_roots, start=1):
            try:
                shutil.rmtree(root)
                removed_dirs += 1
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"removed {root.name}",
                    )
            except FileNotFoundError:
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"already missing {root.name}",
                    )
                continue
            except Exception:
                pts = sorted(p for p in root.rglob("*.pt") if p.is_file())
                for pt_file in pts:
                    try:
                        pt_file.unlink()
                        fallback_deleted_pts += 1
                    except FileNotFoundError:
                        continue
                if report is not None:
                    report.update_subtask_console(
                        current=idx,
                        total=len(unique_roots),
                        extra=f"fallback .pt cleanup in {root.name}",
                    )
    finally:
        if unique_roots and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        for root in unique_roots:
            report.write(f"PT_CLEANUP_TRANSIENT root={root}")
        report.write(
            "PT_CLEANUP_TRANSIENT "
            f"removed_dirs={removed_dirs} "
            f"fallback_deleted_pts={fallback_deleted_pts} "
            f"track_work_dir={work_dir}"
        )

    return removed_dirs + fallback_deleted_pts

# =============================================================================
# Label reading + gates
# =============================================================================

def _task_name_to_label_attr(task_name: str) -> Optional[str]:
    m = re.search(r"_h(\d+)$", str(task_name))
    if not m:
        return None
    h = m.group(1)
    if str(task_name).startswith("endogenous_transmission_majority_h"):
        return f"y_h{h}_trans_majority"
    if str(task_name).startswith("endogenous_importation_majority_h"):
        return f"y_h{h}_endog_majority"
    if str(task_name).startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac_cls"
    return None


def _preferred_label_attr_for_task(task_name: str) -> Optional[str]:
    return _task_name_to_label_attr(task_name)


def _read_label_from_pt(
    pt_path: Path,
    preferred_label_attr: Optional[str] = None,
    *,
    strict_preferred: bool = False,
) -> Optional[int]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)

    def _to_int(x: Any) -> Optional[int]:
        try:
            if hasattr(x, "item"):
                return int(x.item())
            return int(x)
        except Exception:
            return None

    preferred = str(preferred_label_attr).strip() if preferred_label_attr else None

    if strict_preferred and not preferred:
        return None

    if preferred:
        if hasattr(obj, preferred):
            v = _to_int(getattr(obj, preferred))
            if v in (0, 1):
                return v
        if strict_preferred:
            return None

    candidates: List[str] = []
    if hasattr(obj, "y_h7_trans_majority"):
        candidates.append("y_h7_trans_majority")
    if hasattr(obj, "y_h7_endog_majority") and "y_h7_endog_majority" not in candidates:
        candidates.append("y_h7_endog_majority")
    if hasattr(obj, "y"):
        candidates.append("y")

    for name in dir(obj):
        if name.startswith("y_") and name not in candidates:
            candidates.append(name)

    for name in candidates:
        try:
            x = getattr(obj, name)
        except Exception:
            continue
        v = _to_int(x)
        if v in (0, 1):
            return v

    return None


def label_stats_for_pt_folder(
    folder: Path,
    preferred_label_attr: Optional[str] = None,
    *,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    pts = sorted(folder.glob("*.pt"))
    zeros: List[Path] = []
    ones: List[Path] = []
    unknown: List[Path] = []

    for p in pts:
        lab = _read_label_from_pt(
            p,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=strict_preferred,
        )
        if lab == 0:
            zeros.append(p)
        elif lab == 1:
            ones.append(p)
        else:
            unknown.append(p)

    return {
        "n_total": len(pts),
        "n0": len(zeros),
        "n1": len(ones),
        "n_unknown": len(unknown),
        "files0": zeros,
        "files1": ones,
        "files_unknown": unknown,
    }


def report_label_stats(
    report: Report,
    *,
    folder: Path,
    name: str,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        report.write(f"[{name}] folder={folder} (missing)")
        return {"n_total": 0, "n0": 0, "n1": 0, "n_unknown": 0}
    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    report.write(f"[{name}] folder={folder}")
    report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")
    return stats


def assert_nonempty_known_labels_folder(
    folder: Path,
    *,
    name: str,
    report: Optional[Report] = None,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    if report is not None:
        report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")

    if stats["n_total"] == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    if stats["n_unknown"] > 0:
        ex = stats["files_unknown"][0]
        raise RuntimeError(
            f"[{name}] unknown label for {stats['n_unknown']} files (example: {ex.name}). Refusing to proceed."
        )

    return stats


def warn_if_single_label(stats: Dict[str, Any], *, name: str, report: Optional[Report] = None) -> None:
    if int(stats.get("n0", 0)) == 0 or int(stats.get("n1", 0)) == 0:
        msg = (
            f"WARNING: [{name}] TRAIN is single-label (n0={stats.get('n0', 0)}, n1={stats.get('n1', 0)}). "
            "Training will run; ROC/AUROC may be degenerate."
        )
        if report is not None:
            report.write(msg)
        else:
            print(msg, flush=True)


def assert_two_class_folder(
    folder: Path,
    *,
    name: str,
    require_balanced: bool,
    balance_tolerance: int = 1,
    report: Optional[Report] = None,
    preferred_label_attr: Optional[str] = None,
    strict_preferred: bool = False,
) -> Dict[str, Any]:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    stats = label_stats_for_pt_folder(
        folder,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=strict_preferred,
    )
    if report is not None:
        report.write(f"[{name}] n_total={stats['n_total']} n0={stats['n0']} n1={stats['n1']} unknown={stats['n_unknown']}")

    if stats["n_total"] == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    if stats["n_unknown"] > 0:
        ex = stats["files_unknown"][0]
        raise RuntimeError(
            f"[{name}] unknown label for {stats['n_unknown']} files (example: {ex.name}). Refusing to proceed."
        )

    if stats["n0"] == 0 or stats["n1"] == 0:
        raise RuntimeError(f"[{name}] single-label dataset: n0={stats['n0']} n1={stats['n1']} in {folder.resolve()}")

    if require_balanced and abs(stats["n0"] - stats["n1"]) > int(balance_tolerance):
        raise RuntimeError(
            f"[{name}] test folder not balanced: n0={stats['n0']} n1={stats['n1']} "
            f"(tolerance={balance_tolerance}) in {folder.resolve()}"
        )

    return stats


# =============================================================================
# PT metadata / contiguity helpers
# =============================================================================

def _to_int_maybe(x: Any) -> Optional[int]:
    try:
        if hasattr(x, "item"):
            return int(x.item())
        return int(x)
    except Exception:
        return None


def _extract_sim_id_and_day_from_pt_obj(pt_path: Path, obj: Any) -> Tuple[str, Optional[int]]:
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
    if sim_id is None:
        m_flat = re.search(r"^(.+__sim_\d+?)__(.+)$", name, flags=re.IGNORECASE)
        if m_flat:
            sim_id = m_flat.group(1)
    if sim_id is None:
        m = re.search(r"^(sim[^_]*?__[^_]+__[^.]+?)(?:__|_|-)(?:day|d)\d{1,4}\.pt$", name, flags=re.IGNORECASE)
        if m:
            sim_id = m.group(1)
        else:
            m2 = re.search(r"^(sim[^.]+?)(?:__amr|__|\.pt$)", name, flags=re.IGNORECASE)
            if m2:
                sim_id = m2.group(1)

    if sim_id is None:
        sim_id = f"{pt_path.parent.name}::{pt_path.stem}"

    if day is None:
        for pat in [
            r"(?:^|[_-]|__)(?:day)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
            r"(?:^|[_-]|__)(?:d)[_-]?(\d{1,4})(?:[_-]|__|\.|$)",
        ]:
            mm = re.search(pat, name, flags=re.IGNORECASE)
            if mm:
                day = int(mm.group(1))
                break

    return str(sim_id), day


def _read_sim_day(pt_path: Path) -> Tuple[str, int]:
    import torch

    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    sim_id, day = _extract_sim_id_and_day_from_pt_obj(pt_path, obj)
    if day is None:
        raise RuntimeError(
            f"Could not extract day index for pt: {pt_path.name} (sim_id='{sim_id}'). "
            "Cannot validate contiguity safely."
        )
    if int(day) <= 0:
        raise RuntimeError(f"Invalid day={day} for pt: {pt_path.name} (sim_id='{sim_id}')")
    return str(sim_id), int(day)


def assert_contiguous_days_per_sim(
    folder: Path,
    *,
    name: str,
    report: Optional[Report] = None,
) -> None:
    if not folder.exists():
        raise FileNotFoundError(f"[{name}] folder not found: {folder.resolve()}")

    pts = sorted(folder.glob("*.pt"))
    if len(pts) == 0:
        raise RuntimeError(f"[{name}] no .pt files found in {folder.resolve()}")

    sim_days: Dict[str, List[int]] = {}
    for p in pts:
        sim_id, day = _read_sim_day(p)
        sim_days.setdefault(sim_id, []).append(int(day))

    dup_bad: List[Tuple[str, int]] = []
    gap_bad: List[Tuple[str, int, int]] = []

    for sim_id, days in sim_days.items():
        ds_sorted = sorted(int(d) for d in days)
        for i in range(len(ds_sorted) - 1):
            if ds_sorted[i + 1] == ds_sorted[i]:
                dup_bad.append((sim_id, ds_sorted[i]))
                break

        ds = sorted(set(ds_sorted))
        for i in range(len(ds) - 1):
            if ds[i + 1] != ds[i] + 1:
                gap_bad.append((sim_id, ds[i], ds[i + 1]))
                break

    if dup_bad:
        sim_id, d = dup_bad[0]
        msg = (
            f"[{name}] Duplicate day entries detected for sim_id='{sim_id}', day={d}. "
            "This will confuse TemporalGraphDataset metadata grouping."
        )
        if report is not None:
            report.write(msg)
        raise RuntimeError(msg)

    if gap_bad:
        sim_id, d0, d1 = gap_bad[0]
        msg = (
            f"[{name}] Non-contiguous days detected for sim_id='{sim_id}': "
            f"... {d0}, {d1}, ... (gap={d1 - d0}). "
            "This will crash TemporalGraphDataset in metadata mode."
        )
        if report is not None:
            report.write(msg)
        raise RuntimeError(msg)

    if report is not None:
        report.write(f"[{name}] contiguity OK (per-sim day sequences are consecutive, no duplicates).")


# =============================================================================
# Canonical trajectory generation / PT preparation / frozen baseline build
# =============================================================================

def _extra_args_from_env(env_key: str) -> List[str]:
    s = str(os.environ.get(env_key, "")).strip()
    return shlex.split(s) if s else []


def _canonical_sim_dirs(work_dir: Path, trajectory_name: str) -> List[Path]:
    traj_dir = work_dir / trajectory_name
    return sorted([p for p in traj_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")]) if traj_dir.exists() else []


def _trajectory_pt_flat_dir(trajectory_name: str) -> Path:
    return Path(f"{trajectory_name}_pt_flat")


def _global_sim_id_from_sim_dir(sim_dir: Path) -> str:
    return f"{sim_dir.parent.name}__{sim_dir.name}"


def _prepare_graph_folder_figures_compare_pool(
    *,
    src_roots: List[Path],
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would prepare compare pool -> {dst_root}")
        return

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    sim_counter = 0

    for src_root in src_roots:
        if not src_root.exists():
            raise FileNotFoundError(f"Missing raw graph source folder: {src_root.resolve()}")

        sim_dirs = sorted([p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("sim_")])
        if sim_dirs:
            for sim_dir in sim_dirs:
                dst_sim = dst_root / f"sim_{sim_counter:03d}__{_safe_tag(src_root.name)}"
                shutil.copytree(sim_dir, dst_sim)
                sim_counter += 1
                copied += 1
        else:
            for graph_file in sorted(src_root.rglob("*.graphml")):
                rel = graph_file.relative_to(src_root)
                dst_file = dst_root / _safe_tag(src_root.name) / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(graph_file, dst_file)
                copied += 1

            labels_dir = src_root / "labels"
            if labels_dir.exists() and labels_dir.is_dir():
                for label_file in sorted(labels_dir.rglob("*")):
                    if not label_file.is_file():
                        continue
                    rel = label_file.relative_to(src_root)
                    dst_file = dst_root / _safe_tag(src_root.name) / rel
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(label_file, dst_file)

    if report is not None:
        report.write(f"GRAPH_COMPARE_POOL prepared={dst_root} copied_entries={copied}")


def archive_step1_baseline_pair_figures(
    *,
    py: str,
    work_dir: Path,
    archive_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> Optional[Path]:
    train_sources = [work_dir / name for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_sources = [work_dir / name for name in CANONICAL_TEST_TRAJECTORIES]

    if not all(_has_graphml_files(p) for p in train_sources + test_sources):
        if report is not None:
            report.write(
                "SKIP step1 baseline pair figures: missing raw GraphML under one or more canonical trajectory folders."
            )
        return None

    tmp_root = work_dir / "_tmp_step1_baseline_graph_compare"
    train_pool = tmp_root / "baseline_train_raw"
    test_pool = tmp_root / "baseline_test_frozen_raw"
    out_dir = archive_root / DATASET_FIGURES_REL / "step1_baseline_train_vs_test"

    try:
        _prepare_graph_folder_figures_compare_pool(
            src_roots=train_sources,
            dst_root=train_pool,
            dry=dry,
            report=report,
        )
        _prepare_graph_folder_figures_compare_pool(
            src_roots=test_sources,
            dst_root=test_pool,
            dry=dry,
            report=report,
        )

        archive_dataset_pair_figures(
            py=py,
            graph_dir=train_pool,
            compare_dir=test_pool,
            dst_dir=out_dir,
            dry=dry,
            report=report,
            cwd=cwd,
            identity=identity,
            title="Step 1 baseline raw-graph summary: train vs frozen test",
            label="baseline_train",
            compare_label="baseline_test_frozen",
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
    finally:
        if not dry and tmp_root.exists():
            _safe_rmtree(tmp_root)

    return out_dir


def ensure_step1_baseline_pair_figures_before_purge(
    *,
    py: str,
    work_dir: Path,
    archive_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    cwd: Optional[Path] = None,
    identity: str = "Harry Triantafyllidis",
    enable_graph_folder_figures: bool = False,
) -> Optional[Path]:
    if not enable_graph_folder_figures:
        if report is not None:
            report.write("STEP1 baseline pair figures disabled by CLI")
        return None

    out_dir = archive_root / DATASET_FIGURES_REL / "step1_baseline_train_vs_test"

    if out_dir.exists() and any(out_dir.glob("*.png")):
        if report is not None:
            report.write(f"STEP1 baseline pair figures already present -> {out_dir}")
        return out_dir

    return archive_step1_baseline_pair_figures(
        py=py,
        work_dir=work_dir,
        archive_root=archive_root,
        dry=dry,
        report=report,
        cwd=cwd,
        identity=identity,
        enable_graph_folder_figures=enable_graph_folder_figures,
    )


def _existing_baseline_keep_sources(dst_root: Path, expected_names: List[str]) -> List[Path]:
    if not dst_root.exists():
        return []
    resolved: List[Path] = []
    for name in expected_names:
        p = dst_root / name
        if not p.exists():
            return []
        resolved.append(p)
    return resolved



def _safe_rmtree(path: Path, *, retries: int = 8, delay_s: float = 0.25) -> None:
    """Best-effort recursive delete with small retries for transient macOS directory races."""
    last_exc: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_exc = exc
            time.sleep(float(delay_s))
    if path.exists() and last_exc is not None:
        raise last_exc



def _pool_into_existing_keep_root(
    *,
    src_roots: List[Path],
    dst_root: Path,
    dry: bool,
    report: Optional[Report] = None,
    label: str,
) -> None:
    tmp_root = dst_root.parent / f"{dst_root.name}__pooled_tmp"
    backup_root = dst_root.parent / f"{dst_root.name}__prepool_backup"
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would pool baseline keep roots -> {dst_root} label={label}")
        return

    if tmp_root.exists():
        _safe_rmtree(tmp_root)
    if backup_root.exists():
        _safe_rmtree(backup_root)

    _prepare_graph_folder_figures_compare_pool(
        src_roots=src_roots,
        dst_root=tmp_root,
        dry=False,
        report=report,
    )

    if dst_root.exists():
        dst_root.rename(backup_root)
    tmp_root.rename(dst_root)
    if backup_root.exists():
        _safe_rmtree(backup_root)

    if report is not None:
        report.write(f"GRAPHML_BASELINE_POOL_READY dst={dst_root} label={label}")



def keep_step1_baseline_graphml_before_purge(
    *,
    work_dir: Path,
    graphml_keep_root: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    step_name = "step4_baseline"
    train_dst = _kept_graphml_split_dir(graphml_keep_root, step_name, "train")
    test_dst = _kept_graphml_split_dir(graphml_keep_root, step_name, "test")

    kept_train_sources = _existing_baseline_keep_sources(train_dst, list(CANONICAL_TRAIN_TRAJECTORIES))
    kept_test_sources = _existing_baseline_keep_sources(test_dst, list(CANONICAL_TEST_TRAJECTORIES))

    if kept_train_sources and kept_test_sources:
        _pool_into_existing_keep_root(
            src_roots=kept_train_sources,
            dst_root=train_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::train",
        )
        _pool_into_existing_keep_root(
            src_roots=kept_test_sources,
            dst_root=test_dst,
            dry=dry,
            report=report,
            label=f"{step_name}::test",
        )
        return

    train_sources = [work_dir / name for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_sources = [work_dir / name for name in CANONICAL_TEST_TRAJECTORIES]
    keep_step_graphml_split(
        graphml_keep_root=graphml_keep_root,
        step_name=step_name,
        train_src_roots=train_sources,
        test_src_roots=test_sources,
        dry=dry,
        report=report,
        pool_train=all(_has_graphml_files(p) for p in train_sources),
        pool_test=all(_has_graphml_files(p) for p in test_sources),
    )


def _admission_import_weights_from_spec(spec: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Return CS/CR/IS/IR admission-importation weights for a trajectory spec.

    New IS/IR importation fields default to the corresponding CS/CR regime
    weight. This keeps future trajectory specs from silently switching off
    imported infected cases when only the older CS/CR fields are provided.
    """
    p_cs = float(spec["p_admit_import_cs"])
    p_cr = float(spec["p_admit_import_cr"])
    p_is = float(spec.get("p_admit_import_is", p_cs))
    p_ir = float(spec.get("p_admit_import_ir", p_cr))
    return p_cs, p_cr, p_is, p_ir


def _run_generate_amr_sim(
    *,
    py: str,
    out_dir: Path,
    seed: int,
    num_days: int,
    p_cs: float,
    p_cr: float,
    p_is: float = 0.0,
    p_ir: float = 0.0,
    discharge_frac: float = 0.0,
    discharge_min: int,
    extra_cli_args: Optional[List[str]],
    dry: bool,
    cwd: Path,
    report: Report,
) -> None:
    cmd: List[str] = [
        py,
        "generate_amr_data.py",
        "--output_dir", str(out_dir),
        "--seed", str(int(seed)),
        "--num_days", str(int(num_days)),
        "--daily_discharge_frac", str(float(discharge_frac)),
        "--daily_discharge_min_per_ward", str(int(discharge_min)),
        "--p_admit_import_cs", str(float(p_cs)),
        "--p_admit_import_cr", str(float(p_cr)),
        "--p_admit_import_is", str(float(p_is)),
        "--p_admit_import_ir", str(float(p_ir)),
    ]
    cmd += _extra_args_from_env("DT_SIM_EXTRA_ARGS")
    if extra_cli_args:
        cmd += [str(x) for x in extra_cli_args]
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def step1_generate_canonical_trajectories(*, work_dir: Path, py: str, dry: bool, report: Report) -> None:
    cfg = CONFIG["STEP1"]
    n_sims = int(cfg["n_sims_per_trajectory"])
    num_days = int(cfg["num_days"])
    traj_cfg: Dict[str, Dict[str, Any]] = dict(cfg["trajectories"])

    missing = [name for name in CANONICAL_TRAJECTORY_NAMES if name not in traj_cfg]
    if missing:
        raise RuntimeError(f"STEP1 config missing canonical trajectories: {missing}")

    total_jobs = len(CANONICAL_TRAJECTORY_NAMES) * n_sims
    done = 0

    for name in CANONICAL_TRAJECTORY_NAMES:
        spec = traj_cfg[name]
        p_cs, p_cr, p_is, p_ir = _admission_import_weights_from_spec(spec)
        traj_dir = work_dir / name
        report.kv(
            f"trajectory::{name}",
            {
                "dir": str(traj_dir),
                "seed_base": int(spec["seed_base"]),
                "p_admit_import_cs": p_cs,
                "p_admit_import_cr": p_cr,
                "p_admit_import_is": p_is,
                "p_admit_import_ir": p_ir,
                "daily_discharge_frac": float(spec["daily_discharge_frac"]),
                "daily_discharge_min_per_ward": int(spec["daily_discharge_min_per_ward"]),
                "extra_sim_args": list(spec.get("extra_sim_args", [])),
                "n_sims": n_sims,
                "num_days": num_days,
            },
        )

        if not dry and traj_dir.exists():
            shutil.rmtree(traj_dir)
        ensure_dir(traj_dir, dry)

        for r in range(n_sims):
            sim_dir = traj_dir / f"sim_{r:03d}"
            ensure_dir(sim_dir, dry)

            _report_progress(
                report,
                prefix="STEP1 progress",
                current=done + 1,
                total=total_jobs,
                extra=f"{name} / sim_{r:03d}",
            )

            _run_generate_amr_sim(
                py=py,
                out_dir=sim_dir,
                seed=int(spec["seed_base"]) + r,
                num_days=num_days,
                p_cs=p_cs,
                p_cr=p_cr,
                p_is=p_is,
                p_ir=p_ir,
                discharge_frac=float(spec["daily_discharge_frac"]),
                discharge_min=int(spec["daily_discharge_min_per_ward"]),
                extra_cli_args=list(spec.get("extra_sim_args", [])),
                dry=dry,
                cwd=work_dir,
                report=report,
            )
            done += 1


def _task_uses_fixed_early_outbreak_threshold(task_name: str) -> bool:
    return _task_family(task_name) == "early_outbreak_warning"


def _ensure_task_specific_convert_support_files(*, work_dir: Path, dry: bool, report: Optional[Report] = None) -> List[str]:
    task_name = str(CONFIG["MODEL"].get("task", "")).strip()
    extra_args: List[str] = []

    if _task_uses_fixed_early_outbreak_threshold(task_name):
        threshold_value = float(CONFIG["MODEL"].get("early_outbreak_fixed_threshold", 0.55))
        threshold_path = work_dir / "early_outbreak_warning_threshold.json"
        payload = {
            "h14_resistant_frac_threshold": float(threshold_value),
            "threshold": float(threshold_value),
            "task": str(task_name),
        }

        if not dry:
            threshold_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        if report is not None:
            report.write(
                f"EARLY_OUTBREAK_THRESHOLD file={threshold_path} value={float(threshold_value):.6f} task={task_name}"
            )

        extra_args += [
            "--early_res_frac_threshold", str(float(threshold_value)),
            "--early_res_frac_threshold_file", str(threshold_path),
            "--early_res_frac_threshold_out", str(threshold_path),
        ]

    return extra_args


def _run_convert_one_sim(*, py: str, sim_dir: Path, dry: bool, cwd: Path, report: Report) -> None:
    cmd: List[str] = [
        py,
        "convert_to_pt.py",
        "--graphml_dir", str(sim_dir),
        "--label_csv_dir", str(sim_dir / "labels"),
    ]
    cmd += _extra_args_from_env("DT_CONVERT_EXTRA_ARGS")
    cmd += _ensure_task_specific_convert_support_files(work_dir=cwd, dry=dry, report=report)
    run_cmd(cmd, dry=dry, cwd=cwd, report=report, stream_output=False)


def _collect_flat_pt_from_sim(*, sim_dir: Path, out_dir: Path, dry: bool, report: Optional[Report] = None) -> int:
    import torch

    pts = sorted(sim_dir.glob("*.pt"))
    if len(pts) == 0:
        raise RuntimeError(f"No .pt files found after conversion in {sim_dir.resolve()}")
    n = 0
    if dry:
        return len(pts)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_sim_id = _global_sim_id_from_sim_dir(sim_dir)

    for pt in pts:
        obj = torch.load(pt, map_location="cpu", weights_only=False)
        setattr(obj, "sim_id", global_sim_id)
        new_name = f"{sim_dir.parent.name}__{sim_dir.name}__{pt.name}"
        torch.save(obj, out_dir / new_name)
        n += 1

    if report is not None:
        report.write(f"FLATTEN_PT sim={sim_dir} copied={n} out={out_dir} global_sim_id={global_sim_id}")
    return n


def step2_convert_and_flatten_canonical_trajectories(
    *,
    work_dir: Path,
    py: str,
    dry: bool,
    report: Report,
    progress_start: int = 0,
    progress_total: Optional[int] = None,
    enable_graph_folder_figures: bool = False,
) -> List[Path]:
    out_dirs: List[Path] = []
    default_total_jobs = len(CANONICAL_TRAJECTORY_NAMES) * (int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2)
    total_jobs = int(progress_total) if progress_total is not None else default_total_jobs
    done = int(progress_start)

    for name in CANONICAL_TRAJECTORY_NAMES:
        traj_dir = work_dir / name
        if dry:
            out_dir = work_dir / _trajectory_pt_flat_dir(name)
            out_dirs.append(out_dir)
            report.write(f"[DRY_RUN] would convert+flatten trajectory {traj_dir} -> {out_dir}")
            done += int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2
            continue

        if not traj_dir.exists():
            raise FileNotFoundError(f"Missing Step 1 trajectory folder: {traj_dir.resolve()}")

        sim_dirs = _canonical_sim_dirs(work_dir, name)
        expected = int(CONFIG["STEP1"]["n_sims_per_trajectory"])
        if len(sim_dirs) != expected:
            raise RuntimeError(
                f"Trajectory {name} expected {expected} sim_* folders, found {len(sim_dirs)} in {traj_dir.resolve()}"
            )

        out_dir = work_dir / _trajectory_pt_flat_dir(name)
        out_dirs.append(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir, dry)

        total_pt = 0
        for sim_dir in sim_dirs:
            _report_progress(
                report,
                prefix="STEP2 progress",
                current=done + 1,
                total=total_jobs,
                extra=f"convert {name} / {sim_dir.name}",
                target="step",
            )
            _run_convert_one_sim(py=py, sim_dir=sim_dir, dry=dry, cwd=work_dir, report=report)
            total_pt += _collect_flat_pt_from_sim(sim_dir=sim_dir, out_dir=out_dir, dry=dry, report=report)
            done += 1

        _report_progress(
            report,
            prefix="STEP2 progress",
            current=done + 1,
            total=total_jobs,
            extra=f"figures {name}",
            target="step",
        )
        archive_single_dataset_figures(
            py=py,
            graph_dir=traj_dir,
            dst_dir=work_dir / "repro_artifacts_steps_1_9" / DATASET_FIGURES_REL / "step2_canonical" / _safe_tag(name),
            dry=dry,
            report=report,
            cwd=work_dir,
            identity="Harry Triantafyllidis",
            title=f"Step 2 canonical dataset: {name}",
            label=_safe_tag(name),
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
        done += 1

        _report_progress(
            report,
            prefix="STEP2 progress",
            current=done + 1,
            total=total_jobs,
            extra=f"purge {name}",
            target="step",
        )
        _purge_graphml_under(
            traj_dir,
            dry=dry,
            report=report,
            label=f"step2::{name}::after_dataset_figures",
        )
        done += 1

        if not dry:
            _assert_canonical_trajectory_expected_label_fraction(
                folder=out_dir,
                task_name=str(CONFIG["MODEL"]["task"]).strip(),
                trajectory_name=name,
                report=report,
            )

        report.write(f"STEP2_TRAJECTORY_DONE name={name} sims={len(sim_dirs)} total_pt={total_pt} out={out_dir}")

    return out_dirs

def _find_latest_dir(work_dir: Path, patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pat in patterns:
        for candidate in work_dir.glob(pat):
            if candidate.is_dir():
                candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(key=lambda pp: pp.stat().st_mtime, reverse=True)
    return candidates[0]


def _pool_pt_folders(
    *,
    src_folders: List[Path],
    dst_folder: Path,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would pool PT folders into {dst_folder}")
        return

    if dst_folder.exists():
        shutil.rmtree(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    all_pts: List[Tuple[Path, Path]] = []
    for src in src_folders:
        if not src.exists():
            raise FileNotFoundError(f"Missing source folder: {src.resolve()}")
        for pt_path in sorted(src.glob("*.pt")):
            all_pts.append((src, pt_path))

    total = len(all_pts)
    copied = 0

    if total > 0 and report is not None:
        report.start_subtask_console(f"_pool_pt_folders:{dst_folder.name}", total=total, extra=str(dst_folder))

    manifest = dst_folder / "manifest.csv"
    try:
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pooled_file", "source_folder", "source_file"])
            for src, pt_path in all_pts:
                actual_name = _copy_with_collision_guard(pt_path, dst_folder, source_hint=src)
                w.writerow([actual_name, src.as_posix(), pt_path.name])
                copied += 1
                if report is not None:
                    report.update_subtask_console(current=copied, total=total, extra=str(dst_folder))
    finally:
        if total > 0 and report is not None:
            report.finish_subtask_console(force_complete=True)

    if report is not None:
        report.write(f"POOLED_PT total={total} out={dst_folder}")

def _task_requires_two_class_pooled_train(task_name: str) -> bool:
    family = _task_family(task_name)
    if family == "early_outbreak_warning":
        return bool(CONFIG["STEP1"].get("outbreak_require_two_class_pooled_baseline", True))
    if family == "mechanism_split":
        return True
    return False


def _canonical_expected_label_for_trajectory(task_name: str, trajectory_name: str) -> Optional[int]:
    if _task_family(task_name) != "early_outbreak_warning":
        return None

    name = str(trajectory_name).strip().lower()
    if name.startswith("outbreak_low_"):
        return 0
    if name.startswith("outbreak_high_"):
        return 1
    return None


def _assert_canonical_trajectory_expected_label_fraction(
    *,
    folder: Path,
    task_name: str,
    trajectory_name: str,
    report: Optional[Report] = None,
) -> Dict[str, Any]:
    expected_label = _canonical_expected_label_for_trajectory(task_name, trajectory_name)
    preferred_label_attr = _preferred_label_attr_for_task(task_name)
    if expected_label is None or preferred_label_attr is None:
        return label_stats_for_pt_folder(folder, preferred_label_attr=preferred_label_attr, strict_preferred=True)

    stats = assert_nonempty_known_labels_folder(
        folder,
        name=f"Canonical trajectory {trajectory_name}",
        report=report,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=True,
    )

    n_expected = int(stats["n0"] if expected_label == 0 else stats["n1"])
    n_total = int(stats["n_total"])
    frac_expected = 0.0 if n_total <= 0 else float(n_expected) / float(n_total)
    min_frac = float(CONFIG["STEP1"].get("outbreak_expected_label_min_frac", 0.3))

    if report is not None:
        report.write(
            f"[Canonical trajectory {trajectory_name}] expected_label={expected_label} "
            f"n_total={n_total} n_expected={n_expected} frac_expected={frac_expected:.4f} "
            f"required_min_frac={min_frac:.4f}"
        )

    if frac_expected < min_frac:
        raise RuntimeError(
            f"[Canonical trajectory {trajectory_name}] expected outbreak label dominance not achieved. "
            f"Expected label {expected_label} fraction={frac_expected:.4f}, required >= {min_frac:.4f}. "
            "Tighten or further separate the Step 1 outbreak canonical regimes before training."
        )

    return stats


def _assert_label_ratio_not_extreme(
    stats: Dict[str, Any],
    *,
    name: str,
    max_majority_frac: float,
    report: Optional[Report] = None,
) -> None:
    """Fail fast when a binary dataset is technically two-class but too imbalanced.

    This is used for Step 8 after shifted train/test pooling, before expensive
    tuning/training starts. It prevents silently accepting shifted benchmarks
    that collapse to a dominant mechanism label.
    """
    n0 = int(stats.get("n0", 0))
    n1 = int(stats.get("n1", 0))
    total = n0 + n1

    if total <= 0:
        raise RuntimeError(f"[{name}] no known binary labels.")

    if n0 <= 0 or n1 <= 0:
        raise RuntimeError(
            f"[{name}] single-label dataset after generation: "
            f"n0={n0}, n1={n1}. Revise CONFIG settings before training."
        )

    max_allowed = float(max_majority_frac)
    if not (0.5 <= max_allowed <= 1.0):
        raise ValueError(
            f"[{name}] max_majority_frac must be in [0.5, 1.0], "
            f"got {max_allowed:.4f}."
        )

    majority_frac = max(n0, n1) / float(total)

    if report is not None:
        report.write(
            f"[{name}] label_balance_gate n0={n0} n1={n1} "
            f"majority_frac={majority_frac:.4f} "
            f"max_allowed={max_allowed:.4f}"
        )

    if majority_frac > max_allowed:
        raise RuntimeError(
            f"[{name}] label imbalance too extreme: "
            f"n0={n0}, n1={n1}, majority_frac={majority_frac:.4f}. "
            f"Maximum allowed is {max_allowed:.4f}. "
            "Revise the generating regimes before training."
        )


def _build_step8_shift_trajectories(num_days: int) -> Dict[str, Dict[str, Any]]:
    """Build the active Step 8 trajectory specification from CONFIG["STEP8"]."""
    step8_cfg = dict(CONFIG.get("STEP8", {}) or {})
    cfg_num_days = int(step8_cfg.get("num_days", num_days))
    if int(num_days) != int(cfg_num_days):
        num_days = cfg_num_days

    train_specs = dict(step8_cfg.get("train_trajectories", {}) or {})
    test_specs = dict(step8_cfg.get("test_trajectories", {}) or {})
    merged: Dict[str, Dict[str, Any]] = {}
    merged.update(train_specs)
    merged.update(test_specs)
    if not merged:
        raise RuntimeError("CONFIG['STEP8'] must define non-empty train_trajectories/test_trajectories")
    return merged


def generate_named_trajectories(
    *,
    work_dir: Path,
    py: str,
    dry: bool,
    report: Report,
    trajectory_specs: Dict[str, Dict[str, Any]],
    n_sims: int,
    num_days: int,
    progress_prefix: str,
) -> None:
    total_jobs = len(trajectory_specs) * int(n_sims)
    done = 0
    for name, spec in trajectory_specs.items():
        p_cs, p_cr, p_is, p_ir = _admission_import_weights_from_spec(spec)
        traj_dir = work_dir / name
        if not dry and traj_dir.exists():
            shutil.rmtree(traj_dir)
        ensure_dir(traj_dir, dry)
        for r in range(int(n_sims)):
            sim_dir = traj_dir / f"sim_{r:03d}"
            ensure_dir(sim_dir, dry)
            _report_progress(report, prefix=progress_prefix, current=done + 1, total=total_jobs, extra=f"{name} / sim_{r:03d}", target="step")
            _run_generate_amr_sim(
                py=py,
                out_dir=sim_dir,
                seed=int(spec["seed_base"]) + r,
                num_days=int(num_days),
                p_cs=p_cs,
                p_cr=p_cr,
                p_is=p_is,
                p_ir=p_ir,
                discharge_frac=float(spec["daily_discharge_frac"]),
                discharge_min=int(spec["daily_discharge_min_per_ward"]),
                extra_cli_args=list(spec.get("extra_sim_args", [])),
                dry=dry,
                cwd=work_dir,
                report=report,
            )
            done += 1


def convert_and_flatten_named_trajectories(
    *,
    work_dir: Path,
    py: str,
    dry: bool,
    report: Report,
    trajectory_names: List[str],
    progress_prefix: str,
) -> List[Path]:
    out_dirs: List[Path] = []
    total_jobs = 0
    for name in trajectory_names:
        total_jobs += len(_canonical_sim_dirs(work_dir, name))
    done = 0
    for name in trajectory_names:
        traj_dir = work_dir / name
        out_dir = work_dir / _trajectory_pt_flat_dir(name)
        out_dirs.append(out_dir)
        if dry:
            report.write(f"[DRY_RUN] would convert+flatten trajectory {traj_dir} -> {out_dir}")
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir, dry)
        for sim_dir in _canonical_sim_dirs(work_dir, name):
            _report_progress(report, prefix=progress_prefix, current=done + 1, total=max(1, total_jobs), extra=f"{name} / {sim_dir.name}", target="step")
            _run_convert_one_sim(py=py, sim_dir=sim_dir, dry=dry, cwd=work_dir, report=report)
            _collect_flat_pt_from_sim(sim_dir=sim_dir, out_dir=out_dir, dry=dry, report=report)
            done += 1
    return out_dirs


def archive_raw_pair_figures_from_trajectory_groups(
    *,
    py: str,
    work_dir: Path,
    archive_root: Path,
    dry: bool,
    report: Optional[Report],
    cwd: Optional[Path],
    identity: str,
    enable_graph_folder_figures: bool,
    stage_tag: str,
    title: str,
    train_names: List[str],
    test_names: List[str],
) -> Optional[Path]:
    train_sources = [work_dir / name for name in train_names]
    test_sources = [work_dir / name for name in test_names]
    if not all(_has_graphml_files(p) for p in train_sources + test_sources):
        return None
    tmp_root = work_dir / f"_tmp_{_safe_tag(stage_tag)}_graph_compare"
    train_pool = tmp_root / "train_raw"
    test_pool = tmp_root / "test_raw"
    out_dir = archive_root / DATASET_FIGURES_REL / _safe_tag(stage_tag)
    try:
        _prepare_graph_folder_figures_compare_pool(src_roots=train_sources, dst_root=train_pool, dry=dry, report=report)
        _prepare_graph_folder_figures_compare_pool(src_roots=test_sources, dst_root=test_pool, dry=dry, report=report)
        archive_dataset_pair_figures(
            py=py,
            graph_dir=train_pool,
            compare_dir=test_pool,
            dst_dir=out_dir,
            dry=dry,
            report=report,
            cwd=cwd,
            identity=identity,
            title=title,
            label="train",
            compare_label="test",
            enable_graph_folder_figures=enable_graph_folder_figures,
        )
    finally:
        if not dry and tmp_root.exists():
            _safe_rmtree(tmp_root)
    return out_dir


def build_pooled_train_test_from_trajectories(
    *,
    work_dir: Path,
    train_names: List[str],
    test_names: List[str],
    train_folder: Path,
    test_folder: Path,
    dry: bool,
    report: Report,
) -> None:
    train_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in train_names]
    test_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in test_names]
    _pool_pt_folders(src_folders=train_srcs, dst_folder=train_folder, dry=dry, report=report)
    _pool_pt_folders(src_folders=test_srcs, dst_folder=test_folder, dry=dry, report=report)


def build_frozen_baseline_from_canonical_trajectories(
    *,
    work_dir: Path,
    train_folder: Path,
    test_folder: Path,
    dry: bool,
    report: Report,
    preferred_label_attr: Optional[str] = None,
) -> None:
    if preferred_label_attr is None:
        raise ValueError(
            f"No supported preferred label attribute could be inferred for task '{CONFIG['MODEL']['task']}'. "
            "Refusing to build baseline with ambiguous labels."
        )

    train_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in CANONICAL_TRAIN_TRAJECTORIES]
    test_srcs = [work_dir / _trajectory_pt_flat_dir(name) for name in CANONICAL_TEST_TRAJECTORIES]
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))

    report.section("BUILD FROZEN BASELINE FROM CANONICAL TRAJECTORIES")
    for src in train_srcs + test_srcs:
        report.kv(f"source::{src.name}", src)
        if not dry and not src.exists():
            raise FileNotFoundError(f"Missing Step 2 PT-flat folder: {src.resolve()}")

    _pool_pt_folders(src_folders=train_srcs, dst_folder=train_folder, dry=dry, report=report)
    _pool_pt_folders(src_folders=test_srcs, dst_folder=test_folder, dry=dry, report=report)

    if dry:
        return

    if _task_requires_two_class_pooled_train(str(CONFIG["MODEL"]["task"]).strip()):
        st_train = assert_two_class_folder(
            train_folder,
            name="Frozen BASELINE TRAIN",
            require_balanced=False,
            balance_tolerance=int(CONFIG["TEST"].get("balance_tolerance", 1)),
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
    else:
        st_train = assert_nonempty_known_labels_folder(
            train_folder,
            name="Frozen BASELINE TRAIN",
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        warn_if_single_label(st_train, name="Frozen BASELINE TRAIN", report=report)
    assert_contiguous_days_per_sim(train_folder, name="Frozen BASELINE TRAIN", report=report)

    assert_two_class_folder(
        test_folder,
        name="Frozen BASELINE TEST",
        require_balanced=require_balanced_test,
        balance_tolerance=int(CONFIG["TEST"].get("balance_tolerance", 1)),
        report=report,
        preferred_label_attr=preferred_label_attr,
        strict_preferred=True,
    )
    assert_contiguous_days_per_sim(test_folder, name="Frozen BASELINE TEST", report=report)


def _restore_live_test_from_baseline(
    *,
    work_dir: Path,
    archive_root: Optional[Path] = None,
    dry: bool,
    report: Optional[Report] = None,
) -> None:
    src = work_dir / BASELINE_TEST_REL
    dst = work_dir / LIVE_TEST_REL

    if dry:
        if report is not None:
            report.write(f"[DRY_RUN] would mirror baseline test {src} -> {dst}")
        return

    if not src.exists():
        fallback_src = None
        if archive_root is not None:
            fallback_src = _resolve_existing_dataset_dir(
                requested_folder=src,
                work_dir=work_dir,
                archive_root=archive_root,
                step_tag="step4_baseline",
                split_name="test_folder",
                report=report,
                fallback_step_tags=[],
            )
        if fallback_src is None or not _dir_has_pt_files(fallback_src):
            raise FileNotFoundError(f"Missing baseline test folder: {src.resolve()}")
        src = fallback_src

    if src.resolve() == dst.resolve():
        if report is not None:
            report.write(f"LIVE TEST already equals frozen baseline test: {src}")
        return

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    if report is not None:
        report.write(f"RESTORED live test folder from baseline: {src} -> {dst}")


def _require_pipeline_prereqs(
    *,
    work_dir: Path,
    start_v: float,
    report: Report,
) -> None:
    if start_v <= 1.0:
        return

    checks: List[Tuple[str, Path]] = []
    if start_v > 1.0:
        checks.extend([(name, work_dir / name) for name in CANONICAL_TRAJECTORY_NAMES])
    if start_v > 2.0:
        checks.extend([(f"{name}_pt_flat", work_dir / _trajectory_pt_flat_dir(name)) for name in CANONICAL_TRAJECTORY_NAMES])
    if start_v > 3.0:
        checks.extend([
            (str(BASELINE_TRAIN_REL), work_dir / BASELINE_TRAIN_REL),
            (str(BASELINE_TEST_REL), work_dir / BASELINE_TEST_REL),
        ])

    missing = [(label, p) for label, p in checks if not p.exists()]
    if missing:
        msg_lines = [
            f"Cannot start at step {start_v} because prerequisite artifacts are missing from this persistent work_dir.",
            "Missing prerequisites:",
        ]
        msg_lines.extend([f"- {label}: {p}" for label, p in missing[:20]])
        msg_lines.append("Run earlier steps first in the same experiments_results root, or restart from step 1 for this track.")
        msg = "\n".join(msg_lines)
        report.write(msg)
        raise FileNotFoundError(msg)


# =============================================================================
# Sanity / runtime consistency helpers
# =============================================================================

def _validate_step1_length_compatibility(
    *,
    report: Report,
    horizons: List[int],
    T_list: List[int],
) -> None:
    num_days = int(CONFIG["STEP1"]["num_days"])
    max_h = max(int(h) for h in horizons) if horizons else int(CONFIG["MODEL"].get("pred_horizon", 7))
    max_t = max(int(t) for t in T_list) if T_list else int(CONFIG["MODEL"].get("T", 7))

    required_min_days = max_t + max_h
    report.kv("step1_num_days", num_days)
    report.kv("max_requested_T", max_t)
    report.kv("max_requested_horizon", max_h)
    report.kv("required_min_days_sanity", required_min_days)

    if num_days < required_min_days:
        raise RuntimeError(
            "STEP1 num_days is too small for the requested runtime configuration. "
            f"Got num_days={num_days}, but max(T)+max(horizon)={required_min_days}. "
            "Increase CONFIG['STEP1']['num_days'] or reduce T / horizons."
        )


# =============================================================================
# Step-specific folder validators
# =============================================================================

def validate_train_folder_for_task(
    *,
    folder: Path,
    task_name: str,
    folder_name: str,
    report: Report,
) -> Dict[str, Any]:
    preferred = _preferred_label_attr_for_task(task_name)
    if preferred is None:
        raise ValueError(f"Unsupported task for strict TRAIN validation: {task_name}")

    stats = report_label_stats(
        report,
        folder=folder,
        name=f"{folder_name} [{task_name}]",
        preferred_label_attr=preferred,
        strict_preferred=True,
    )
    stats = assert_nonempty_known_labels_folder(
        folder,
        name=f"{folder_name} [{task_name}]",
        report=report,
        preferred_label_attr=preferred,
        strict_preferred=True,
    )
    warn_if_single_label(stats, name=f"{folder_name} [{task_name}]", report=report)
    assert_contiguous_days_per_sim(
        folder,
        name=f"{folder_name} [{task_name}] (contiguity gate)",
        report=report,
    )
    return stats


def _make_baseline_test_checker(*, work_dir: Path, archive_root: Path, tol: int, report: Report) -> Callable[[str], None]:
    return _make_test_folder_checker(
        work_dir=work_dir,
        archive_root=archive_root,
        requested_folder=LIVE_TEST_REL,
        archive_step_tags=["step4_baseline"],
        tol=tol,
        report=report,
    )


def _validate_baseline_for_all_requested_horizons(
    *,
    work_dir: Path,
    archive_root: Path,
    base_task: str,
    horizons: List[int],
    run_all_horizons: bool,
    report: Report,
    tol: int,
) -> None:
    test_folder = _resolve_existing_dataset_dir(
        requested_folder=BASELINE_TEST_REL,
        work_dir=work_dir,
        archive_root=archive_root,
        step_tag="step4_baseline",
        split_name="test_folder",
        report=report,
        fallback_step_tags=[],
    )
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))

    if not run_all_horizons:
        task_eff = base_task
        preferred = _preferred_label_attr_for_task(task_eff)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict baseline validation: {task_eff}")
        assert_two_class_folder(
            test_folder,
            name=f"Baseline TEST [{task_eff}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            test_folder,
            name=f"Baseline TEST [{task_eff}] (contiguity gate)",
            report=report,
        )
        return

    for h_val in horizons:
        task_eff = _task_with_horizon(base_task, int(h_val))
        preferred = _preferred_label_attr_for_task(task_eff)
        if preferred is None:
            raise ValueError(f"Unsupported task for strict baseline validation: {task_eff}")
        assert_two_class_folder(
            test_folder,
            name=f"Baseline TEST [{task_eff}]",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(
            test_folder,
            name=f"Baseline TEST [{task_eff}] (contiguity gate)",
            report=report,
        )


# =============================================================================
# Multi-(T,H) training orchestrator
# =============================================================================


def run_training_step(
    *,
    step_tag: str,
    data_folder: Path,
    work_dir: Path,
    archive_root: Path,
    py: str,
    dry: bool,
    report: Report,
    horizons: List[int],
    run_all_horizons: bool,
    T_list: List[int],
    run_all_T: bool,
    archive_train_test_folders: bool,
    ensure_test_folder_for_task: Optional[Callable[[str], None]] = None,
    eval_test_folder: Path = LIVE_TEST_REL,
    test_folder_for_archive: Path = LIVE_TEST_REL,
    no_train: bool = False,
    pretrained_model_path_override: Optional[Any] = None,
    archive_train_step_tags: Optional[List[str]] = None,
    archive_eval_test_step_tags: Optional[List[str]] = None,
    archive_test_folder_step_tags: Optional[List[str]] = None,
    model_cfg_override: Optional[Any] = None,
) -> None:
    tag = _safe_tag(step_tag)

    active_model_cfg = dict(CONFIG["MODEL"])
    if isinstance(model_cfg_override, dict) and not any(isinstance(k, tuple) for k in model_cfg_override.keys()):
        active_model_cfg = dict(model_cfg_override)

    if run_all_T:
        if len(T_list) == 0:
            raise RuntimeError("run_all_T enabled but T_list is empty.")
        ts_vals = [int(t) for t in T_list]
    else:
        ts_vals = [int(active_model_cfg["T"])]

    if run_all_horizons:
        if len(horizons) == 0:
            raise RuntimeError("run_all_horizons enabled but horizons list is empty.")
        hs_vals = [int(h) for h in horizons]
    else:
        hs_vals = []
    base_task = str(active_model_cfg["task"]).strip()
    total_jobs = len(ts_vals) * (len(hs_vals) if run_all_horizons else 1)
    done = 0

    report.start_subtask_console(f"run_training_step:{tag}", total=total_jobs, extra=str(data_folder))

    try:
        for t_val in ts_vals:
            horizon_iter: List[Optional[int]] = hs_vals if run_all_horizons else [None]

            for h_val in horizon_iter:
                task_eff = _task_with_horizon(base_task, int(h_val)) if h_val is not None else base_task

                dst = _archive_dst_for_training_run(
                    archive_root=archive_root,
                    step_tag=tag,
                    t_val=int(t_val),
                    h_val=h_val,
                    run_all_T=run_all_T,
                    run_all_horizons=run_all_horizons,
                )

                resolved_data_folder_abs = _resolve_existing_dataset_dir(
                    requested_folder=data_folder,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="train_folder",
                    report=report if (no_train or not ((data_folder if data_folder.is_absolute() else (work_dir / data_folder)).exists())) else None,
                    preferred_archive_dir=dst,
                    fallback_step_tags=list(archive_train_step_tags or []),
                )
                resolved_eval_test_folder_abs = _resolve_existing_dataset_dir(
                    requested_folder=eval_test_folder,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="test_folder",
                    report=report if (no_train or not ((eval_test_folder if eval_test_folder.is_absolute() else (work_dir / eval_test_folder)).exists())) else None,
                    preferred_archive_dir=dst,
                    fallback_step_tags=list(archive_eval_test_step_tags or []),
                )
                resolved_test_folder_for_archive_abs = _resolve_existing_dataset_dir(
                    requested_folder=test_folder_for_archive,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="test_folder",
                    report=report if not ((test_folder_for_archive if test_folder_for_archive.is_absolute() else (work_dir / test_folder_for_archive)).exists()) else None,
                    preferred_archive_dir=dst,
                    fallback_step_tags=list(archive_test_folder_step_tags or []),
                )

                if no_train:
                    if not _dir_has_pt_files(resolved_data_folder_abs):
                        raise FileNotFoundError(
                            f"No-train repair could not find train folder for step '{tag}': {resolved_data_folder_abs.resolve()}"
                        )
                    out_dir_rel = dst.relative_to(work_dir)
                else:
                    if h_val is None:
                        out_dir_rel = Path(f"training_outputs_{tag}_T{int(t_val)}") if run_all_T else Path(TRAINING_OUT_REL)
                    else:
                        out_dir_rel = Path(f"training_outputs_{tag}_T{int(t_val)}_h{int(h_val)}")

                extra_bits = [f"T={int(t_val)}"]
                if h_val is not None:
                    extra_bits.append(f"h={int(h_val)}")
                extra_bits.append("eval_only" if no_train else "train_eval")
                _report_progress(
                    report,
                    prefix=f"TRAIN {tag}",
                    current=done + 1,
                    total=total_jobs,
                    extra=" ".join(extra_bits),
                    target="subtask",
                )

                if not dry:
                    validate_train_folder_for_task(
                        folder=resolved_data_folder_abs,
                        task_name=task_eff,
                        folder_name=f"TRAIN CHECK {tag}",
                        report=report,
                    )

                if ensure_test_folder_for_task is not None:
                    ensure_test_folder_for_task(task_eff)

                resolved_pretrained_model_path: Optional[Path] = None
                if no_train:
                    if callable(pretrained_model_path_override):
                        resolved_pretrained_model_path = Path(pretrained_model_path_override(int(t_val), h_val))
                    elif pretrained_model_path_override is not None:
                        resolved_pretrained_model_path = Path(pretrained_model_path_override)
                    else:
                        resolved_pretrained_model_path = dst / "trained_model.pt"

                if no_train and not dry:
                    model_path = resolved_pretrained_model_path if resolved_pretrained_model_path is not None else (dst / "trained_model.pt")
                    model_parent = model_path.parent
                    if not model_parent.exists():
                        raise FileNotFoundError(
                            f"Evaluation-only repair requested, but checkpoint parent folder is missing: {model_parent.resolve()}"
                        )
                    if not model_path.exists():
                        raise FileNotFoundError(
                            f"Evaluation-only repair requested, but trained_model.pt is missing in: {model_path.resolve()}"
                        )

                    dst.mkdir(parents=True, exist_ok=True)
                    dst_model_path = dst / "trained_model.pt"
                    if model_path.resolve() != dst_model_path.resolve():
                        shutil.copy2(model_path, dst_model_path)
                        if report is not None:
                            report.write(
                                f"EVAL_ONLY_CHECKPOINT_STAGED src={model_path} dst={dst_model_path} step={tag}"
                            )

                combo_model_cfg = active_model_cfg
                if callable(model_cfg_override):
                    maybe_cfg = model_cfg_override(int(t_val), h_val)
                    if isinstance(maybe_cfg, dict):
                        combo_model_cfg = dict(maybe_cfg)
                elif isinstance(model_cfg_override, dict) and all(isinstance(k, tuple) for k in model_cfg_override.keys()):
                    maybe_cfg = model_cfg_override.get((int(t_val), h_val))
                    if isinstance(maybe_cfg, dict):
                        combo_model_cfg = dict(maybe_cfg)

                train_args = _build_train_args_from_config(
                    combo_model_cfg,
                    py,
                    data_folder=resolved_data_folder_abs,
                    work_dir=work_dir,
                    task_override=task_eff,
                    out_dir=str(out_dir_rel),
                    T_override=int(t_val),
                    test_folder=resolved_eval_test_folder_abs,
                    train_model_override=(False if no_train else None),
                )
                run_cmd(train_args, dry, cwd=work_dir, report=report, stream_output=True)

                if not no_train:
                    archive_training_outputs(dst, work_dir=work_dir, dry=dry, report=report, src_dir=out_dir_rel)

                    if archive_train_test_folders:
                        archive_dataset_folder(
                            dst / "train_folder",
                            work_dir=work_dir,
                            folder=resolved_data_folder_abs,
                            dry=dry,
                            report=report,
                            label="train_folder",
                        )
                        archive_dataset_folder(
                            dst / "test_folder",
                            work_dir=work_dir,
                            folder=resolved_test_folder_for_archive_abs,
                            dry=dry,
                            report=report,
                            label="test_folder",
                        )

                done += 1
    finally:
        report.finish_subtask_console(force_complete=True)

# =============================================================================
# Pipeline
# =============================================================================

def run_pipeline_once(
    *,
    work_dir: Path,
    archive_root: Path,
    graphml_keep_root: Path,
    py: str,
    dry: bool,
    start_v: float,
    stop_v: float,
    archive_train_test_folders: bool,
    test_frac_per_class: float,
    report: Report,
    horizons: List[int],
    run_all_horizons: bool,
    T_list: List[int],
    run_all_T: bool,
    enable_graph_folder_figures: bool,
    keep_step_train_graphml: bool,
    no_train: bool,
    no_simulation: bool,
    tune_step4: bool,
    tune_step8: bool,
    tune_trials_quick: int,
    tune_finalists: int,
    tune_quick_epochs: int,
    tune_full_epochs: int,
    tune_split_seed: int,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> None:
    del test_frac_per_class
    ensure_dir(archive_root, dry)

    _require_pipeline_prereqs(work_dir=work_dir, start_v=start_v, report=report)

    tol = int(CONFIG["TEST"].get("balance_tolerance", 1))
    require_balanced_test = bool(CONFIG["TEST"].get("require_balanced_test", False))
    base_task = str(CONFIG["MODEL"]["task"]).strip()
    preferred_label_attr = _preferred_label_attr_for_task(base_task)
    if preferred_label_attr is None:
        raise ValueError(f"Unsupported task for strict experiments pipeline: {base_task}")
    t_needed_runtime = _get_max_T_needed_runtime(T_list)

    report.section("PIPELINE SETTINGS")
    report.kv("T_needed_runtime_for_baseline", t_needed_runtime)
    report.kv("keep_step_train_graphml", int(bool(keep_step_train_graphml)))
    report.kv("graphml_keep_root", graphml_keep_root)
    report.kv("no_train", int(bool(no_train)))
    report.kv("no_simulation", int(bool(no_simulation)))
    skip_data_generation = bool(no_train or no_simulation)
    report.kv("skip_data_generation", int(skip_data_generation))
    planned_parallel_steps = set(_planned_parallel_step_keys(start_v, stop_v))

    step4_shared_tuned_cfg_map: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
    runtime_t_values = [int(t) for t in (T_list if run_all_T else [int(CONFIG["MODEL"]["T"])])]
    runtime_h_values: List[Optional[int]] = [int(h) for h in horizons] if run_all_horizons else [None]

    def _ensure_step4_shared_tuned_cfg_map() -> Dict[Tuple[int, Optional[int]], Dict[str, Any]]:
        nonlocal step4_shared_tuned_cfg_map
        if step4_shared_tuned_cfg_map:
            return step4_shared_tuned_cfg_map
        step4_shared_tuned_cfg_map = _load_archived_tuned_cfg_map(
            step_tag="step4_baseline",
            work_dir=work_dir,
            archive_root=archive_root,
            report=report,
            t_values=runtime_t_values,
            h_values=runtime_h_values,
        )
        return step4_shared_tuned_cfg_map
      
    def _step4_model_cfg_for_combo(t_val: int, h_val: Optional[int]) -> Dict[str, Any]:
        cfg_map = _ensure_step4_shared_tuned_cfg_map()
        cfg = cfg_map.get((int(t_val), h_val)) if isinstance(cfg_map, dict) else None
        if isinstance(cfg, dict) and cfg:
            return dict(cfg)
        return dict(CONFIG["MODEL"])

    def _progress_step_started(active_step: str) -> None:
        if progress_callback is not None:
            progress_callback("started", str(active_step))

    def _progress_step_completed(step_key: str) -> None:
        if progress_callback is not None and str(step_key) in planned_parallel_steps:
            progress_callback("completed", str(step_key))

    def _progress_step_stage(step_label: str, stage_label: str) -> None:
        if progress_callback is not None:
            progress_callback("started", f"{str(step_label).strip()} | {str(stage_label).strip()}")

    _validate_step1_length_compatibility(report=report, horizons=horizons, T_list=T_list)

    if start_v <= 1.0 <= stop_v:
        _progress_step_started("STEP 1")
        _progress_step_stage("STEP 1", "data generation")
        step1_total = len(CANONICAL_TRAJECTORY_NAMES) * int(CONFIG["STEP1"]["n_sims_per_trajectory"])
        report.section("STEP 1", total=step1_total)
        if skip_data_generation:
            _report_progress(
                report,
                prefix="STEP1 progress",
                current=step1_total,
                total=step1_total,
                extra="skip generation and reuse existing canonical trajectories",
                target="step",
            )
        else:
            step1_generate_canonical_trajectories(work_dir=work_dir, py=py, dry=dry, report=report)
        report.finish_step_console(force_complete=True)
        _progress_step_completed("1")

    if start_v <= 2.0 <= stop_v:
        _progress_step_started("STEP 2")
        _progress_step_stage("STEP 2", "conversion and flattening")
        step2_total = 1 + len(CANONICAL_TRAJECTORY_NAMES) * (int(CONFIG["STEP1"]["n_sims_per_trajectory"]) + 2)
        report.section("STEP 2", total=step2_total)

        if skip_data_generation:
            _report_progress(
                report,
                prefix="STEP2 progress",
                current=step2_total,
                total=step2_total,
                extra="skip conversion/collection and reuse existing canonical PT folders",
                target="step",
            )
            if keep_step_train_graphml:
                keep_step1_baseline_graphml_before_purge(
                    work_dir=work_dir,
                    graphml_keep_root=graphml_keep_root,
                    dry=dry,
                    report=report,
                )
        else:
            _report_progress(
                report,
                prefix="STEP2 progress",
                current=1,
                total=step2_total,
                extra="baseline raw figures",
                target="step",
            )

            ensure_step1_baseline_pair_figures_before_purge(
                py=py,
                work_dir=work_dir,
                archive_root=archive_root,
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml:
                keep_step1_baseline_graphml_before_purge(
                    work_dir=work_dir,
                    graphml_keep_root=graphml_keep_root,
                    dry=dry,
                    report=report,
                )

            step2_convert_and_flatten_canonical_trajectories(
                work_dir=work_dir,
                py=py,
                dry=dry,
                report=report,
                progress_start=1,
                progress_total=step2_total,
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("2")

    if start_v <= 3.0 <= stop_v:
        _progress_step_started("STEP 3")
        _progress_step_stage("STEP 3", "pooling and validation")
        report.section("STEP 3", total=4)

        _report_progress(report, prefix="STEP3 progress", current=1, total=4, extra="pool canonical PT folders", target="step")
        build_frozen_baseline_from_canonical_trajectories(
            work_dir=work_dir,
            train_folder=work_dir / BASELINE_TRAIN_REL,
            test_folder=work_dir / BASELINE_TEST_REL,
            dry=dry,
            report=report,
            preferred_label_attr=preferred_label_attr,
        )

        _report_progress(report, prefix="STEP3 progress", current=2, total=4, extra="validate frozen baseline", target="step")
        _validate_baseline_for_all_requested_horizons(
            work_dir=work_dir,
            archive_root=archive_root,
            base_task=base_task,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            report=report,
            tol=tol,
        )

        _report_progress(report, prefix="STEP3 progress", current=3, total=4, extra="baseline ready", target="step")
        _report_progress(report, prefix="STEP3 progress", current=4, total=4, extra="done", target="step")
        report.finish_step_console(force_complete=True)
        _progress_step_completed("3")

    if start_v <= 4.0 <= stop_v:
        _progress_step_started("STEP 4")
        _progress_step_stage("STEP 4", "preflight")
        report.section("STEP 4", total=(3 if tune_step4 and not no_train else 2))

        _report_progress(report, prefix="STEP4 progress", current=1, total=2, extra="preflight", target="step")
        _restore_live_test_from_baseline(work_dir=work_dir, archive_root=archive_root, dry=dry, report=report)

        step4_train_folder = _resolve_existing_dataset_dir(
            requested_folder=BASELINE_TRAIN_REL,
            work_dir=work_dir,
            archive_root=archive_root,
            step_tag="step4_baseline",
            split_name="train_folder",
            report=report,
            fallback_step_tags=[],
        )
        step4_test_folder = _resolve_existing_dataset_dir(
            requested_folder=BASELINE_TEST_REL,
            work_dir=work_dir,
            archive_root=archive_root,
            step_tag="step4_baseline",
            split_name="test_folder",
            report=report,
            fallback_step_tags=[],
        )

        report_label_stats(
            report,
            folder=step4_train_folder,
            name="Step4 BASELINE TRAIN",
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        report_label_stats(
            report,
            folder=step4_test_folder,
            name="Step4 BASELINE TEST",
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )

        if _task_requires_two_class_pooled_train(base_task):
            st_train = assert_two_class_folder(
                step4_train_folder,
                name="Step4 BASELINE TRAIN",
                require_balanced=False,
                balance_tolerance=tol,
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
        else:
            st_train = assert_nonempty_known_labels_folder(
                step4_train_folder,
                name="Step4 BASELINE TRAIN",
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
            warn_if_single_label(st_train, name="Step4 BASELINE TRAIN", report=report)
        assert_contiguous_days_per_sim(step4_train_folder, name="Step4 BASELINE TRAIN", report=report)

        _validate_baseline_for_all_requested_horizons(
            work_dir=work_dir,
            archive_root=archive_root,
            base_task=base_task,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            report=report,
            tol=tol,
        )

        assert_two_class_folder(
            _resolve_existing_dataset_dir(requested_folder=LIVE_TEST_REL, work_dir=work_dir, archive_root=archive_root, step_tag="step4_baseline", split_name="test_folder", report=report, fallback_step_tags=[]),
            name="Step4 LIVE TEST",
            require_balanced=require_balanced_test,
            balance_tolerance=tol,
            report=report,
            preferred_label_attr=preferred_label_attr,
            strict_preferred=True,
        )
        assert_contiguous_days_per_sim(_resolve_existing_dataset_dir(requested_folder=LIVE_TEST_REL, work_dir=work_dir, archive_root=archive_root, step_tag="step4_baseline", split_name="test_folder", report=report, fallback_step_tags=[]), name="Step4 LIVE TEST", report=report)

        step4_tuned_cfg_map: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
        if tune_step4 and not dry and not no_train:
            _progress_step_stage("STEP 4", "hyperparameter tuning")
            _report_progress(report, prefix="STEP4 progress", current=2, total=3, extra="hyperparameter tuning", target="step")
            step4_tuned_cfg_map = _run_lightweight_tuning(
                step_tag="step4_baseline",
                work_dir=work_dir,
                archive_root=archive_root,
                py=py,
                dry=dry,
                report=report,
                data_folder=step4_train_folder,
                task_name=base_task,
                t_values=runtime_t_values,
                h_values=runtime_h_values,
                tune_trials_quick=tune_trials_quick,
                tune_finalists=tune_finalists,
                tune_quick_epochs=tune_quick_epochs,
                tune_full_epochs=tune_full_epochs,
                tune_split_seed=tune_split_seed,
            )
            step4_shared_tuned_cfg_map = dict(step4_tuned_cfg_map)
        _progress_step_stage("STEP 4", "training and evaluation")
        _report_progress(report, prefix="STEP4 progress", current=3 if tune_step4 and not no_train else 2, total=3 if tune_step4 and not no_train else 2, extra="train/evaluate", target="step")
        run_training_step(
            step_tag="step4_baseline",
            data_folder=BASELINE_TRAIN_REL,
            work_dir=work_dir,
            archive_root=archive_root,
            py=py,
            dry=dry,
            report=report,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            T_list=T_list,
            run_all_T=run_all_T,
            archive_train_test_folders=archive_train_test_folders,
            ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, archive_root=archive_root, tol=tol, report=report),
            eval_test_folder=LIVE_TEST_REL,
            test_folder_for_archive=BASELINE_TEST_REL,
            no_train=no_train,
            archive_train_step_tags=["step4_baseline"],
            archive_eval_test_step_tags=["step4_baseline"],
            archive_test_folder_step_tags=["step4_baseline"],
            model_cfg_override=(step4_tuned_cfg_map if step4_tuned_cfg_map else None),
        )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("4")

    if stop_v >= 5.0 and not step4_shared_tuned_cfg_map:
        _ensure_step4_shared_tuned_cfg_map()

    if start_v <= 5.0 <= stop_v:
        _progress_step_started("STEP 5")
        _progress_step_stage("STEP 5", "ablation preparation")
        step5_folders = [
            ("no_edge_weights", "step5_no_edge_weights"),
            ("core_node_features_only", "step5_core_node_features_only"),
        ]
        step5_total = 2 + len(step5_folders)
        report.section("STEP 5", total=step5_total)

        _report_progress(report, prefix="STEP5 progress", current=1, total=step5_total, extra="restore baseline test", target="step")
        _restore_live_test_from_baseline(work_dir=work_dir, archive_root=archive_root, dry=dry, report=report)

        step5_root = work_dir / STEP5_ABLATION_REL
        step5_required_dirs = [work_dir / STEP5_ABLATION_REL / ab_folder for ab_folder, _ in step5_folders]
        missing_step5_dirs = [p for p in step5_required_dirs if not p.exists()]
        _report_progress(
            report,
            prefix="STEP5 progress",
            current=2,
            total=step5_total,
            extra=("reuse ablations" if skip_data_generation else "build ablations"),
            target="step",
        )
        if skip_data_generation:
            if missing_step5_dirs:
                report.write(
                    "STEP5 reuse will resolve missing live ablation folders from archived train_folder copies where available: "
                    + ", ".join(str(p) for p in missing_step5_dirs)
                )
        else:
            if not dry:
                (work_dir / STEP5_ABLATION_REL / "no_edge_weights").mkdir(parents=True, exist_ok=True)
                (work_dir / STEP5_ABLATION_REL / "core_node_features_only").mkdir(parents=True, exist_ok=True)
            run_cmd([py, "ablate_edge_weights.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "ablate_node_features.py"], dry, cwd=work_dir, report=report, stream_output=False)

        if step5_root.exists() or skip_data_generation:
            base_idx = 2
            for idx, (ab_folder, tag) in enumerate(step5_folders, start=1):
                _report_progress(
                    report,
                    prefix="STEP5 progress",
                    current=base_idx + idx,
                    total=step5_total,
                    extra=ab_folder,
                    target="step",
                )

                data_folder = STEP5_ABLATION_REL / ab_folder
                resolved_step5_train = _resolve_existing_dataset_dir(
                    requested_folder=data_folder,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="train_folder",
                    report=report if (skip_data_generation or not (work_dir / data_folder).exists()) else None,
                    fallback_step_tags=[],
                )
                if not _dir_has_pt_files(resolved_step5_train):
                    raise FileNotFoundError(
                        f"Step 5 requested in reuse mode, but no live or archived train folder could be resolved for {tag}: {resolved_step5_train}"
                    )

                report_label_stats(
                    report,
                    folder=resolved_step5_train,
                    name=f"{tag} TRAIN",
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                st = assert_nonempty_known_labels_folder(
                    resolved_step5_train,
                    name=f"{tag} TRAIN",
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                assert_contiguous_days_per_sim(resolved_step5_train, name=f"{tag} TRAIN", report=report)

                step5_test_folder = _resolve_existing_dataset_dir(
                    requested_folder=LIVE_TEST_REL,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="test_folder",
                    report=report if skip_data_generation else None,
                    fallback_step_tags=["step4_baseline"],
                )
                assert_two_class_folder(
                    step5_test_folder,
                    name=f"{tag} TEST",
                    require_balanced=require_balanced_test,
                    balance_tolerance=tol,
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                assert_contiguous_days_per_sim(step5_test_folder, name=f"{tag} TEST", report=report)

                run_training_step(
                    step_tag=tag,
                    data_folder=resolved_step5_train,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    py=py,
                    dry=dry,
                    report=report,
                    horizons=horizons,
                    run_all_horizons=run_all_horizons,
                    T_list=T_list,
                    run_all_T=run_all_T,
                    archive_train_test_folders=archive_train_test_folders,
                    ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, archive_root=archive_root, tol=tol, report=report),
                    test_folder_for_archive=BASELINE_TEST_REL,
                    no_train=no_train,
                    archive_train_step_tags=[tag],
                    archive_eval_test_step_tags=[tag, "step4_baseline"],
                    archive_test_folder_step_tags=[tag, "step4_baseline"],
                    model_cfg_override=(step4_shared_tuned_cfg_map if step4_shared_tuned_cfg_map else None),
                )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("5")

    if start_v <= 6.1 <= stop_v:
        _progress_step_started("STEP 6")
        _progress_step_stage("STEP 6", "delay/frequency robustness")
        report.section("STEP 6.1", total=3)

        _report_progress(
            report,
            prefix="STEP6.1 progress",
            current=1,
            total=3,
            extra="reuse existing delay grid" if no_train else "generate + convert delay grid",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, archive_root=archive_root, dry=dry, report=report)
        if not skip_data_generation:
            run_cmd([py, "generate_observation_delay_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_delay_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP6.1 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if skip_data_generation else "dataset figures + purge",
            target="step",
        )
        if not skip_data_generation:
            delay_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step6.1_delay_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and delay_dataset_roots:
                keep_step_graphml_split(
                    graphml_keep_root=graphml_keep_root,
                    step_name="step6.1_delay",
                    train_src_roots=delay_dataset_roots,
                    test_src_roots=[work_dir / name for name in CANONICAL_TEST_TRAJECTORIES],
                    dry=dry,
                    report=report,
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step6.1_post_figures")

        _report_progress(report, prefix="STEP6.1 progress", current=3, total=3, extra="train delay conditions", target="step")
        if not dry:
            delay_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "delay"))
            delay_conditions = _discover_condition_train_dirs(
                work_dir=work_dir,
                archive_root=archive_root,
                live_root=delay_root,
                archive_tag_prefix="step6_delay_",
                tag_prefix="step6_delay_",
                report=report if skip_data_generation else None,
            )
            if delay_root is not None:
                report.kv("delay_root", delay_root)
            if not delay_conditions:
                raise FileNotFoundError("Could not find Step 6.1 delay PT-flat folders in live outputs or archived train_folder copies.")

            for tag, resolved_delay_train in delay_conditions:
                report_label_stats(
                    report,
                    folder=resolved_delay_train,
                    name=f"{tag} TRAIN",
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                st = assert_nonempty_known_labels_folder(
                    resolved_delay_train,
                    name=f"{tag} TRAIN",
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                assert_contiguous_days_per_sim(resolved_delay_train, name=f"{tag} TRAIN", report=report)

                delay_test_folder = _resolve_existing_dataset_dir(
                    requested_folder=LIVE_TEST_REL,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="test_folder",
                    report=report if skip_data_generation else None,
                    fallback_step_tags=["step4_baseline"],
                )
                assert_two_class_folder(
                    delay_test_folder,
                    name=f"{tag} TEST",
                    require_balanced=require_balanced_test,
                    balance_tolerance=tol,
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                assert_contiguous_days_per_sim(delay_test_folder, name=f"{tag} TEST", report=report)

                run_training_step(
                    step_tag=tag,
                    data_folder=resolved_delay_train,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    py=py,
                    dry=dry,
                    report=report,
                    horizons=horizons,
                    run_all_horizons=run_all_horizons,
                    T_list=T_list,
                    run_all_T=run_all_T,
                    archive_train_test_folders=archive_train_test_folders,
                    ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, archive_root=archive_root, tol=tol, report=report),
                    test_folder_for_archive=BASELINE_TEST_REL,
                    no_train=no_train,
                    archive_train_step_tags=[tag],
                    archive_eval_test_step_tags=[tag, "step4_baseline"],
                    archive_test_folder_step_tags=[tag, "step4_baseline"],
                    model_cfg_override=(step4_shared_tuned_cfg_map if step4_shared_tuned_cfg_map else None),
                )
        report.finish_step_console(force_complete=True)
        if stop_v < 6.2:
            _progress_step_completed("6")

    if start_v <= 6.2 <= stop_v:
        if start_v > 6.1:
            _progress_step_started("STEP 6")
        report.section("STEP 6.2", total=3)

        _report_progress(
            report,
            prefix="STEP6.2 progress",
            current=1,
            total=3,
            extra="reuse existing frequency grid" if skip_data_generation else "generate + convert frequency grid",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, archive_root=archive_root, dry=dry, report=report)
        if not skip_data_generation:
            run_cmd([py, "generate_screen_freq_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_freq_grid.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP6.2 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if skip_data_generation else "dataset figures + purge",
            target="step",
        )
        if not skip_data_generation:
            freq_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step6.2_frequency_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and freq_dataset_roots:
                keep_step_graphml_split(
                    graphml_keep_root=graphml_keep_root,
                    step_name="step6.2_frequency",
                    train_src_roots=freq_dataset_roots,
                    test_src_roots=[work_dir / name for name in CANONICAL_TEST_TRAJECTORIES],
                    dry=dry,
                    report=report,
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step6.2_post_figures")

        _report_progress(report, prefix="STEP6.2 progress", current=3, total=3, extra="train frequency conditions", target="step")
        if not dry:
            freq_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "frequency"))
            freq_conditions = _discover_condition_train_dirs(
                work_dir=work_dir,
                archive_root=archive_root,
                live_root=freq_root,
                archive_tag_prefix="step6_freq_",
                tag_prefix="step6_freq_",
                report=report if skip_data_generation else None,
            )
            if freq_root is not None:
                report.kv("freq_root", freq_root)
            if not freq_conditions:
                raise FileNotFoundError("Could not find Step 6.2 frequency PT-flat folders in live outputs or archived train_folder copies.")

            for tag, resolved_freq_train in freq_conditions:
                report_label_stats(
                    report,
                    folder=resolved_freq_train,
                    name=f"{tag} TRAIN",
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                st = assert_nonempty_known_labels_folder(
                    resolved_freq_train,
                    name=f"{tag} TRAIN",
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                warn_if_single_label(st, name=f"{tag} TRAIN", report=report)
                assert_contiguous_days_per_sim(resolved_freq_train, name=f"{tag} TRAIN", report=report)

                freq_test_folder = _resolve_existing_dataset_dir(
                    requested_folder=LIVE_TEST_REL,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag=tag,
                    split_name="test_folder",
                    report=report if skip_data_generation else None,
                    fallback_step_tags=["step4_baseline"],
                )
                assert_two_class_folder(
                    freq_test_folder,
                    name=f"{tag} TEST",
                    require_balanced=require_balanced_test,
                    balance_tolerance=tol,
                    report=report,
                    preferred_label_attr=preferred_label_attr,
                    strict_preferred=True,
                )
                assert_contiguous_days_per_sim(freq_test_folder, name=f"{tag} TEST", report=report)

                run_training_step(
                    step_tag=tag,
                    data_folder=resolved_freq_train,
                    work_dir=work_dir,
                    archive_root=archive_root,
                    py=py,
                    dry=dry,
                    report=report,
                    horizons=horizons,
                    run_all_horizons=run_all_horizons,
                    T_list=T_list,
                    run_all_T=run_all_T,
                    archive_train_test_folders=archive_train_test_folders,
                    ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, archive_root=archive_root, tol=tol, report=report),
                    test_folder_for_archive=BASELINE_TEST_REL,
                    no_train=no_train,
                    archive_train_step_tags=[tag],
                    archive_eval_test_step_tags=[tag, "step4_baseline"],
                    archive_test_folder_step_tags=[tag, "step4_baseline"],
                    model_cfg_override=(step4_shared_tuned_cfg_map if step4_shared_tuned_cfg_map else None),
                )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("6")

    if start_v <= 7.0 <= stop_v:
        _progress_step_started("STEP 7")
        _progress_step_stage("STEP 7", "sweep generation")
        report.section("STEP 7", total=3)

        _report_progress(
            report,
            prefix="STEP7 progress",
            current=1,
            total=3,
            extra="reuse existing sweep" if skip_data_generation else "generate + convert sweep",
            target="step",
        )
        _restore_live_test_from_baseline(work_dir=work_dir, archive_root=archive_root, dry=dry, report=report)
        if not skip_data_generation:
            run_cmd([py, "generate_sweep_regime.py"], dry, cwd=work_dir, report=report, stream_output=False)
            run_cmd([py, "convert_collect_sweep.py"], dry, cwd=work_dir, report=report, stream_output=False)

        _report_progress(
            report,
            prefix="STEP7 progress",
            current=2,
            total=3,
            extra="skip dataset figures + purge" if skip_data_generation else "dataset figures + purge",
            target="step",
        )
        if not skip_data_generation:
            sweep_dataset_roots = archive_dataset_graph_figures_before_purge(
                py=py,
                work_dir=work_dir,
                search_root=work_dir,
                archive_root=archive_root,
                stage_tag="step7_sweep_datasets",
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
            )
            if keep_step_train_graphml and sweep_dataset_roots:
                keep_step_graphml_split(
                    graphml_keep_root=graphml_keep_root,
                    step_name="step7_sweep",
                    train_src_roots=sweep_dataset_roots,
                    test_src_roots=[work_dir / name for name in CANONICAL_TEST_TRAJECTORIES],
                    dry=dry,
                    report=report,
                )
            _purge_graphml_under(work_dir, dry=dry, report=report, label="step7_post_figures")

        _report_progress(report, prefix="STEP7 progress", current=3, total=3, extra="train sweep", target="step")
        if not dry:
            sweep_root = _find_latest_dir(work_dir, _step_dataset_globs(base_task, "sweep"))
            if sweep_root is not None:
                report.kv("sweep_root", sweep_root)
                resolved_sweep_train = sweep_root
            else:
                resolved_sweep_train = _resolve_existing_dataset_dir(
                    requested_folder=Path("step7_sweep"),
                    work_dir=work_dir,
                    archive_root=archive_root,
                    step_tag="step7_sweep",
                    split_name="train_folder",
                    report=report if skip_data_generation else None,
                    fallback_step_tags=[],
                )
                if not _dir_has_pt_files(resolved_sweep_train):
                    raise FileNotFoundError("Could not find Step 7 sweep PT-flat folder or archived sweep run.")
            data_folder_for_step7 = resolved_sweep_train

            report_label_stats(
                report,
                folder=resolved_sweep_train,
                name="Step7 SWEEP TRAIN",
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
            st = assert_nonempty_known_labels_folder(
                resolved_sweep_train,
                name="Step7 SWEEP TRAIN",
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
            warn_if_single_label(st, name="Step7 SWEEP TRAIN", report=report)
            assert_contiguous_days_per_sim(resolved_sweep_train, name="Step7 SWEEP TRAIN", report=report)

            baseline_like_test = _resolve_existing_dataset_dir(
                requested_folder=LIVE_TEST_REL,
                work_dir=work_dir,
                archive_root=archive_root,
                step_tag="step7_sweep",
                split_name="test_folder",
                report=report if skip_data_generation else None,
                fallback_step_tags=["step4_baseline"],
            )
            assert_two_class_folder(
                baseline_like_test,
                name="Step7 TEST",
                require_balanced=require_balanced_test,
                balance_tolerance=tol,
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )
            assert_contiguous_days_per_sim(baseline_like_test, name="Step7 TEST", report=report)

            run_training_step(
                step_tag="step7_sweep",
                data_folder=data_folder_for_step7,
                work_dir=work_dir,
                archive_root=archive_root,
                py=py,
                dry=dry,
                report=report,
                horizons=horizons,
                run_all_horizons=run_all_horizons,
                T_list=T_list,
                run_all_T=run_all_T,
                archive_train_test_folders=archive_train_test_folders,
                ensure_test_folder_for_task=_make_baseline_test_checker(work_dir=work_dir, archive_root=archive_root, tol=tol, report=report),
                test_folder_for_archive=BASELINE_TEST_REL,
                no_train=no_train,
                archive_train_step_tags=["step7_sweep"],
                archive_eval_test_step_tags=["step7_sweep", "step4_baseline"],
                archive_test_folder_step_tags=["step7_sweep", "step4_baseline"],
                model_cfg_override=(step4_shared_tuned_cfg_map if step4_shared_tuned_cfg_map else None),
            )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("7")

    if start_v <= 8.0 <= stop_v:
        _progress_step_started("STEP 8")
        _progress_step_stage("STEP 8", "distribution-shift preparation")
        report.section("STEP 8", total=(6 if tune_step8 and not no_train else 5))
        step8_num_days = int(CONFIG.get("STEP8", {}).get("num_days", int(CONFIG["STEP1"]["num_days"]) * 2))
        step8_specs = _build_step8_shift_trajectories(step8_num_days)
        step8_train_names = list((CONFIG.get("STEP8", {}).get("train_trajectories", {}) or {}).keys())
        step8_test_names = list((CONFIG.get("STEP8", {}).get("test_trajectories", {}) or {}).keys())

        _report_progress(report, prefix="STEP8 progress", current=1, total=5, extra=("reuse existing shifted datasets" if skip_data_generation else "generate shifted trajectories"), target="step")
        if not skip_data_generation:
            generate_named_trajectories(
                work_dir=work_dir,
                py=py,
                dry=dry,
                report=report,
                trajectory_specs=step8_specs,
                n_sims=int(CONFIG.get("STEP8", {}).get("n_sims_per_trajectory", CONFIG["STEP1"]["n_sims_per_trajectory"])),
                num_days=step8_num_days,
                progress_prefix="STEP8 generate",
            )

        _report_progress(report, prefix="STEP8 progress", current=2, total=5, extra=("reuse converted shifted PT" if skip_data_generation else "convert shifted trajectories"), target="step")
        if not skip_data_generation:
            archive_raw_pair_figures_from_trajectory_groups(
                py=py,
                work_dir=work_dir,
                archive_root=archive_root,
                dry=dry,
                report=report,
                cwd=work_dir,
                identity="Harry Triantafyllidis",
                enable_graph_folder_figures=enable_graph_folder_figures,
                stage_tag="step8_distribution_shift_train_vs_test",
                title="Step 8 distribution-shift raw-graph summary: train vs test",
                train_names=step8_train_names,
                test_names=step8_test_names,
            )
            if keep_step_train_graphml:
                keep_step_graphml_split(
                    graphml_keep_root=graphml_keep_root,
                    step_name="step8_distribution_shift",
                    train_src_roots=[work_dir / n for n in step8_train_names],
                    test_src_roots=[work_dir / n for n in step8_test_names],
                    dry=dry,
                    report=report,
                )
            convert_and_flatten_named_trajectories(
                work_dir=work_dir,
                py=py,
                dry=dry,
                report=report,
                trajectory_names=list(step8_specs.keys()),
                progress_prefix="STEP8 convert",
            )
            for name in step8_specs.keys():
                _purge_graphml_under(work_dir / name, dry=dry, report=report, label=f"step8::{name}::post_conversion")

        _report_progress(report, prefix="STEP8 progress", current=3, total=5, extra=("reuse pooled shifted train/test PT" if skip_data_generation else "pool shifted train/test PT"), target="step")
        if not skip_data_generation:
            build_pooled_train_test_from_trajectories(
                work_dir=work_dir,
                train_names=step8_train_names,
                test_names=step8_test_names,
                train_folder=work_dir / STEP8_TRAIN_REL,
                test_folder=work_dir / STEP8_TEST_REL,
                dry=dry,
                report=report,
            )

        _report_progress(report, prefix="STEP8 progress", current=4, total=5, extra="validate shifted benchmark", target="step")
        if not dry:
            step8_train_folder = _resolve_existing_dataset_dir(requested_folder=STEP8_TRAIN_REL, work_dir=work_dir, archive_root=archive_root, step_tag="step8_distribution_shift", split_name="train_folder", report=report if skip_data_generation else None, fallback_step_tags=[])
            step8_test_folder = _resolve_existing_dataset_dir(requested_folder=STEP8_TEST_REL, work_dir=work_dir, archive_root=archive_root, step_tag="step8_distribution_shift", split_name="test_folder", report=report if skip_data_generation else None, fallback_step_tags=[])
            step8_train_stats = validate_train_folder_for_task(
                folder=step8_train_folder,
                task_name=base_task,
                folder_name="Step8 SHIFT TRAIN",
                report=report,
            )
            step8_test_stats = assert_two_class_folder(
                step8_test_folder,
                name="Step8 SHIFT TEST",
                require_balanced=require_balanced_test,
                balance_tolerance=tol,
                report=report,
                preferred_label_attr=preferred_label_attr,
                strict_preferred=True,
            )

            step8_max_majority_frac = float(CONFIG.get("STEP8", {}).get("max_majority_frac", 0.70))
            step8_continue_on_balance_failure = bool(
                CONFIG.get("STEP8", {}).get("continue_to_step9_on_balance_failure", True)
            )
            step8_skip_training_due_to_balance = False
            step8_balance_errors: List[str] = []

            for _stats, _name in (
                (step8_train_stats, "Step8 SHIFT TRAIN"),
                (step8_test_stats, "Step8 SHIFT TEST"),
            ):
                try:
                    _assert_label_ratio_not_extreme(
                        _stats,
                        name=_name,
                        max_majority_frac=step8_max_majority_frac,
                        report=report,
                    )
                except RuntimeError as exc:
                    if not step8_continue_on_balance_failure:
                        raise
                    step8_skip_training_due_to_balance = True
                    step8_balance_errors.append(str(exc))
                    report.write(
                        "STEP8_BALANCE_GATE_NONFATAL "
                        f"dataset={_name!r} continue_to_step9=1 message={str(exc)!r}"
                    )

            if step8_skip_training_due_to_balance:
                _progress_step_stage("STEP 8", "balance gate failed; skip Step 8 training")
                _report_progress(
                    report,
                    prefix="STEP8 progress",
                    current=6 if tune_step8 and not no_train else 5,
                    total=6 if tune_step8 and not no_train else 5,
                    extra="balance gate failed; skip Step8 training/evaluation and continue",
                    target="step",
                )
                for msg in step8_balance_errors:
                    report.write(f"STEP8_TRAIN_EVAL_SKIPPED reason={msg}")
                # Step 9 only needs the shifted test folder and the Step 4 baseline model.
                # Keep a strict contiguity check on the shifted test set before allowing
                # baseline-to-shift evaluation to proceed.
                assert_contiguous_days_per_sim(step8_test_folder, name="Step8 SHIFT TEST", report=report)
            else:
                assert_contiguous_days_per_sim(step8_train_folder, name="Step8 SHIFT TRAIN", report=report)
                assert_contiguous_days_per_sim(step8_test_folder, name="Step8 SHIFT TEST", report=report)
        else:
            step8_skip_training_due_to_balance = False

        step8_tuned_cfg_map: Dict[Tuple[int, Optional[int]], Dict[str, Any]] = {}
        if not step8_skip_training_due_to_balance:
            if tune_step8 and not dry and not no_train:
                _progress_step_stage("STEP 8", "hyperparameter tuning")
                _report_progress(report, prefix="STEP8 progress", current=5, total=6, extra="hyperparameter tuning", target="step")
                step8_tuned_cfg_map = _run_lightweight_tuning(
                    step_tag="step8_distribution_shift",
                    work_dir=work_dir,
                    archive_root=archive_root,
                    py=py,
                    dry=dry,
                    report=report,
                    data_folder=step8_train_folder,
                    task_name=base_task,
                    t_values=[int(t) for t in (T_list if run_all_T else [int(CONFIG["MODEL"]["T"])])],
                    h_values=[int(h) for h in horizons] if run_all_horizons else [None],
                    tune_trials_quick=tune_trials_quick,
                    tune_finalists=tune_finalists,
                    tune_quick_epochs=tune_quick_epochs,
                    tune_full_epochs=tune_full_epochs,
                    tune_split_seed=tune_split_seed,
                )
            _progress_step_stage("STEP 8", "training and evaluation")
            _report_progress(report, prefix="STEP8 progress", current=6 if tune_step8 and not no_train else 5, total=6 if tune_step8 and not no_train else 5, extra="train/evaluate under complete shift", target="step")
            run_training_step(
                step_tag="step8_distribution_shift",
                data_folder=STEP8_TRAIN_REL,
                work_dir=work_dir,
                archive_root=archive_root,
                py=py,
                dry=dry,
                report=report,
                horizons=horizons,
                run_all_horizons=run_all_horizons,
                T_list=T_list,
                run_all_T=run_all_T,
                archive_train_test_folders=archive_train_test_folders,
                ensure_test_folder_for_task=_make_test_folder_checker(
                    work_dir=work_dir,
                    archive_root=archive_root,
                    requested_folder=STEP8_TEST_REL,
                    archive_step_tags=["step8_distribution_shift"],
                    tol=tol,
                    report=report,
                ),
                eval_test_folder=STEP8_TEST_REL,
                test_folder_for_archive=STEP8_TEST_REL,
                no_train=no_train,
                archive_train_step_tags=["step8_distribution_shift"],
                archive_eval_test_step_tags=["step8_distribution_shift"],
                archive_test_folder_step_tags=["step8_distribution_shift"],
                model_cfg_override=(step8_tuned_cfg_map if step8_tuned_cfg_map else None),
            )
        report.finish_step_console(force_complete=True)
        _progress_step_completed("8")

    if start_v <= 9.0 <= stop_v:
        _progress_step_started("STEP 9")
        _progress_step_stage("STEP 9", "baseline-to-shift testing")
        report.section("STEP 9", total=2)
        _report_progress(report, prefix="STEP9 progress", current=1, total=2, extra="locate baseline checkpoint", target="step")
        def _step9_baseline_ckpt(t_val: int, h_val: Optional[int]) -> Path:
            return _archive_dst_for_training_run(
                archive_root=archive_root,
                step_tag="step4_baseline",
                t_val=int(t_val),
                h_val=h_val,
                run_all_T=run_all_T,
                run_all_horizons=run_all_horizons,
            ) / "trained_model.pt"

        if not dry:
            t_iter = T_list if run_all_T else [int(CONFIG["MODEL"]["T"])]
            h_iter = horizons if run_all_horizons else [None]
            for _t in t_iter:
                for _h in h_iter:
                    ck = _step9_baseline_ckpt(int(_t), _h)
                    if not ck.exists():
                        raise FileNotFoundError(f"Missing baseline checkpoint for Step 9: {ck.resolve()}")

        _report_progress(report, prefix="STEP9 progress", current=2, total=2, extra="evaluate baseline model on Step8 shifted test", target="step")
        run_training_step(
            step_tag="step9_baseline_to_shift_test",
            data_folder=BASELINE_TRAIN_REL,
            work_dir=work_dir,
            archive_root=archive_root,
            py=py,
            dry=dry,
            report=report,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            T_list=T_list,
            run_all_T=run_all_T,
            archive_train_test_folders=archive_train_test_folders,
            ensure_test_folder_for_task=_make_test_folder_checker(
                work_dir=work_dir,
                archive_root=archive_root,
                requested_folder=STEP8_TEST_REL,
                archive_step_tags=["step8_distribution_shift"],
                tol=tol,
                report=report,
            ),
            eval_test_folder=STEP8_TEST_REL,
            test_folder_for_archive=STEP8_TEST_REL,
            no_train=True,
            pretrained_model_path_override=_step9_baseline_ckpt,
            archive_train_step_tags=["step4_baseline"],
            archive_eval_test_step_tags=["step8_distribution_shift"],
            archive_test_folder_step_tags=["step8_distribution_shift"],
            model_cfg_override=None,
        )
        report.finish_step_console(force_complete=True)

# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="1")
    ap.add_argument("--stop", type=str, default="9")
    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument(
        "--keep_graphml",
        action="store_true",
        help="If set, preserve GraphML files instead of purging them after conversion/collection.",
    )
    ap.add_argument(
        "--keep_step_train_graphml",
        action="store_true",
        help="If set, keep only the GraphML datasets needed to preserve each generated train set per step and one frozen test set per track.",
    )
    ap.add_argument("--run_both_state_modes", action="store_true")
    ap.add_argument("--archive_train_test_folders", action="store_true")
    ap.add_argument(
        "--run_graph_folder_figures",
        action="store_true",
        help="If set, run graph_folder_figures.py before GraphML purge. Default: disabled.",
    )

    ap.add_argument("--emit_latex", action="store_true")
    ap.add_argument(
        "--emit_latex_only",
        action="store_true",
        help="Skip the full pipeline and only rebuild the Overleaf package from an already completed results root.",
    )
    ap.add_argument(
        "--no_train",
        action="store_true",
        help="Do not retrain. Reuse existing trained_model.pt files to rerun evaluation, regenerate missing test plots/metrics in place, and rebuild LaTeX. Also skips all data generation/conversion stages, including Steps 1--2.",
    )
    ap.add_argument(
        "--no_simulation",
        action="store_true",
        help="Do not generate or convert any data in Steps 1--7. Reuse all existing prepared datasets and still retrain/evaluate the requested models.",
    )
    ap.add_argument("--overleaf_dir", type=str, default=DEFAULT_OVERLEAF_DIRNAME)
    ap.add_argument("--tune_step4", action="store_true", help="Run lightweight validation-only hyperparameter tuning before Step 4 training.")
    ap.add_argument("--tune_step8", action="store_true", help="Run lightweight validation-only hyperparameter tuning before Step 8 training.")
    ap.add_argument("--tune_trials_quick", type=int, default=10)
    ap.add_argument("--tune_finalists", type=int, default=3)
    ap.add_argument("--tune_quick_epochs", type=int, default=12)
    ap.add_argument("--tune_full_epochs", type=int, default=35)
    ap.add_argument("--tune_split_seed", type=int, default=0)

    ap.add_argument(
        "--test_frac_per_class",
        type=float,
        default=float(CONFIG["TEST"].get("test_frac_per_class", 0.5)),
        help="Legacy argument retained for CLI compatibility; ignored by the explicit canonical train/test baseline build.",
    )

    ap.add_argument(
        "--run_all_horizons",
        action="store_true",
        help="If set, train/evaluate each training step across a horizon list.",
    )
    ap.add_argument(
        "--horizons",
        type=str,
        default="",
        help="Optional comma-separated horizons overriding CONFIG['CONVERT']['horizons'] when --run_all_horizons is set.",
    )

    ap.add_argument(
        "--run_all_T",
        action="store_true",
        help="If set, train/evaluate each training step across a list of window lengths T.",
    )
    ap.add_argument(
        "--T_list",
        type=str,
        default="",
        help="Optional comma-separated T values overriding CONFIG['MODEL']['T_list'].",
    )

    ap.add_argument("--results_parent", type=str, default=str(DEFAULT_RESULTS_PARENT))
    ts = ap.add_mutually_exclusive_group()
    ts.add_argument("--timestamped", action="store_true")
    ts.add_argument("--no_timestamp", action="store_true")
    args = ap.parse_args()
    run_started_at = time.time()

    start_v = step_key_to_float(args.start)
    stop_v = step_key_to_float(args.stop)
    dry = bool(args.dry_run)
    py = sys.executable
    timestamped = False

    test_frac = float(args.test_frac_per_class)
    if not (0.0 < test_frac <= 1.0):
        raise ValueError(f"--test_frac_per_class must be in (0,1], got {test_frac}")

    run_all_horizons = bool(args.run_all_horizons)
    run_all_T = bool(args.run_all_T)
    archive_train_test_folders = bool(args.archive_train_test_folders)
    no_train = bool(args.no_train)
    no_simulation = bool(args.no_simulation)
    skip_data_generation = bool(no_train or no_simulation)
    tune_step4 = bool(args.tune_step4)
    tune_step8 = bool(args.tune_step8)
    tune_trials_quick = int(args.tune_trials_quick)
    tune_finalists = int(args.tune_finalists)
    tune_quick_epochs = int(args.tune_quick_epochs)
    tune_full_epochs = int(args.tune_full_epochs)
    tune_split_seed = int(args.tune_split_seed)

    horizons: List[int] = []
    if run_all_horizons:
        if str(args.horizons).strip() != "":
            horizons = _parse_horizons_list(args.horizons)
        else:
            horizons = _parse_horizons_list(str(CONFIG.get("CONVERT", {}).get("horizons", "")).strip())
        if len(horizons) == 0:
            raise RuntimeError("--run_all_horizons is set but no horizons were provided/found.")
    else:
        task_h = _infer_horizon_from_task_name(str(CONFIG["MODEL"].get("task", "")).strip())
        if task_h is not None:
            horizons = [int(task_h)]
        else:
            horizons = [int(CONFIG["MODEL"].get("pred_horizon", 7))]

    t_list: List[int] = []
    if run_all_T:
        if str(args.T_list).strip() != "":
            t_list = _parse_T_list(args.T_list)
        else:
            cfg_t_list = str(CONFIG.get("MODEL", {}).get("T_list", "")).strip()
            if cfg_t_list != "":
                t_list = _parse_T_list(cfg_t_list)
            else:
                t_list = [int(CONFIG["MODEL"]["T"])]
        if len(t_list) == 0:
            raise RuntimeError("--run_all_T is set but no T values were provided/found.")
    else:
        t_list = [int(CONFIG["MODEL"]["T"])]

    project_root = Path(__file__).resolve().parent
    results_parent = Path(args.results_parent)

    dirs = _make_run_dirs(results_parent, timestamped=timestamped, dry=dry)
    run_root = dirs["run_root"]
    report_path = dirs["report_path"]

    report = Report(report_path, dry=dry)
    report.section("RUN METADATA")
    report.kv("utc_started", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
    report.kv("python", sys.executable)
    report.kv("run_root", run_root)
    report.kv("start", args.start)
    report.kv("stop", args.stop)
    report.kv("test_frac_per_class", test_frac)
    report.kv("test_frac_per_class_note", "ignored by explicit canonical train/test baseline build")
    report.kv("run_all_horizons", int(run_all_horizons))
    report.kv("horizons", ",".join(str(h) for h in horizons))
    report.kv("run_all_T", int(run_all_T))
    report.kv("T_list", ",".join(str(t) for t in t_list))
    report.kv("archive_train_test_folders", int(archive_train_test_folders))
    report.kv("run_graph_folder_figures", int(bool(args.run_graph_folder_figures)))
    report.kv("emit_latex", int(bool(args.emit_latex)))
    report.kv("emit_latex_only", int(bool(args.emit_latex_only)))
    report.kv("no_train", int(bool(args.no_train)))
    report.kv("no_simulation", int(bool(args.no_simulation)))
    report.kv("skip_data_generation", int(bool(skip_data_generation)))
    report.kv("tune_step4", int(bool(tune_step4)))
    report.kv("tune_step8", int(bool(tune_step8)))
    report.kv("tune_trials_quick", tune_trials_quick)
    report.kv("tune_finalists", tune_finalists)
    report.kv("tune_quick_epochs", tune_quick_epochs)
    report.kv("tune_full_epochs", tune_full_epochs)
    report.kv("tune_split_seed", tune_split_seed)
    report.kv("resume_root_mode", 1)
    if no_train and no_simulation:
        report.write("NOTE: both --no_train and --no_simulation were set; evaluation-only behavior takes precedence for model training.")
    report.kv("timestamped_flags_ignored", 1)
    report.kv("results_parent", str(results_parent))

    if bool(args.emit_latex_only):
        if not run_root.exists():
            raise FileNotFoundError(f"Results root does not exist: {run_root}")

        total_runtime_sec = time.time() - run_started_at
        total_runtime_hms = _format_duration_hms(total_runtime_sec)

        export_overleaf_package(
            run_root=run_root,
            archive_root=run_root,
            out_dir=Path(str(args.overleaf_dir)),
            dry=dry,
            report=report,
        )

        report.section("DONE")
        report.kv("utc_finished", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
        report.kv("total_runtime_sec", round(total_runtime_sec, 3))
        report.kv("total_runtime_hms", total_runtime_hms)
        report.kv("report_txt", report_path)
        if os.environ.get("DT_SUPPRESS_FINAL_PRINT", "0") != "1":
            print(f"TOTAL RUN TIME: {total_runtime_hms}", flush=True)
        return 0

    required = [
        "generate_amr_data.py",
        "convert_to_pt.py",
        "train_amr_dygformer.py",
        "tune_hparams.py",
        "temporal_graph_dataset.py",
        "ablate_edge_weights.py",
        "ablate_node_features.py",
        "generate_sweep_regime.py",
        "convert_collect_sweep.py",
        "generate_observation_delay_grid.py",
        "convert_collect_delay_grid.py",
        "generate_screen_freq_grid.py",
        "convert_collect_freq_grid.py",
        "graph_folder_figures.py",
        "models_amr.py",
        "tasks.py",
    ]

    if dry:
        for f in required:
            require_file(project_root / f)
        report.write(f"Required scripts present in project_root: {len(required)}")
        report.write("DRY_RUN: code staging skipped")
    else:
        report.write("Code will be staged into persistent per-track work_dir locations")

    os.environ["DT_SIM_EXTRA_ARGS"] = _build_sim_extra_args_global_only(CONFIG["SIM"])
    effective_convert_cfg = dict(CONFIG["CONVERT"])
    effective_convert_cfg["horizons"] = ",".join(str(h) for h in horizons)

    os.environ["DT_CONVERT_EXTRA_ARGS"] = _build_convert_extra_args(effective_convert_cfg)
    #os.environ["DT_KEEP_GRAPHML"] = "0"
    os.environ["DT_KEEP_GRAPHML"] = "1" if bool(args.keep_graphml) else "0"

    report.section("ENVIRONMENT")
    report.kv("DT_SIM_EXTRA_ARGS", os.environ["DT_SIM_EXTRA_ARGS"])
    report.kv("DT_CONVERT_EXTRA_ARGS", os.environ["DT_CONVERT_EXTRA_ARGS"])
    report.kv("DT_KEEP_GRAPHML", os.environ["DT_KEEP_GRAPHML"])
    report.kv("keep_graphml_note", "GraphML purge is skipped when --keep_graphml is set.")
    report.kv("keep_step_train_graphml", int(bool(args.keep_step_train_graphml)))
    report.kv("keep_step_train_graphml_note", "When set, each preserved kept_graphml step folder uses a simple layout: kept_graphml/<step_name>/train and kept_graphml/<step_name>/test.")
    report.kv("no_simulation_note", "When set, all simulation/data-generation/conversion stages are skipped and existing prepared datasets are reused.")
    
    def _run_one_track(track_name: str, state_mode: str, pt_out_dir: str, progress_path: Optional[Path] = None) -> None:
        track_dirs = _make_track_dirs(run_root, track_name, dry=dry)
        work_dir = track_dirs["work_dir"]
        archive_root = track_dirs["archive_root"]

        report.section(f"TRACK: {track_name}")
        report.kv("track_root", track_dirs["track_root"])
        report.kv("work_dir", work_dir)
        report.kv("archive_root", archive_root)
        report.kv("graphml_keep_root", track_dirs["graphml_keep_root"])

        if dry:
            for f in required:
                require_file(project_root / f)
        else:
            stage_code_into_workdir(project_root, work_dir, dry=dry)
            for f in required:
                require_file(work_dir / f)

        os.environ["DT_STATE_MODE"] = state_mode
        os.environ["DT_PT_OUT_DIR"] = pt_out_dir
        os.environ["PYTHONPATH"] = str(work_dir.resolve())

        report.kv("DT_STATE_MODE", os.environ["DT_STATE_MODE"])
        report.kv("DT_PT_OUT_DIR", os.environ["DT_PT_OUT_DIR"])
        report.kv("PYTHONPATH", os.environ["PYTHONPATH"])

        planned_step_keys = _planned_parallel_step_keys(start_v, stop_v)
        completed_step_keys: set[str] = set()
        progress_started_at = time.time()
        active_step_label = ""
        stage_counter = 0

        def _emit_track_progress(event: str, step_value: str) -> None:
            nonlocal active_step_label, stage_counter
            if progress_path is None:
                return
            event_s = str(event).strip().lower()
            step_s = str(step_value).strip()
            if event_s == "completed" and step_s:
                completed_step_keys.add(step_s)
            elif event_s == "started":
                active_step_label = step_s
                stage_counter += 1
            payload = {
                "track_name": track_name,
                "state_mode": state_mode,
                "status": "running",
                "active_step": active_step_label,
                "completed_steps": len(completed_step_keys),
                "total_steps": len(planned_step_keys),
                "stage_counter": stage_counter,
                "elapsed_sec": max(0.0, time.time() - progress_started_at),
                "updated_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
            }
            _write_track_progress_state(progress_path, payload)

        if progress_path is not None:
            _write_track_progress_state(
                progress_path,
                {
                    "track_name": track_name,
                    "state_mode": state_mode,
                    "status": "starting",
                    "active_step": "",
                    "completed_steps": 0,
                    "total_steps": len(planned_step_keys),
                    "elapsed_sec": 0.0,
                    "stage_counter": 0,
                    "updated_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                },
            )

        run_pipeline_once(
            start_v=start_v,
            stop_v=stop_v,
            dry=dry,
            py=py,
            archive_train_test_folders=archive_train_test_folders,
            archive_root=archive_root,
            graphml_keep_root=track_dirs["graphml_keep_root"],
            work_dir=work_dir,
            test_frac_per_class=test_frac,
            report=report,
            horizons=horizons,
            run_all_horizons=run_all_horizons,
            T_list=t_list,
            run_all_T=run_all_T,
            enable_graph_folder_figures=bool(args.run_graph_folder_figures),
            keep_step_train_graphml=bool(args.keep_step_train_graphml),
            no_train=no_train,
            no_simulation=no_simulation,
            tune_step4=tune_step4,
            tune_step8=tune_step8,
            tune_trials_quick=tune_trials_quick,
            tune_finalists=tune_finalists,
            tune_quick_epochs=tune_quick_epochs,
            tune_full_epochs=tune_full_epochs,
            tune_split_seed=tune_split_seed,
            progress_callback=_emit_track_progress,
        )

        if progress_path is not None:
            _write_track_progress_state(
                progress_path,
                {
                    "track_name": track_name,
                    "state_mode": state_mode,
                    "status": "finalizing",
                    "active_step": "",
                    "completed_steps": len(completed_step_keys),
                    "total_steps": len(planned_step_keys),
                    "stage_counter": stage_counter,
                    "elapsed_sec": max(0.0, time.time() - progress_started_at),
                    "updated_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                },
            )

        if no_train or no_simulation:
            cleaned_n = 0
            skip_reason = "--no_train" if no_train else "--no_simulation"
            report.write(
                f"PT_CLEANUP skipped for track={track_name} under {work_dir} because {skip_reason} was set"
            )
        else:
            cleaned_n = _cleanup_transient_pt_files(
                work_dir=work_dir,
                pt_out_dir=Path(pt_out_dir),
                dry=dry,
                report=report,
            )

            report.write(
                f"PT_CLEANUP completed for track={track_name} under {work_dir}; transient_pt_deleted={cleaned_n}"
            )

        if progress_path is not None:
            _write_track_progress_state(
                progress_path,
                {
                    "track_name": track_name,
                    "state_mode": state_mode,
                    "status": "done",
                    "active_step": "",
                    "completed_steps": len(planned_step_keys),
                    "total_steps": len(planned_step_keys),
                    "stage_counter": max(stage_counter, len(planned_step_keys)),
                    "elapsed_sec": max(0.0, time.time() - progress_started_at),
                    "updated_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                },
            )

    def _child_cmd_without_parallel_flags() -> List[str]:
        filtered: List[str] = []
        skip_next = False
        for token in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if token == "--run_both_state_modes":
                continue
            if token == "--emit_latex":
                continue
            filtered.append(token)
        return [py, str(Path(__file__).resolve())] + filtered

    def _run_both_tracks_in_parallel() -> None:
        track_specs = [
            ("TRACK_ground_truth", "ground_truth"),
            ("TRACK_partial_observation", "partial_observation"),
        ]
        procs: List[Tuple[str, subprocess.Popen[str], Path]] = []
        report.write("PARALLEL_TRACKS enabled=1 mode=subprocess")
        for track_name, state_mode in track_specs:
            pt_out_dir = str(_default_pt_out_dir_for_track(run_root, track_name))
            progress_path = run_root / track_name / "track_progress.json"
            env = os.environ.copy()
            env["DT_STATE_MODE"] = state_mode
            env["DT_PT_OUT_DIR"] = pt_out_dir
            env["DT_REPORT_FILENAME"] = f"run_report_{track_name}.txt"
            env["DT_PROGRESS_FILE"] = str(progress_path)
            env["DT_DISABLE_CONSOLE_PROGRESS"] = "1"
            env["DT_SUPPRESS_FINAL_PRINT"] = "1"
            env["DT_TRAIN_CPU_BUDGET"] = str(max(1, _machine_cpu_count() // 2))
            cmd = _child_cmd_without_parallel_flags()
            report.write(
                f"PARALLEL_TRACK_LAUNCH track={track_name} state_mode={state_mode} pt_out_dir={pt_out_dir} report_file={env['DT_REPORT_FILENAME']} progress_file={progress_path}"
            )
            _write_track_progress_state(
                progress_path,
                {
                    "track_name": track_name,
                    "state_mode": state_mode,
                    "status": "queued",
                    "active_step": "",
                    "completed_steps": 0,
                    "total_steps": len(_planned_parallel_step_keys(start_v, stop_v)),
                    "elapsed_sec": 0.0,
                    "stage_counter": 0,
                    "updated_at_utc": _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                },
            )
            proc = subprocess.Popen(cmd, env=env)
            procs.append((track_name, proc, progress_path))

        console_initialized = False
        failures: List[Tuple[str, int]] = []
        still_running = {track_name for track_name, _, _ in procs}

        while still_running:
            states: List[Dict[str, Any]] = []
            for track_name, proc, progress_path in procs:
                state = _read_track_progress_state(progress_path)
                if not state:
                    state = {
                        "track_name": track_name,
                        "status": "queued",
                        "completed_steps": 0,
                        "total_steps": len(_planned_parallel_step_keys(start_v, stop_v)),
                        "elapsed_sec": 0.0,
                        "active_step": "",
                        "stage_counter": 0,
                    }
                state.setdefault("track_name", track_name)
                rc = proc.poll()
                if rc is None:
                    still_running.add(track_name)
                else:
                    if track_name in still_running:
                        still_running.remove(track_name)
                        report.write(f"PARALLEL_TRACK_DONE track={track_name} returncode={rc}")
                        if rc != 0:
                            failures.append((track_name, rc))
                            state["status"] = "failed"
                        else:
                            state["status"] = state.get("status", "done") or "done"
                        _write_track_progress_state(progress_path, state)
                states.append(state)

            console_initialized = _render_parallel_tracks_console(states, initialized=console_initialized)
            if still_running:
                time.sleep(0.5)

        final_states = [
            _read_track_progress_state(progress_path) or {
                "track_name": track_name,
                "status": "done",
                "completed_steps": len(_planned_parallel_step_keys(start_v, stop_v)),
                "total_steps": len(_planned_parallel_step_keys(start_v, stop_v)),
                "elapsed_sec": 0.0,
                "active_step": "",
                "stage_counter": len(_planned_parallel_step_keys(start_v, stop_v)),
            }
            for track_name, _, progress_path in procs
        ]
        console_initialized = _render_parallel_tracks_console(final_states, initialized=console_initialized)
        if console_initialized and sys.stdout.isatty() and str(os.environ.get("TERM", "")).strip().lower() != "dumb":
            line_count = max(1, int(_PARALLEL_CONSOLE_LINE_COUNT or len(final_states) or 1))
            if line_count > 1:
                sys.stdout.write(f"\033[{line_count - 1}E")
            else:
                sys.stdout.write("\r")
            sys.stdout.write("\033[?25h")
            sys.stdout.write("\n")
            sys.stdout.flush()

        if failures:
            fail_bits: List[str] = []
            for track_name, rc in failures:
                report_file = run_root / track_name / f"run_report_{track_name}.txt"
                fail_bits.append(f"{track_name}={rc} (report={report_file})")
            fail_msg = ", ".join(fail_bits)
            raise RuntimeError(f"Parallel track execution failed: {fail_msg}")

    try:
        if args.run_both_state_modes:
            _run_both_tracks_in_parallel()
        else:
            state_mode_single = os.environ.get("DT_STATE_MODE", "ground_truth").strip() or "ground_truth"
            track_name_single = f"TRACK_{state_mode_single}"
            pt_out_dir_single = os.environ.get("DT_PT_OUT_DIR", "").strip()
            if pt_out_dir_single == "":
                pt_out_dir_single = str(_default_pt_out_dir_for_track(run_root, track_name_single))

            progress_file_single = os.environ.get("DT_PROGRESS_FILE", "").strip()
            _run_one_track(
                track_name=track_name_single,
                state_mode=state_mode_single,
                pt_out_dir=pt_out_dir_single,
                progress_path=(Path(progress_file_single) if progress_file_single else None),
            )

        total_runtime_sec = time.time() - run_started_at
        total_runtime_hms = _format_duration_hms(total_runtime_sec)

        report.section("DONE")
        report.kv("utc_finished", _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"))
        report.kv("total_runtime_sec", round(total_runtime_sec, 3))
        report.kv("total_runtime_hms", total_runtime_hms)
        report.kv("report_txt", report_path)

        if bool(args.emit_latex) and not dry:
            export_overleaf_package(
                run_root=run_root,
                archive_root=run_root,
                out_dir=Path(str(args.overleaf_dir)),
                dry=dry,
                report=report,
            )

        print(f"TOTAL RUN TIME: {total_runtime_hms}", flush=True)

        return 0
    finally:
        report.finish_step_console(force_complete=True)


if __name__ == "__main__":
    raise SystemExit(main())
