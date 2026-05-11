# AMR Digital Twin: Hospital Simulation and Temporal Graph Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch%20Geometric](https://img.shields.io/badge/PyTorch%20Geometric-enabled-orange)
![R](https://img.shields.io/badge/R-Shiny-276DC3)
![Workflow](https://img.shields.io/badge/Workflow-Reproducible-success)

A reproducible research platform for simulating hospital antimicrobial resistance (AMR) dynamics as daily directed contact graphs, transforming them into temporal graph-learning datasets, and training temporal graph neural networks for mechanism-aware forecasting tasks.

This repository integrates three layers in a single workflow:

1. **Mechanistic hospital simulation** of transmission, importation, selection, screening, isolation, and regime shifts.
2. **Graph conversion and target construction** from daily GraphML trajectories to PyTorch Geometric datasets.
3. **End-to-end experimental orchestration** for baselines, ablations, robustness analyses, translational attribution figures, and paper-ready artifact generation.

The main reproducibility entrypoint is:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

For calibration-enabled runs, add `--tune_step4 --tune_step8`.

---

## Contents

- [Overview](#overview)
- [Core capabilities](#core-capabilities)
- [Digital-twin simulation update](#digital-twin-simulation-update)
- [Repository layout](#repository-layout)
- [Environment reproduction](#environment-reproduction)
- [Quick start](#quick-start)
- [Full reproducibility pipeline](#full-reproducibility-pipeline)
- [Resuming and partial reruns](#resuming-and-partial-reruns)
- [Evaluation-only, no-simulation, and tuning modes](#evaluation-only-no-simulation-and-tuning-modes)
- [Important command-line options](#important-command-line-options)
- [Outputs](#outputs)
- [GraphML retention and statistics workflow](#graphml-retention-and-statistics-workflow)
- [Overleaf export](#overleaf-export)
- [Reproducibility and design notes](#reproducibility-and-design-notes)
- [Citation](#citation)
- [License](#license)

---

## Overview

The platform is designed for controlled, mechanism-grounded AMR experimentation in hospital environments. It represents each hospital day as a directed contact graph whose nodes correspond to patients and staff, and whose edges encode contact opportunities and transmission-relevant interactions. These daily graphs are then assembled into temporal sequences for downstream graph learning.

The repository currently supports two experimental state modes:

- **`ground_truth`**
- **`partial_observation`**

The active reproducibility driver in this repository is **`experiments_pbc7.py`**.

---

## Core capabilities

### 1) Mechanistic AMR hospital simulator

`generate_amr_data.py` simulates ward-structured hospital AMR dynamics with support for:

- patients and staff as distinct node types
- directed, weighted contact graphs
- ward assignment, home-ward structure, and multi-ward staffing
- admission-based importation pressure
- within-hospital transmission with individual transmission-event tracking
- bacterial-type inheritance from infectious source agents for successful transmission events
- competing same-day exposure resolution using antibiotic context
- infection-state-dependent infectivity via configurable beta multipliers
- antibiotic exposure and within-host selection effects
- leaky cross-ward staff hand hygiene for staff-patient contacts outside staff home wards
- screening and isolation interventions, including symptomatic-only admission screening, capacity-limited passive screening, active screening on C→I progression, and resistant-only isolation triggers
- optional partial-observation regimes
- configurable outbreak, surveillance, seasonality, and superspreader settings

The simulator writes daily GraphML snapshots together with trajectory-level metadata and label-relevant summaries used downstream for audit and task construction.

## Digital-twin simulation update

This version includes updated digital-twin transmission, screening, and isolation mechanics. These changes affect the simulator outputs and therefore require full regeneration of the predictive datasets before retraining or re-evaluating models. Previous results generated under the older simulator are not directly comparable unless the exact same simulator version and configuration are used.

### Transmission mechanics

- Successful infectious contacts are tracked as individual transmission events.
- When agent `i` acquires infection or carriage from infectious contact with agent `j`, the acquiring agent inherits the bacterial type carried by `j`: resistant or sensitive.
- Multiple successful infectious contacts within the same day are resolved using antibiotic context:
  - no antibiotics: sensitive exposure is favoured over resistant exposure;
  - antibiotics: resistant exposure is favoured over sensitive exposure.
- Infectivity is state dependent. The default-style configuration uses lower infectivity for colonised sensitive and infected sensitive states relative to their resistant counterparts, while keeping the resistant global multiplier neutral (`etaR = 1`).
- Leaky hand hygiene is applied to cross-ward staff-patient contacts when staff interact outside their home ward.

### Screening and isolation baseline

- Admission screening is symptomatic-only by default: only patients in symptomatic infected states (`IS` or `IR`) are eligible for admission screening.
- Passive screening is a capacity-limited convenient sample.
- Active screening is triggered when a patient progresses from colonisation to symptomatic infection.
- Isolation is triggered only for resistant-positive detections by default.

### Explicitly not modelled in this version

- Bed/environment contamination is not included.
- Isolation does not physically move patients to a separate isolation ward; it remains implemented as a contact/transmission modifier.

### Clinically relevant ablation

The repository includes `ablate_staff_patient_contacts.py` for constructing an ablation in which detailed staff-patient contact information is hidden. This is intended to test whether predictive performance depends on explicit staff-patient mixing detail.


### 2) GraphML to PyTorch Geometric conversion

`convert_to_pt.py` converts GraphML trajectories into `.pt` graph objects for temporal learning.

Converted samples store, depending on task and configuration:

- node features
- edge features
- graph-level targets
- stable node identifiers for traceability
- horizon-specific labels for forecasting tasks
- ward-aware metadata used for downstream translational attribution, including:
  - `ward_id`
  - `ward_ids`
  - `ward_cover_count`

Auxiliary CSV outputs are also generated for auditing and downstream checks.

### 3) Temporal graph learning

`train_amr_dygformer.py` trains the temporal learning model used throughout the experiments. The model stack combines:

- a daily graph encoder built around **GraphSAGE-style message passing**
- a **Transformer-based temporal encoder** over windows of length `T`
- task-specific target logic from `tasks.py`

The canonical paper pipeline currently uses:

```text
task = endogenous_importation_majority_h7
T = 7
horizon = 7
```

Available task definitions can be listed via:

```bash
python list_tasks.py
```

### 4) Translational attribution outputs

The training pipeline can export post hoc translational figures derived from learned node-attention summaries and preserved ward metadata. These are intended as interpretability summaries rather than causal claims.

Current figure families include:

- **ward attribution heatmaps**
- **staff bridge plots**
- **ward importance vs downstream burden plots**
- **home-vs-cross-ward contribution decompositions**

These are exported into `translational_figures/` within training outputs and can be collected into the Overleaf export.

### 5) End-to-end reproducibility workflow

`experiments_pbc7.py` orchestrates the full paper-style workflow, including:

- canonical data generation
- conversion and flattening of PT datasets
- baseline train/test construction
- frozen-test evaluation
- ablation studies
- observation-delay robustness experiments
- screening-frequency robustness experiments
- sweep-regime benchmarking
- a complete distribution-shift benchmark
- a baseline-to-shift transfer probe
- translational figure export
- figure generation and archive creation
- Overleaf-ready export

---

## Repository layout

### Core pipeline

- `experiments_pbc7.py` — main reproducibility driver for Steps 1–9
- `generate_amr_data.py` — mechanistic AMR simulator
- `convert_to_pt.py` — GraphML to PyTorch Geometric conversion
- `train_amr_dygformer.py` — temporal graph learning trainer
- `temporal_graph_dataset.py` — temporal dataset construction
- `models_amr.py` — model definitions
- `tasks.py` — forecasting tasks and target extraction
- `list_tasks.py` — utility to inspect registered tasks

### Dataset generation and transformation

- `run_turnover_cohorts.py`
- `prepare_pt_flat_from_turnover.py`
- `make_combined_pt_folder.py`
- `generate_observation_delay_grid.py`
- `convert_collect_delay_grid.py`
- `generate_screen_freq_grid.py`
- `convert_collect_freq_grid.py`
- `generate_sweep_regime.py`
- `convert_collect_sweep.py`

### Evaluation, diagnostics, and figures

- `ablate_edge_weights.py`
- `ablate_node_features.py`
- `graph_folder_figures.py`
- `run_graph_folder_figures_batch.py`
- `mechanism_separation_from_sims.py`
- `summarise_mechanism_components.py`
- `audit_endog_import_labels.py`
- `audit_pt_endog_import_h7.py`
- `check_folder_label_balance.py`
- `check_test_label_balance.py`
- `prune_overleaf_package.py`

### Dataset-folder builders

- `build_balanced_test_folder.py`
- `build_contiguous_test_folder.py`
- `build_delay_test_folder.py`
- `build_sweep_test_folder.py`

### R / Shiny interface

- `run.R`
- `simulator.R`
- `renv.lock`

---

## Environment reproduction

The included conda environment export is:

- `idp_ns_environment.yml`

### Recommended reproduction

To recreate the Python environment used by the main pbc7 workflow:

```bash
conda env create -f idp_ns_environment.yml
conda activate idp_ns
```

If conda solving is slow, the same file can be used with mamba:

```bash
mamba env create -f idp_ns_environment.yml
mamba activate idp_ns
```

### Verification

After recreation, verify the key packages used by the Python workflow:

```bash
python -c "import torch, torch_geometric, networkx, pandas, sklearn; print('torch', torch.__version__); print('pyg', torch_geometric.__version__)"
```

The R/Shiny side is tracked separately through `renv.lock`. Restore the R environment from R if you need to run `run.R` or `simulator.R`:

```bash
R -e "renv::restore()"
```

---

## Quick start

### 1) Generate a single simulation

```bash
python generate_amr_data.py --output_dir demo_sim --seed 123 --num_days 30 --export_yaml
```

### 2) Convert GraphML snapshots to `.pt`

```bash
python convert_to_pt.py --graphml_dir demo_sim
```

### 3) Train a temporal model manually

```bash
python train_amr_dygformer.py --data_folder demo_sim --task endogenous_importation_majority_h7 --T 7 --epochs 10 --batch_size 16
```

### 4) Inspect registered tasks

```bash
python list_tasks.py
```

---

## Full reproducibility pipeline

The recommended full experiment run is:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

To run the same workflow with validation-only hyperparameter calibration before Step 4 and Step 8 training:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml --tune_step4 --tune_step8
```

For a preflight check of required scripts and argument parsing without running the experiment:

```bash
python experiments_pbc7.py --dry_run
```

If you also want the pipeline itself to generate dataset graph summaries while running, add:

```bash
--run_graph_folder_figures
```

Example:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml --run_graph_folder_figures
```

This pipeline is designed to be:

- deterministic
- resumable
- strict about train/test dataset integrity
- track-aware across `ground_truth` and `partial_observation`
- suitable for paper-grade figure and archive generation

When `--run_both_state_modes` is used, `experiments_pbc7.py` launches the `ground_truth` and `partial_observation` tracks as subprocesses and writes per-track reports in addition to the root `run_report.txt`.

### Canonical design

The pipeline is organized around four explicit canonical Step 1 trajectory families:

- `endog_high_train`
- `import_high_train`
- `endog_high_test`
- `import_high_test`

These are used to construct a frozen baseline split:

- `synthetic_amr_graphs_train`
- `synthetic_amr_graphs_test_frozen`

### Step structure

- **Step 1**: generate the four canonical trajectory families
- **Step 2**: convert each trajectory family to `.pt` and flatten family-specific PT folders
- **Step 3**: build the baseline train/test pair from the canonical families
- **Step 4**: train and evaluate the baseline against the frozen test set
- **Step 5**: run ablations
- **Step 6.1**: run observation-delay robustness experiments
- **Step 6.2**: run screening-frequency robustness experiments
- **Step 7**: run sweep-regime experiments
- **Step 8**: run a complete distribution-shift benchmark with shifted train and shifted test cohorts
- **Step 9**: evaluate the baseline Step 4 model directly on the Step 8 shifted test set to measure transfer without retraining

A central design safeguard is that later canonical-side steps may alter the **training side** while preserving evaluation semantics through the **frozen baseline test set**.

Before each canonical training call, the frozen baseline test dataset is restored to the live trainer path:

```text
synthetic_amr_graphs_test
```

This prevents accidental evaluation drift across experimental stages.

### Single-track runs

Ground-truth mode only:

```bash
DT_STATE_MODE=ground_truth python experiments_pbc7.py --emit_latex
```

Partial-observation mode only:

```bash
DT_STATE_MODE=partial_observation python experiments_pbc7.py --emit_latex
```

### Multi-horizon and multi-window runs

```bash
python experiments_pbc7.py --run_both_state_modes --run_all_horizons --run_all_T --emit_latex
```

---

## Resuming and partial reruns

The main pipeline is resumable by step. Step keys are inclusive for both `--start` and `--stop`.

### Example: rerun Steps 4 to 9 for both tracks

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

### Example: rerun Steps 6.2 to 9 for both tracks

```bash
python experiments_pbc7.py --run_both_state_modes --start 6.2 --stop 9 --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

### Example: rerun Steps 6.2 to 9 for ground-truth mode only

```bash
DT_STATE_MODE=ground_truth python experiments_pbc7.py --start 6.2 --stop 9 --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

### Supported step keys

- `1`
- `2`
- `3`
- `4`
- `5`
- `6.1`
- `6.2`
- `7`
- `8`
- `9`

If Steps 1–3 have already completed successfully and the canonical datasets are unchanged, resuming from **Step 4** is equivalent to continuing the original full run.

---

## Evaluation-only, no-simulation, and tuning modes

`experiments_pbc7.py` includes two reuse modes for working from an existing `experiments_results/` tree.

### Reuse prepared datasets and retrain models

Use `--no_simulation` to skip simulation, conversion, and dataset generation/collection work while still retraining and evaluating the requested models on already prepared folders:

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --no_simulation --emit_latex --run_all_T
```

### Rerun evaluation/export without retraining

Use `--no_train` to reuse existing `trained_model.pt` checkpoints, rerun evaluation and missing plots/metrics, and rebuild LaTeX exports. This mode also skips data generation and conversion, so the relevant archived datasets and checkpoints must already exist:

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 9 --no_train --emit_latex --run_all_T
```

If both `--no_train` and `--no_simulation` are supplied, evaluation-only behavior takes precedence.

### Lightweight hyperparameter calibration

Use `--tune_step4` and/or `--tune_step8` to run validation-only hyperparameter searches through `tune_hparams.py` before the corresponding training stage:

```bash
python experiments_pbc7.py --run_both_state_modes --start 4 --stop 4 --tune_step4
python experiments_pbc7.py --run_both_state_modes --start 8 --stop 8 --tune_step8
```

Tuning outputs are archived with the same step-specific artifact workflow as model outputs. Step 4 tuned settings are reused for downstream canonical-side training stages when available; Step 8 keeps its own distribution-shift tuning configuration.

---

## Important command-line options

Commonly used options in `experiments_pbc7.py` include:

```bash
--start
--stop
--dry_run
--run_both_state_modes
--archive_train_test_folders
--run_graph_folder_figures
--emit_latex
--emit_latex_only
--overleaf_dir
--results_parent
--run_all_horizons
--horizons 7,14
--run_all_T
--T_list 7,14
--keep_graphml
--keep_step_train_graphml
--no_simulation
--no_train
--tune_step4
--tune_step8
--tune_trials_quick 10
--tune_finalists 3
--tune_quick_epochs 12
--tune_full_epochs 35
--tune_split_seed 0
```

Retained for compatibility but ignored by the deterministic pbc7 results layout or canonical baseline construction:

```bash
--timestamped
--no_timestamp
--test_frac_per_class
```

Notes:

- `--keep_step_train_graphml` selectively preserves train/test GraphML folders under `kept_graphml/` for later statistics and figure workflows.
- `--keep_graphml` disables GraphML purging after conversion/collection. This can use substantially more disk space than `--keep_step_train_graphml`.
- `--no_simulation` reuses existing prepared datasets but still runs training/evaluation.
- `--no_train` reuses existing checkpoints and runs evaluation/export only.
- `--test_frac_per_class` is retained for CLI compatibility but is not used in the explicit canonical baseline build.
- `--timestamped` and `--no_timestamp` are accepted, but pbc7 writes to the deterministic `--results_parent` root.

---

## Outputs

A typical full run writes to a deterministic results root such as:

```text
experiments_results/
├── run_report.txt
├── run_report_TRACK_ground_truth.txt
├── run_report_TRACK_partial_observation.txt
├── TRACK_ground_truth/
│   ├── kept_graphml/
│   ├── track_progress.json
│   └── work/
│       ├── synthetic_amr_graphs_train/
│       ├── synthetic_amr_graphs_test/
│       ├── synthetic_amr_graphs_test_frozen/
│       ├── training_outputs*/
│       ├── tuning_outputs*/
│       └── repro_artifacts_steps_1_9/
└── TRACK_partial_observation/
    ├── kept_graphml/
    ├── track_progress.json
    └── work/
        ├── synthetic_amr_graphs_train/
        ├── synthetic_amr_graphs_test/
        ├── synthetic_amr_graphs_test_frozen/
        ├── training_outputs*/
        ├── tuning_outputs*/
        └── repro_artifacts_steps_1_9/
```

Archived outputs may include:

- training curves
- confusion matrices
- ROC figures
- saved model checkpoints
- tuning outputs and best hyperparameter configurations when tuning is enabled
- translational figures
- copied train/test datasets when archiving is enabled
- dataset graph-figure summaries
- LaTeX-ready figure bundles for manuscript integration

---

## GraphML retention and statistics workflow

If you want to generate pooled baseline train-vs-frozen-test diagnostic pages after the main run, keep GraphML during the experiment:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex --run_all_T --archive_train_test_folders --keep_step_train_graphml
```

This preserves selected GraphML folders under:

```text
experiments_results/TRACK_ground_truth/kept_graphml/
experiments_results/TRACK_partial_observation/kept_graphml/
```

These retained folders can then be processed by:

```bash
python run_graph_folder_figures_batch.py --max_graphs 70%
```

This batch script:

- compares pooled baseline train vs frozen test for each track
- runs `graph_folder_figures.py` in compare mode
- builds three summary pages per track
- writes a manuscript-ready:

```text
graph_folder_figures_batch/statistics.tex
```

Important notes:

- `--keep_step_train_graphml` is the recommended retention mode for post-run statistics because it preserves selected train/test GraphML folders without keeping every transient GraphML file.
- `--keep_graphml` keeps all GraphML files that would otherwise be purged after conversion/collection and can require much more disk space.
- `--run_graph_folder_figures` inside `experiments_pbc7.py` only calls `graph_folder_figures.py` directly for archived datasets.
- it does **not** call `run_graph_folder_figures_batch.py`.
- the batch script is a separate post-processing step.

---

## Overleaf export

When `--emit_latex` is enabled, the pipeline assembles an Overleaf-ready package by collecting archived figures and generating a `latex.txt` file with reusable manuscript figure blocks.

Example:

```bash
python experiments_pbc7.py --run_both_state_modes --emit_latex
```

Default export location:

```text
experiments_results/overleaf_package/
```

Typical contents:

- `figures/`
- `latex.txt`

The Overleaf export may include:

- main predictive figures
- cross-track comparison figures
- dataset diagnostic figures
- translational attribution figures when present in archived outputs

For rebuilding only the manuscript figure package from an already completed results root, use:

```bash
python experiments_pbc7.py --emit_latex_only
```

To rebuild from a non-default results root or write to a custom package directory:

```bash
python experiments_pbc7.py --emit_latex_only --results_parent experiments_results --overleaf_dir overleaf_package
```

---

## Reproducibility and design notes

This repository is structured to support rigorous, repeatable experimentation.

Key implementation safeguards include:

- deterministic per-track working directories
- validation of required scripts before execution
- strict folder and label integrity checks
- horizon-aware validation of training labels
- contiguity checks across simulations and days
- frozen-test restoration before each canonical training run
- optional graph-folder figure generation before cleanup
- automatic GraphML cleanup after conversion to control storage growth
- selective GraphML retention when `--keep_step_train_graphml` is used
- archiving of step-specific training outputs
- archiving of tuning outputs when `--tune_step4` or `--tune_step8` is enabled
- optional archiving of the exact train/test folders used in each experiment
- evaluation-only repair through `--no_train`
- dataset-reuse retraining through `--no_simulation`

Two practical points are worth noting:

1. The pipeline writes into a deterministic results layout rather than timestamped run folders.
2. The `--timestamped` and `--no_timestamp` flags are accepted for compatibility, but pbc7 still uses the deterministic `--results_parent` layout.
3. The experimental workflow includes cleanup logic for transient artifacts to keep large runs manageable in disk usage.

---

## Citation

If you use this repository, please cite the associated manuscript.

---

## License

Add the repository license here.
