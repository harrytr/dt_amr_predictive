#!/usr/bin/env python3
"""Minimal, opt-in repetition of the T=7/H=7 predictive experiments.

Invoked by experiments_pbc7.py --predictive_repeats. See README_PREDICTIVE_REPEATS.md.
Legacy experiments, results and datasets are never modified by this workflow.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager

VERSION = "1.0.0"
METRICS = ("auroc", "accuracy", "macro_f1", "sensitivity", "specificity")
SPLITS = ("train", "validation", "test")
MODES = ("ground_truth", "partial_observation")
STEPS = ("step4_baseline", "step8_distribution_shift", "step9_baseline_to_shift_test")
FACTORS = ("resistant_transmission", "resistant_admission_weights", "turnover")
MODEL_KEYS = ("task", "T", "sliding_step", "hidden", "heads", "dropout", "transformer_layers",
              "sage_layers", "use_cls", "batch_size", "epochs", "lr", "max_neighbors",
              "neighbor_sampling", "num_neighbors", "seed_count", "seed_strategy",
              "seed_batch_size", "max_sub_batches", "early_stopping", "patience", "min_delta",
              "save_best_only", "lr_scheduler_on_plateau", "lr_scheduler_factor",
              "lr_scheduler_patience", "lr_scheduler_min_lr", "split_seed")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def run_lock(root):
    # OS-owned lock: automatically released after a crash; no stale lock files.
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(f"Another process is using {root}")
    try:
        yield
    finally:
        handle.close()


def run_command(command, repo, log, seed=0, deterministic=True):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["MPLBACKEND"] = "Agg"
    # Explicit CLI below owns simulator/converter settings; ambient legacy
    # environment variables must not redirect outputs or change observation mode.
    for key in ("DT_PT_OUT_DIR", "DT_STATE_MODE", "DT_SIM_EXTRA_ARGS", "DT_CONVERT_EXTRA_ARGS"):
        env.pop(key, None)
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[run] {shlex.join([str(x) for x in command])}\n      log: {log}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nCOMMAND " + shlex.join([str(x) for x in command]) + "\n")
        handle.flush()
        proc = subprocess.Popen([str(x) for x in command], cwd=repo, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"[running] {log.name}; follow {log}", flush=True)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}). Inspect {log}; rerun the same launch command after fixing the cause.")


def validate_config(cfg, driver):
    required = {"simulation_seed_offsets", "model_seeds", "parameter_seed", "parameter_multipliers",
                "parameter_rationale", "split_seed", "tracks", "include_xgboost", "bootstrap_replicates",
                "bootstrap_seed", "deterministic", "threads", "loader_workers", "conversion_workers",
                "keep_raw_graphml"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing explicit repeat configuration: {sorted(missing)}")
    unknown = set(cfg) - required - {"draws", "model_overrides", "description"}
    if unknown:
        raise ValueError(f"Unknown repeat configuration keys: {sorted(unknown)}")
    n = len(cfg["model_seeds"])
    if n < 2 or len(cfg["simulation_seed_offsets"]) != n:
        raise ValueError("At least two repeats are required; model_seeds and simulation_seed_offsets must have equal lengths")
    for key in ("model_seeds", "simulation_seed_offsets"):
        if len(set(cfg[key])) != n or any(type(x) is not int or x < 0 for x in cfg[key]):
            raise ValueError(f"{key} must contain distinct nonnegative integers")
    if not cfg["tracks"] or len(set(cfg["tracks"])) != len(cfg["tracks"]) or set(cfg["tracks"]) - set(MODES):
        raise ValueError("tracks must contain ground_truth and/or partial_observation once each")
    if cfg["include_xgboost"] and "ground_truth" not in cfg["tracks"]:
        raise ValueError("The existing XGBoost comparator is ground-truth only")
    for key in ("include_xgboost", "deterministic", "keep_raw_graphml"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    for mode, steps in cfg.get("model_overrides", {}).items():
        if mode not in cfg["tracks"] or set(steps) - set(STEPS[:2]):
            raise ValueError("model_overrides must name a selected track and Step 4 or Step 8")
    if any(seed + 1000000 >= 2**32 for seed in cfg["model_seeds"]):
        raise ValueError("Model and derived evaluation seeds must fit in uint32")
    if not str(cfg["parameter_rationale"]).strip():
        raise ValueError("State the source or rationale for the parameter draw ranges")
    if set(cfg["parameter_multipliers"]) != set(FACTORS):
        raise ValueError(f"parameter_multipliers must explicitly specify {FACTORS}")
    for key, bounds in cfg["parameter_multipliers"].items():
        if len(bounds) != 2 or not 0 < float(bounds[0]) < float(bounds[1]) or not all(math.isfinite(float(x)) for x in bounds):
            raise ValueError(f"{key}: require finite positive [low, high] with low < high")
    if "draws" in cfg:
        if len(cfg["draws"]) != n:
            raise ValueError("Explicit draws must have one entry per repeat")
        for row in cfg["draws"]:
            if set(row) != set(FACTORS) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in row.values()):
                raise ValueError("Each explicit draw must specify all three positive finite multipliers")
    for key in ("threads", "bootstrap_replicates"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("loader_workers", "conversion_workers", "parameter_seed", "bootstrap_seed", "split_seed"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if driver.CONFIG["MODEL"]["task"] != "endogenous_importation_majority_h7" or int(driver.CONFIG["MODEL"]["T"]) != 7:
        raise ValueError("Repeat mode is scoped to the submitted T=7/H=7 prediction task")
    if int(driver.CONFIG["SIM"]["num_regions"]) != 1:
        raise ValueError("Repeat mode preserves the submitted one-region trajectory identity convention")


def model_settings(driver, cfg, settings_root, xgb):
    settings, sources = {}, {}
    for mode in cfg["tracks"]:
        settings[mode] = {}
        for step in STEPS[:2]:
            model = copy.deepcopy(driver.CONFIG["MODEL"])
            model["split_seed"] = cfg["split_seed"]
            if settings_root is not None:
                archive = settings_root / f"TRACK_{mode}" / "work" / "repro_artifacts_steps_1_9"
                path = xgb.original_metrics_path(archive, step)
                if path is None:
                    raise FileNotFoundError(f"No unambiguous T=7/H=7 model settings for {mode}/{step} in {archive}")
                saved = read_json(path)["config"]
                for key in MODEL_KEYS:
                    if key in saved:
                        model[key] = saved[key]
                sources[f"{mode}/{step}"] = {"path": str(path), "sha256": file_hash(path)}
            overrides = cfg.get("model_overrides", {}).get(mode, {}).get(step, {})
            if set(overrides) - set(MODEL_KEYS):
                raise ValueError(f"Unsupported model override: {set(overrides) - set(MODEL_KEYS)}")
            model.update(overrides)
            if model["task"] != "endogenous_importation_majority_h7" or model["T"] != 7 or model["sliding_step"] != 1:
                raise ValueError("Repeat mode must retain the submitted task, T=7 and sliding_step=1")
            if isinstance(model["num_neighbors"], list):
                model["num_neighbors"] = ",".join(map(str, model["num_neighbors"]))
            # Settings above are now frozen rather than re-resolved from task defaults.
            model["use_task_hparams"] = False
            model.update(evaluate_train=True, export_predictions=True, metrics_only=True,
                         fullgraph_attribution_pass=False, emit_translational_figures=False,
                         deterministic=cfg["deterministic"])
            settings[mode][step] = model
    return settings, sources


def cli_values(argv):
    out, i = {}, 0
    while i < len(argv):
        key = argv[i]
        if not key.startswith("--"):
            raise ValueError(f"Invalid simulator argument: {key}")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[key[2:]] = argv[i + 1]
            i += 2
        else:
            out[key[2:]] = True
            i += 1
    return out


def build_plan(driver, cfg, sources):
    repeats, all_seeds = [], set()
    for r, (offset, model_seed) in enumerate(zip(cfg["simulation_seed_offsets"], cfg["model_seeds"]), 1):
        draw_seed = int(cfg["parameter_seed"]) + r
        rng = random.Random(draw_seed)
        factors = cfg["draws"][r - 1] if "draws" in cfg else {
            key: rng.uniform(*map(float, cfg["parameter_multipliers"][key])) for key in FACTORS}
        sim_cfg = copy.deepcopy(driver.CONFIG["SIM"])
        sim_cfg["beta_res_mult"] = float(sim_cfg.get("beta_res_mult", 1.0)) * factors["resistant_transmission"]
        sim_cfg["export_yaml"], sim_cfg["export_gif"] = True, False
        global_args = shlex.split(driver._build_sim_extra_args_global_only(sim_cfg))
        jobs = []
        for regime in ("canonical", "shift"):
            section = driver.CONFIG["STEP1" if regime == "canonical" else "STEP8"]
            specs = section["trajectories"] if regime == "canonical" else {
                **section["train_trajectories"], **section["test_trajectories"]}
            n_sims = int(section["n_sims_per_trajectory"])
            for family, original in specs.items():
                spec = copy.deepcopy(original)
                # Only resistant admission weights CR/IR are perturbed. Scaling
                # all four equally would cancel under normalization in import-high.
                for key in ("p_admit_import_cr", "p_admit_import_ir"):
                    spec[key] = float(original.get(key, original["p_admit_import_cr"])) * factors["resistant_admission_weights"]
                spec["daily_discharge_frac"] *= factors["turnover"]
                if not 0 <= spec["daily_discharge_frac"] <= 1:
                    raise ValueError("A turnover draw lies outside [0, 1]; narrow the explicit ranges")
                split = "train" if (family.endswith("_train") if regime == "canonical" else family in section["train_trajectories"]) else "test"
                for index in range(n_sims):
                    seed = int(spec["seed_base"]) + offset + index
                    if seed >= 2**32 or seed in all_seeds:
                        raise ValueError(f"Overlapping/out-of-range simulator seed {seed}; revise the seed offsets")
                    all_seeds.add(seed)
                    argv = ["--seed", str(seed), "--num_days", str(section["num_days"]),
                            "--daily_discharge_frac", str(spec["daily_discharge_frac"]),
                            "--daily_discharge_min_per_ward", str(spec["daily_discharge_min_per_ward"])]
                    for key, value in zip(("cs", "cr", "is", "ir"), driver._admission_import_weights_from_spec(spec)):
                        argv += [f"--p_admit_import_{key}", str(value)]
                    argv += global_args + list(map(str, spec.get("extra_sim_args", [])))
                    values = cli_values(argv)
                    if not math.isclose(float(values["beta_res_mult"]), sim_cfg["beta_res_mult"]):
                        raise ValueError("A per-family argument overrides the requested transmission draw")
                    job = {"family": family, "index": index, "sim_id": f"{family}__sim_{index:03d}",
                           "regime": regime, "split": split, "seed": seed, "days": int(section["num_days"]),
                           "argv": argv, "values": values, "generator_sha256": sources["generate_amr_data.py"]}
                    job["fingerprint"] = digest(job)
                    jobs.append(job)
        repeats.append({"repeat": r, "model_seed": model_seed, "eval_seed": model_seed + 1000000,
                        "parameter_seed": draw_seed, "parameter_multipliers": factors, "jobs": jobs})
    return repeats


def raw_complete(folder, job):
    expected = {f"amr_r0_t{day}.graphml" for day in range(1, job["days"] + 1)}
    paths = list(folder.glob("*.graphml"))
    return {p.name for p in paths} == expected and all(p.stat().st_size > 0 for p in paths)


def same_value(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def metadata_matches(meta, job):
    # Match every effective explicitly supplied scientific setting. Missing
    # metadata is not permission to assume a data directory has the right seed.
    merged = {**meta.get("model_params", {}), **meta.get("cli_args", {})}
    params = meta.get("model_params", {})
    for key, value in job["values"].items():
        if key in {"export_yaml", "no_export_gif"}:
            continue
        observed = merged.get(key)
        if observed is None:
            observed = params.get(key)
        if observed is None or not same_value(observed, value):
            return False
    return True


class ReuseIndex:
    """Read-only search of verified repeat caches and legacy live/archive data."""
    def __init__(self, roots):
        self.roots, self.repeat_entries, self.legacy = roots, {}, []
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Reuse directory not found: {root}")
            for marker in root.rglob("simulation.json"):
                try:
                    record = read_json(marker)
                    self.repeat_entries.setdefault(record["job"]["fingerprint"], []).append(marker.parent)
                except (ValueError, KeyError):
                    continue
            metadata_paths = sorted(root.rglob("run_metadata*.yaml"))
            if metadata_paths:
                try:
                    import yaml
                except ImportError as exc:
                    raise RuntimeError("PyYAML is required to verify legacy simulation metadata (pip install pyyaml)") from exc
                for path in metadata_paths:
                    try:
                        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
                        if isinstance(meta, dict):
                            self.legacy.append((path.parent, meta))
                    except (ValueError, yaml.YAMLError):
                        continue
        self._matching = {}

    def matching_legacy(self, job):
        key = job["fingerprint"]
        if key not in self._matching:
            self._matching[key] = [(folder, meta) for folder, meta in self.legacy if metadata_matches(meta, job)]
        return self._matching[key]

    def raw_source(self, job):
        for folder in self.repeat_entries.get(job["fingerprint"], []):
            raw = folder / "raw"
            if raw_complete(raw, job):
                return raw, "verified_repeat_manifest"
        for folder, _ in self.matching_legacy(job):
            if raw_complete(folder, job):
                return folder, "legacy_parameter_and_seed_metadata; generator_source_hash_unavailable"
        return None

    def pt_sources(self, job, mode):
        for folder in self.repeat_entries.get(job["fingerprint"], []):
            yield folder / mode, None, "verified_repeat_manifest"
        for raw, _ in self.matching_legacy(job):
            # Original flat/pool filenames contain the family and simulation id.
            family, sim = raw.parent.name, raw.name
            if not sim.startswith("sim_"):
                continue
            work = raw.parent.parent
            track = next((p for p in raw.parents if p.name == f"TRACK_{mode}"), None)
            label_manifest = raw / "labels" / "manifest.json"
            if label_manifest.is_file() and read_json(label_manifest).get("state_mode") == mode:
                yield raw, None, "legacy_parameter_and_conversion_metadata"
            if track is None:
                continue
            token = f"{family}__{sim}__"
            yield work / f"{family}_pt_flat", token, "legacy_parameter_metadata_and_track"
            live = ["synthetic_amr_graphs_train", "synthetic_amr_graphs_test_frozen"] if job["regime"] == "canonical" else ["synthetic_step8_shift_train", "synthetic_step8_shift_test"]
            for name in live:
                yield work / name, token, "legacy_parameter_metadata_and_track"
            archive = work / "repro_artifacts_steps_1_9"
            tags = STEPS[:1] if job["regime"] == "canonical" else STEPS[1:]
            for tag in tags:
                for split_dir in (archive / tag).rglob("*_folder"):
                    if split_dir.name in {"train_folder", "test_folder"}:
                        yield split_dir, token, "legacy_parameter_metadata_and_track_archive"
            pt_copies = archive / "pt_copies"
            if pt_copies.exists():
                tokens = [token]
                # Native converter archives retain a path-hashed run tag rather
                # than the flattened family prefix. Recover that tag from its
                # conversion manifest or the simulator's recorded absolute path.
                if label_manifest.is_file():
                    run_tag = read_json(label_manifest).get("run_tag")
                    if run_tag:
                        tokens.append(str(run_tag) + "__")
                for _, meta in self.matching_legacy(job):
                    original_path = str(meta.get("cli_args", {}).get("output_dir", ""))
                    if Path(original_path).is_absolute():
                        base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_path).name).strip("-")
                        tokens.append(base + "__" + hashlib.sha1(original_path.encode()).hexdigest()[:10] + "__")
                for sub in [pt_copies, *[p for p in pt_copies.rglob("*") if p.is_dir()]]:
                    for archive_token in dict.fromkeys(tokens):
                        yield sub, archive_token, "legacy_parameter_metadata_and_track_archive"


def pt_files(folder, token=None):
    return sorted(p for p in folder.glob("*.pt") if token is None or token in p.name)


def semantic_hash(data):
    import torch
    h = hashlib.sha256()
    keys = {"x", "edge_index", "edge_attr", "node_names", "sim_id", "day"}
    keys.update(k for k in data.keys() if k.startswith("y_"))
    for key in sorted(keys):
        value = getattr(data, key, None)
        h.update(key.encode())
        if torch.is_tensor(value):
            arr = value.detach().cpu().contiguous().numpy()
            h.update(str((arr.dtype, arr.shape)).encode())
            h.update(arr.tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def pt_identity(job, mode, converter_hash):
    return {"job": job["fingerprint"], "mode": mode, "converter_sha256": converter_hash}


def completed_pt(folder, identity, days):
    marker = folder / "pt_complete.json"
    if not marker.is_file():
        return False
    record = read_json(marker)
    if record.get("identity") != identity or len(record.get("files", [])) != days:
        return False
    paths = {p.name: p for p in folder.glob("*.pt")}
    if set(paths) != {row["name"] for row in record["files"]}:
        return False
    for row in record["files"]:
        st = paths[row["name"]].stat()
        if st.st_size != row["bytes"]:
            return False
        if st.st_mtime_ns != row["mtime_ns"] and file_hash(paths[row["name"]]) != row["sha256"]:
            return False
    return True


def import_pt(source, dest, token, job, mode, converter_hash, provenance):
    import torch
    files = pt_files(source, token)
    if len(files) != job["days"]:
        return False
    loaded, seen = [], set()
    for path in files:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            day = int(data.day)
            if day in seen or not 1 <= day <= job["days"] or int(data.y_h7_endog_majority) not in (0, 1):
                return False
            if getattr(data, "node_names", None) is None or getattr(data, "x", None) is None:
                return False
            if data.x.ndim != 2 or data.x.shape[1] != 6 or not bool(torch.isfinite(data.x).all()):
                return False
            if len(data.node_names) != data.x.shape[0] or len(set(data.node_names)) != len(data.node_names):
                return False
            if mode == "partial_observation":
                if not bool(((data.x[:, 1] == 0) | (data.x[:, 1] == 1)).all()) or bool((data.x[:, 4:6] != 0).any()):
                    return False
            for key in ("y_h7_trans_res", "y_h7_select_res", "y_h7_import_res"):
                if getattr(data, key, None) is None:
                    return False
        except (OSError, ValueError, AttributeError, RuntimeError, EOFError):
            return False
        seen.add(day)
        loaded.append((day, data))
    # Write the full trajectory atomically, after validating every daily graph.
    temp = dest.with_name(dest.name + ".building")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    rows = []
    for day, data in sorted(loaded, key=lambda pair: pair[0]):
        data.sim_id = job["sim_id"]
        path = temp / f"{job['sim_id']}__amr_r0_t{day}.pt"
        torch.save(data, path)
        st = path.stat()
        rows.append({"name": path.name, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                     "sha256": file_hash(path), "semantic_sha256": semantic_hash(data), "day": day})
    atomic_json(temp / "pt_complete.json", {"identity": pt_identity(job, mode, converter_hash),
                "files": rows, "semantic_sha256": digest([(r["day"], r["semantic_sha256"]) for r in rows]),
                "provenance": {"source": str(source), "verification": provenance}})
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(temp, dest)
    return True


def copy_raw(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.graphml"):
        shutil.copy2(path, dest / path.name)
    for path in source.glob("run_metadata*.yaml"):
        shutil.copy2(path, dest / path.name)


def cache_folder(output, job):
    return output / "data_cache" / job["fingerprint"][:24]


def ensure_simulation(job, cfg, output, repo, index, converter_hash):
    folder = cache_folder(output, job)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "simulation.json", {"job": job})
    missing = []
    for mode in cfg["tracks"]:
        target = folder / mode
        identity = pt_identity(job, mode, converter_hash)
        if completed_pt(target, identity, job["days"]):
            continue
        recovered = False
        for source, token, provenance in index.pt_sources(job, mode):
            if source == target:
                continue
            if provenance == "verified_repeat_manifest" and not completed_pt(source, identity, job["days"]):
                continue
            if import_pt(source, target, token, job, mode, converter_hash, provenance):
                print(f"[reuse PT] {job['sim_id']} {mode}: {source}", flush=True)
                recovered = True
                break
        if not recovered:
            missing.append(mode)
    if not missing:
        return folder
    raw = folder / "raw"
    raw_ready = folder / "raw_complete.json"
    if not (raw_complete(raw, job) and raw_ready.is_file() and read_json(raw_ready).get("job") == job["fingerprint"]):
        if raw.exists():
            shutil.rmtree(raw)  # Only this repeat's incomplete local cache, never the source.
        source = index.raw_source(job)
        if source:
            copy_raw(source[0], raw)
            provenance = {"source": str(source[0]), "verification": source[1]}
        else:
            raw.mkdir(parents=True)
            run_command([sys.executable, repo / "generate_amr_data.py", "--output_dir", raw, *job["argv"]],
                        repo, output / "logs" / f"sim_{job['seed']}.log", job["seed"])
            provenance = {"source": "generated", "generator_sha256": job["generator_sha256"]}
        if not raw_complete(raw, job):
            raise RuntimeError(f"Incomplete simulation {raw}; refusing to pool partial trajectories")
        atomic_json(raw_ready, {"job": job["fingerprint"], "provenance": provenance})
    for mode in missing:
        # The converter writes primary PTs next to raw files. Clear only these
        # scratch PTs, convert one observation mode, then preserve it separately.
        for path in raw.glob("*.pt"):
            path.unlink()
        command = [sys.executable, repo / "convert_to_pt.py", "--graphml_dir", raw,
                   "--label_csv_dir", raw / "labels", "--keep_graphml", "--state_mode", mode,
                   "--horizons", "7,14", "--workers", str(cfg["conversion_workers"])]
        run_command(command, repo, output / "logs" / f"convert_{job['seed']}_{mode}.log", job["seed"])
        if not import_pt(raw, folder / mode, None, job, mode, converter_hash, "converted_matching_simulation"):
            raise RuntimeError(f"Incomplete/invalid converted trajectory in {raw}")
        for path in raw.glob("*.pt"):
            path.unlink()
    if not cfg["keep_raw_graphml"]:
        for path in raw.glob("*.graphml"):
            path.unlink()
    return folder


def link_or_copy(source, destination):
    # Hardlinks avoid multiplying large PT datasets; consumers never edit PTs.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def pool_datasets(rep, cfg, output, driver, converter_hash):
    root = output / f"repeat_{rep['repeat']:03d}"
    datasets = {}
    for mode in cfg["tracks"]:
        work = root / f"TRACK_{mode}" / "work"
        datasets[mode] = {}
        for regime, split, rel in (("canonical", "train", driver.BASELINE_TRAIN_REL),
                                   ("canonical", "test", driver.BASELINE_TEST_REL),
                                   ("shift", "train", driver.STEP8_TRAIN_REL),
                                   ("shift", "test", driver.STEP8_TEST_REL)):
            folder = work / rel
            members, sources = [], []
            for job in rep["jobs"]:
                if (job["regime"], job["split"]) != (regime, split):
                    continue
                source = cache_folder(output, job) / mode
                if not completed_pt(source, pt_identity(job, mode, converter_hash), job["days"]):
                    raise RuntimeError(f"Unverified or incomplete cache: {source}")
                info = read_json(source / "pt_complete.json")
                members.append({"sim_id": job["sim_id"], "semantic_sha256": info["semantic_sha256"]})
                sources.extend((source / row["name"], row) for row in info["files"])
            identity = digest(members)
            marker = folder / "dataset_complete.json"
            expected_names = {p.name for p, _ in sources}
            if len(expected_names) != len(sources):
                raise RuntimeError("Duplicate daily graph names while pooling")
            current = read_json(marker) if marker.is_file() else {}
            complete = current.get("identity") == identity and {p.name for p in folder.glob("*.pt")} == expected_names
            if complete:
                for path, row in sources:
                    target = folder / path.name
                    if target.stat().st_size != row["bytes"] or (target.stat().st_mtime_ns != row["mtime_ns"] and file_hash(target) != row["sha256"]):
                        complete = False
                        break
            if not complete:
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True)
                for source, _ in sources:
                    link_or_copy(source, folder / source.name)
                atomic_json(marker, {"identity": identity, "trajectories": members, "daily_graphs": len(sources)})
            datasets[mode][f"{regime}_{split}"] = {"folder": str(folder), "identity": identity}
    return root, datasets


def verify_transfer_predictions(folder, source):
    import numpy as np
    for split in ("train", "validation"):
        with (folder / f"{split}_predictions.csv").open(newline="") as handle:
            evaluated = list(csv.DictReader(handle))
        with (source / f"{split}_predictions.csv").open(newline="") as handle:
            canonical = list(csv.DictReader(handle))
        keys = ("sim_id", "day", "y", "prediction")
        if [[r[k] for k in keys] for r in evaluated] != [[r[k] for k in keys] for r in canonical]:
            raise RuntimeError(f"Transfer changed canonical {split} windows/predictions: {folder}")
        if not np.allclose([float(r["probability"]) for r in evaluated],
                           [float(r["probability"]) for r in canonical], rtol=1e-6, atol=1e-8):
            raise RuntimeError(f"Transfer changed canonical {split} probabilities: {folder}")


def task_complete(folder, fingerprint, required):
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    saved = read_json(marker)
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Completed task has different inputs/settings: {folder}. Use a new output directory.")
    return all((folder / name).is_file() and file_hash(folder / name) == saved.get("files", {}).get(name)
               for name in required)


def mark_complete(folder, fingerprint, required):
    atomic_json(folder / "complete.json", {"fingerprint": fingerprint,
                "files": {name: file_hash(folder / name) for name in required}})


def check_predictions(folder):
    ids = {}
    for split in SPLITS:
        with (folder / f"{split}_predictions.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Empty {split} predictions: {folder}")
        keys = {(row["sim_id"], int(row["day"])) for row in rows}
        if len(keys) != len(rows):
            raise RuntimeError(f"Duplicate {split} windows: {folder}")
        ids[split] = {row["sim_id"] for row in rows}
        if split == "train" and {int(row["y"]) for row in rows} != {0, 1}:
            raise RuntimeError(f"Training partition has only one class: {folder}. This failure is not redrawn or silently omitted.")
    if ids["train"] & ids["validation"] or (ids["train"] | ids["validation"]) & ids["test"]:
        raise RuntimeError(f"Trajectory leakage in {folder}")


def run_gnn(rep, cfg, datasets, root, output, repo, driver, settings):
    directories = {}
    for mode in cfg["tracks"]:
        directories[mode] = {}
        for step in STEPS:
            transfer = step == STEPS[2]
            source_step = STEPS[0] if transfer else step
            model = copy.deepcopy(settings[mode][source_step])
            model.update(model_seed=rep["model_seed"], eval_seed=rep["eval_seed"])
            train = datasets[mode]["shift_train" if step == STEPS[1] else "canonical_train"]
            test = datasets[mode]["canonical_test" if step == STEPS[0] else "shift_test"]
            folder = root / "gnn" / mode / step
            checkpoint = directories[mode][STEPS[0]] / "trained_model.pt" if transfer else None
            fingerprint = digest({"train": train["identity"], "test": test["identity"], "model": model,
                                  "checkpoint": file_hash(checkpoint) if checkpoint else None})
            required = ["metrics_summary.json", "trained_model.pt", *[f"{s}_predictions.csv" for s in SPLITS]]
            if not task_complete(folder, fingerprint, required):
                folder.mkdir(parents=True, exist_ok=True)
                if transfer:
                    shutil.copy2(checkpoint, folder / "trained_model.pt")
                command = driver._build_train_args_from_config(
                    model, sys.executable, Path(train["folder"]), out_dir=str(folder),
                    test_folder=Path(test["folder"]), train_model_override=not transfer)
                # Repeat-only runtime controls, independent of scientific settings.
                for flag, value in (("--num_workers", cfg["loader_workers"]),
                                    ("--torch_num_threads", cfg["threads"]),
                                    ("--torch_num_interop_threads", min(4, cfg["threads"]))):
                    command[command.index(flag) + 1] = str(value)
                run_command(command, repo, output / "logs" / f"repeat_{rep['repeat']:03d}_{mode}_{step}.log",
                            rep["model_seed"], cfg["deterministic"])
                check_predictions(folder)
                if transfer and file_hash(folder / "trained_model.pt") != file_hash(checkpoint):
                    raise RuntimeError("Step 9 modified its canonical checkpoint")
                if transfer:
                    verify_transfer_predictions(folder, directories[mode][STEPS[0]])
                mark_complete(folder, fingerprint, required)
            else:
                print(f"[resume GNN] repeat {rep['repeat']} {mode} {step}", flush=True)
            directories[mode][step] = folder
    return directories


def run_xgb(rep, cfg, datasets, root, gnn, xgb):
    if not cfg["include_xgboost"]:
        return {}
    mode, data = "ground_truth", {}
    for key, item in datasets[mode].items():
        folder = root / "xgb_features" / key
        required = ["features.npz", "samples.json", "source_files.json"]
        fingerprint = digest({"data": item["identity"], "mode": mode, "xgb_version": xgb.VERSION})
        if not task_complete(folder, fingerprint, required):
            folder.mkdir(parents=True, exist_ok=True)
            source = Path(item["folder"])
            xgb.extract_dataset(source, mode, xgb.input_manifest(source), folder)
            mark_complete(folder, fingerprint, required)
        data[key] = xgb.load_dataset(folder)
    params = xgb.xgb_parameters(rep["model_seed"], cfg["threads"])
    completed = {}
    for step in STEPS:
        for variant in ("all_summaries", "without_counts"):
            transfer = step == STEPS[2]
            count_ablation = variant == "without_counts"
            tag = ({STEPS[0]: "step5_no_admission_turnover", STEPS[1]: "step8_no_admission_turnover",
                    STEPS[2]: "step9_no_admission_turnover"}[step] if count_ablation else step)
            gnn_summary = gnn[mode][step] / "metrics_summary.json"
            plan = {"mode": mode, "tag": tag, "label": f"{step}: {variant}",
                    "train_folder": datasets[mode]["shift_train" if step == STEPS[1] else "canonical_train"]["folder"],
                    "test_folder": datasets[mode]["canonical_test" if step == STEPS[0] else "shift_test"]["folder"],
                    "split_seed": read_json(gnn_summary)["config"]["split_seed"],
                    "exclude_admission_turnover": count_ablation, "original_metrics": str(gnn_summary),
                    "reference_step": step, "transfer_source_step": STEPS[0] if transfer else None}
            td = xgb.apply_feature_selection(data["shift_train" if step == STEPS[1] else "canonical_train"], plan)
            ed = xgb.apply_feature_selection(data["canonical_test" if step == STEPS[0] else "shift_test"], plan)
            ti, vi = xgb.original_split(td[2], plan["split_seed"])
            parity = xgb.verify_original_labels(plan, td[1][vi], ed[1])
            fit_fp = digest({"arrays": xgb.array_hash(td[0], td[1], ti, vi), "parameters": params})
            source = completed[(STEPS[0], variant)] / "model.json" if transfer else None
            fingerprint = digest({"plan": plan, "fit": fit_fp, "test": xgb.array_hash(ed[0], ed[1]),
                                  "source": file_hash(source) if source else None})
            folder = root / "xgb" / step / variant
            required = ["model.json", "metrics.json", *[f"{s}_predictions.csv" for s in SPLITS]]
            if not task_complete(folder, fingerprint, required):
                folder.mkdir(parents=True, exist_ok=True)
                xgb.run_experiment(folder, plan, td, ed, params, cfg["bootstrap_replicates"],
                                   fit_fp, None, source, parity)
                check_predictions(folder)
                if transfer and file_hash(folder / "model.json") != file_hash(source):
                    raise RuntimeError("XGBoost transfer modified the canonical checkpoint")
                if transfer:
                    verify_transfer_predictions(folder, completed[(STEPS[0], variant)])
                mark_complete(folder, fingerprint, required)
            completed[(step, variant)] = folder
    return completed


def preflight(repeats, cfg, output, index, converter_hash):
    counts, details = Counter(), []
    for rep in repeats:
        for job in rep["jobs"]:
            folder, needed = cache_folder(output, job), []
            for mode in cfg["tracks"]:
                ident = pt_identity(job, mode, converter_hash)
                if completed_pt(folder / mode, ident, job["days"]):
                    counts["ready_cached_trajectories"] += 1
                    continue
                candidates = [(p, token) for p, token, how in index.pt_sources(job, mode)
                              if len(pt_files(p, token)) == job["days"] and
                              (how != "verified_repeat_manifest" or completed_pt(p, ident, job["days"]))]
                if candidates:
                    counts["candidate_PT_reuse_requires_content_validation"] += 1
                else:
                    needed.append(mode)
            if needed:
                cached_raw = raw_complete(folder / "raw", job) and (folder / "raw_complete.json").exists()
                action = "convert_existing_raw" if cached_raw or index.raw_source(job) else "generate_then_convert"
                counts[action] += 1
                details.append({"repeat": rep["repeat"], "simulation": job["sim_id"], "seed": job["seed"],
                                "action": action, "modes": needed})
    return {"counts": dict(counts), "details": details,
            "gnn_fits_for_full_design": len(repeats) * len(cfg["tracks"]) * 2,
            "xgboost_fits_for_full_design": len(repeats) * 4 if cfg["include_xgboost"] else 0,
            "note": "Matching seeds, parameter settings and observation mode are required. Existing original-seed data cannot replace new independent simulations."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat_config", default=str(Path(__file__).with_name("predictive_repeats.json")))
    parser.add_argument("--repeat_output", default="predictive_repeat_results")
    parser.add_argument("--reuse_root", action="append", default=[], help="Read-only existing results/cache root; repeatable")
    parser.add_argument("--settings_root", help="Original results root from which to freeze Step 4/8 GNN hyperparameters")
    parser.add_argument("--repeat_ids", help="Optional comma-separated 1-based repeat IDs; same output resumes safely")
    parser.add_argument("--preflight", action="store_true", help="Plan data recovery and report dependency availability; no simulations/fits")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args(argv)
    repo, output = Path(__file__).resolve().parent, Path(args.repeat_output).expanduser().resolve()
    driver = load_module(repo / "experiments_pbc7.py", "_repeat_driver")
    xgb = load_module(repo / "experiments_xgboost.py", "_repeat_xgb")
    cfg = read_json(Path(args.repeat_config).expanduser())
    validate_config(cfg, driver)
    roots = [Path(p).expanduser().resolve() for p in args.reuse_root]
    settings_root = Path(args.settings_root).expanduser().resolve() if args.settings_root else None
    if settings_root and settings_root not in roots:
        roots.append(settings_root)
    for source in roots:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Repeat outputs must be separate from, and not nested in, a reused original results directory")
    settings, setting_sources = model_settings(driver, cfg, settings_root, xgb)
    sources = {p.name: file_hash(p) for p in sorted(repo.glob("*.py"))}
    repeats = build_plan(driver, cfg, sources)
    selected = set(range(1, len(repeats) + 1))
    if args.repeat_ids:
        selected = {int(v) for v in args.repeat_ids.split(",")}
        if not selected or not selected <= set(range(1, len(repeats) + 1)):
            raise ValueError("repeat_ids contains an unknown repeat")
    identity = {"version": VERSION, "config": cfg, "source_sha256": sources,
                "driver_config": driver.CONFIG, "model_settings": settings,
                "settings_sources": setting_sources, "repeats": repeats}
    identity_hash = digest(identity)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity_sha256"] != identity_hash:
        raise RuntimeError("This output directory belongs to a different configuration/source version. Select a new --repeat_output.")
    if args.summarize_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No run manifest: {manifest_path}")
        from summarize_predictive_repeats import summarize
        with run_lock(output):
            summarize(output)
        return 0
    index = ReuseIndex(roots)
    selected_repeats = [rep for rep in repeats if rep["repeat"] in selected]
    plan = preflight(selected_repeats, cfg, output, index, sources["convert_to_pt.py"])
    dependencies = ["numpy", "scipy", "sklearn", "torch", "torch_geometric", "networkx", "matplotlib", "yaml"]
    if cfg["include_xgboost"]:
        dependencies.append("xgboost")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    plan["missing_python_modules"] = missing
    print(json.dumps({k: v for k, v in plan.items() if k != "details"}, indent=2), flush=True)
    if args.preflight:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "preflight.json", {"plan": plan, "identity": identity})
        print(f"Preflight saved: {output / 'preflight.json'}")
        return 0
    if missing:
        raise RuntimeError(f"Use your existing AMR/PyTorch Geometric environment. Missing modules: {', '.join(missing)}")
    if cfg["include_xgboost"]:
        import xgboost as xgboost_package
    import torch
    torch.set_num_threads(cfg["threads"])
    with run_lock(output):
        atomic_json(manifest_path, {"identity_sha256": identity_hash, "identity": identity})
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in sources:
            shutil.copy2(repo / name, snapshot / name)
        atomic_json(snapshot / "repeat_configuration.json", cfg)
        from importlib import metadata
        versions = {}
        for name in ("numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "networkx", "xgboost", "PyYAML"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        runtime = {"python": sys.version, "packages": versions, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        runtime_path = output / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path) != runtime:
            raise RuntimeError("The software/device environment changed since this run started. Use a new output directory.")
        atomic_json(runtime_path, runtime)
        for rep in selected_repeats:
            print(f"\n[repeat {rep['repeat']}/{len(repeats)}] parameters={rep['parameter_multipliers']}", flush=True)
            for i, job in enumerate(rep["jobs"], 1):
                print(f"[data {i}/{len(rep['jobs'])}] {job['sim_id']} seed={job['seed']}", flush=True)
                ensure_simulation(job, cfg, output, repo, index, sources["convert_to_pt.py"])
            root, datasets = pool_datasets(rep, cfg, output, driver, sources["convert_to_pt.py"])
            atomic_json(root / "repeat_manifest.json", rep)
            gnn = run_gnn(rep, cfg, datasets, root, output, repo, driver, settings)
            run_xgb(rep, cfg, datasets, root, gnn, xgb)
            atomic_json(root / "repeat_complete.json", {"repeat": rep["repeat"], "identity_sha256": identity_hash})
            from summarize_predictive_repeats import summarize
            summarize(output)
    print(f"Complete. Tables: {output / 'summary'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
