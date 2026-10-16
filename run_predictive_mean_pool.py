#!/usr/bin/env python3
"""Matched seven-day mean-pooling ablation: predictive Steps 4 and 9 only.

Use the existing predictive_repeat_results/run_manifest.json and source_snapshot.
No simulation, conversion, Step 8 fitting, or Transformer refitting is performed.
The GraphSAGE encoder and classifier are trained end-to-end on canonical data.
Only positional encoding, the Transformer and its optional CLS token are removed.

Default: repeat 1, both available observation tracks (one canonical fit per track).
Use --repeat-ids all for the same five-repeat Student-t analysis as the reference.
A single fit instead supports the reference's conditional trajectory-bootstrap CIs.
Completed fits are reused. An interrupted, incomplete fit restarts from its seed.

Store this script OUTSIDE the original repository's top-level Python files while
predictive repeats are running: their manifest fingerprints that file inventory.
Run in the same conda environment used for the reference experiment. Example:

python ~/Downloads/run_predictive_mean_pool.py \
  --reference-root /Users/harrytriantafyllidis/AMR/predictive_repeat_results \
  --out-root /Users/harrytriantafyllidis/AMR/mean_pool_results --preflight

Re-run without --preflight to fit repeat 1, or add --repeat-ids all for all repeats.
--summarize-only refreshes comparisons after reference Step 9 results arrive.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION = "1.0.1"
STEP4 = "step4_baseline"
STEP9 = "step9_baseline_to_shift_test"
STEPS = (STEP4, STEP9)
TRACKS = ("ground_truth", "partial_observation")
SPLITS = ("train", "validation", "test")
METRICS = ("auroc", "accuracy", "macro_f1", "sensitivity", "specificity")
REQUIRED_SOURCES = (
    "models_amr.py", "train_amr_dygformer.py", "temporal_graph_dataset.py",
    "tasks.py", "predictive_repeats.py", "summarize_predictive_repeats.py",
)
# All effective scientific/training options recorded by the supplied trainer.
VALUE_OPTIONS = (
    "T", "sliding_step", "hidden", "heads", "dropout", "transformer_layers",
    "sage_layers", "batch_size", "epochs", "lr", "max_neighbors",
    "num_neighbors", "seed_count", "seed_strategy", "seed_batch_size",
    "max_sub_batches", "model_seed", "eval_seed", "split_seed", "patience",
    "min_delta", "lr_scheduler_factor", "lr_scheduler_patience", "lr_scheduler_min_lr",
)
BOOL_OPTIONS = (
    "neighbor_sampling", "deterministic", "early_stopping", "save_best_only",
    "lr_scheduler_on_plateau",
)
TASK_FILES = (
    "trained_model.pt", "metrics_summary.json", "train_predictions.csv",
    "validation_predictions.csv", "test_predictions.csv", "model_architecture.json",
)


class AuditError(RuntimeError):
    """A comparison cannot safely be treated as matched."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp, path)


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@contextmanager
def output_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "meanpool.lock").open("a+b")
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
    except OSError as exc:
        handle.close()
        raise AuditError(f"Another mean-pooling process is using {root}") from exc
    try:
        yield
    finally:
        handle.close()


def import_source(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AuditError(f"Cannot import source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_statistics(sources: Path) -> Any:
    sys.path.insert(0, str(sources))
    sys.dont_write_bytecode = True
    import_source(sources / "predictive_repeats.py", "predictive_repeats")
    stats = import_source(sources / "summarize_predictive_repeats.py", "_meanpool_reference_statistics")
    for name in ("binary_metrics", "mean_interval", "trajectory_bootstrap", "read_predictions", "verify_saved_metrics"):
        if not callable(getattr(stats, name, None)):
            raise AuditError(f"Reference summarizer lacks {name}; this runner needs adaptation.")
    return stats


def read_reference(root: Path) -> Tuple[Dict[str, Any], Dict[str, str]]:
    manifest = read_json(root / "run_manifest.json")
    identity = manifest["identity"]
    if digest(identity) != manifest["identity_sha256"]:
        raise AuditError("Reference run_manifest.json has an inconsistent identity hash.")
    sources = identity["source_sha256"]
    for name in REQUIRED_SOURCES:
        path = root / "source_snapshot" / name
        if not path.is_file() or sources.get(name) != file_hash(path):
            raise AuditError(f"Missing or changed frozen reference source: {path}")
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cfg = identity["config"]
    if int(cfg["bootstrap_replicates"]) < 1:
        raise AuditError("The reference must specify a positive bootstrap_replicates count.")
    return manifest, sources


def output_identity(ref: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": VERSION, "script_sha256": file_hash(Path(__file__)),
        "reference_root": str(ref), "reference_identity_sha256": manifest["identity_sha256"],
        "architecture": "GraphSAGE + equal mean of seven daily embeddings + original classifier",
        "steps": list(STEPS),
    }


def validate_output_target(root: Path, ref: Path, manifest: Dict[str, Any]) -> None:
    """Reject unsafe or unowned output paths BEFORE creating even a lock file."""
    if inside(root, ref) or inside(ref, root):
        raise AuditError("Reference and output directories must be separate, non-nested trees.")
    if not root.exists():
        return
    if not root.is_dir():
        raise AuditError(f"Output path is not a directory: {root}")
    # Never follow an existing output link into another result tree. Hard-linked
    # files would also permit writes to change a file in another directory.
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditError(f"Output contains a symbolic link; use a separate ordinary directory: {path}")
        if path.is_file() and path.stat().st_nlink > 1:
            raise AuditError(f"Output contains a hard-linked file; use an independent output copy: {path}")
    marker = root / "meanpool_manifest.json"
    if marker.exists():
        if read_json(marker).get("identity") != output_identity(ref, manifest):
            raise AuditError("Output belongs to a different source/configuration. Use a new --out-root.")
    elif any(path.name != "meanpool.lock" for path in root.iterdir()):
        raise AuditError("Refusing an unowned nonempty output folder; choose a new --out-root.")


def prepare_output(root: Path, ref: Path, manifest: Dict[str, Any], sources: Dict[str, str]) -> Path:
    validate_output_target(root, ref, manifest)
    identity = output_identity(ref, manifest)
    marker = root / "meanpool_manifest.json"
    if marker.exists():
        if read_json(marker)["identity"] != identity:
            raise AuditError("Output belongs to a different source/configuration. Use a new --out-root.")
    else:
        permitted = {"meanpool.lock"}
        if any(p.name not in permitted for p in root.iterdir()):
            raise AuditError("Refusing an unowned nonempty output folder; choose a new --out-root.")
        atomic_json(marker, {"identity": identity, "reference_manifest": manifest})
    destination = root / "sources"
    destination.mkdir(exist_ok=True)
    for name, expected in sources.items():
        # A flat, read-only copy of the exact sources used by the reference run.
        if Path(name).name != name or not name.endswith(".py"):
            raise AuditError(f"Unexpected snapshot filename: {name}")
        origin = ref / "source_snapshot" / name
        target = destination / name
        if not origin.is_file() or file_hash(origin) != expected:
            raise AuditError(f"Reference snapshot changed: {origin}")
        if target.exists():
            if file_hash(target) != expected:
                raise AuditError(f"Local frozen source was modified: {target}")
        else:
            shutil.copy2(origin, target)
    return destination


def reference_task(root: Path, repeat: int, track: str, step: str) -> Path:
    return root / f"repeat_{repeat:03d}" / "gnn" / track / step


def verify_reference_task(folder: Path) -> Optional[Dict[str, Any]]:
    marker = folder / "complete.json"
    if not marker.exists():
        return None
    obj = read_json(marker)
    required = {"metrics_summary.json", "trained_model.pt", *[f"{s}_predictions.csv" for s in SPLITS]}
    if not required <= set(obj.get("files", {})):
        raise AuditError(f"Incomplete reference completion record: {marker}")
    for name, expected in obj["files"].items():
        path = folder / name
        if Path(name).name != name or not path.is_file() or file_hash(path) != expected:
            raise AuditError(f"Completed reference result is missing or changed: {path}")
    return obj


def normal_value(value: Any) -> Any:
    if isinstance(value, str) and "," in value:
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            return value
    return value


def effective_config(summary: Dict[str, Any], identity: Dict[str, Any], rep: Dict[str, Any], track: str) -> Dict[str, Any]:
    cfg = dict(summary["config"])
    required = set(VALUE_OPTIONS) | set(BOOL_OPTIONS) | {"use_cls", "data_folder", "test_folder"}
    missing = required - set(cfg)
    if missing:
        raise AuditError(f"Missing effective reference settings (not guessed): {sorted(missing)}")
    if cfg["T"] != 7 or cfg["sliding_step"] != 1 or summary["task"] != "endogenous_importation_majority_h7":
        raise AuditError("This runner requires the supplied seven-day history/horizon predictive task.")
    if cfg["model_seed"] is None or cfg["eval_seed"] is None:
        raise AuditError("Unrecorded reference seeds cannot be reproduced; use the seeded predictive-repeat output.")
    for key in ("model_seed", "eval_seed"):
        if cfg[key] != rep[key]:
            raise AuditError(f"Reference {key} does not match its repeat manifest.")
    settings = identity["model_settings"][track][STEP4]
    for key in (*VALUE_OPTIONS, *BOOL_OPTIONS, "use_cls"):
        if key in settings and key not in ("model_seed", "eval_seed"):
            if normal_value(cfg[key]) != normal_value(settings[key]):
                raise AuditError(f"Saved effective setting {key} differs from the frozen design.")
    cfg["task"] = summary["task"]
    return cfg


def dataset_location(ref: Path, rep: Dict[str, Any], track: str, kind: str, config: Dict[str, Any]) -> Path:
    work = ref / f"repeat_{rep['repeat']:03d}" / f"TRACK_{track}" / "work"
    conventional = {
        "canonical_train": "synthetic_amr_graphs_train",
        "canonical_test": "synthetic_amr_graphs_test",
        "shift_test": "synthetic_step8_shift_test",
    }
    original = config.get("data_folder" if kind == "canonical_train" else "test_folder") if kind != "shift_test" else None
    choices = [Path(original).expanduser().resolve()] if original else []
    choices.append(work / conventional[kind])
    for path in choices:
        if inside(path, ref) and (path / "dataset_complete.json").is_file():
            return path.resolve()
    raise FileNotFoundError(f"Ready {track}/{kind} dataset not found under {work}")


def verify_dataset(ref: Path, rep: Dict[str, Any], track: str, kind: str,
                   folder: Path, full_hash: bool = False) -> Dict[str, Any]:
    """Match pooled files to original per-trajectory manifests without rewriting them."""
    regime, split = ("shift", "test") if kind == "shift_test" else ("canonical", kind.rsplit("_", 1)[1])
    jobs = [j for j in rep["jobs"] if (j["regime"], j["split"]) == (regime, split)]
    if not jobs:
        raise AuditError(f"No manifest jobs for {kind}")
    pooled = read_json(folder / "dataset_complete.json")
    members, expected_names, stamps = [], set(), []
    for job in jobs:
        path = ref / "data_cache" / job["fingerprint"][:24] / track / "pt_complete.json"
        info = read_json(path)
        records = info["files"]
        if len(records) != job["days"]:
            raise AuditError(f"Incomplete trajectory manifest: {path}")
        members.append({"sim_id": job["sim_id"], "semantic_sha256": info["semantic_sha256"]})
        for row in records:
            name = row["name"]
            if Path(name).name != name or name in expected_names:
                raise AuditError("Invalid/duplicate daily filename in source manifests.")
            expected_names.add(name)
            target = folder / name
            st = target.stat()
            if st.st_size != row["bytes"]:
                raise AuditError(f"Daily graph size mismatch: {target}")
            if full_hash or st.st_mtime_ns != row["mtime_ns"]:
                if file_hash(target) != row["sha256"]:
                    raise AuditError(f"Daily graph content mismatch: {target}")
            stamps.append((name, st.st_size, st.st_mtime_ns))
    if pooled["identity"] != digest(members) or pooled["trajectories"] != members:
        raise AuditError(f"Pooled dataset/trajectory identity mismatch: {folder}")
    if {p.name for p in folder.glob("*.pt")} != expected_names or pooled["daily_graphs"] != len(expected_names):
        raise AuditError(f"Missing or extra daily graphs: {folder}")
    return {"path": str(folder), "identity": pooled["identity"], "n_daily_graphs": len(expected_names),
            "n_trajectories": len(members), "file_stat_digest": digest(sorted(stamps)),
            "verification": "original manifest, file sizes/mtimes; hashes on changed mtimes" if not full_hash else "all original SHA256 hashes"}


def build_cli(config: Dict[str, Any], global_config: Dict[str, Any], train: Path, test: Path,
              output: Path, fit: bool) -> List[str]:
    args = ["--data_folder", str(train), "--test_folder", str(test), "--task", config["task"],
            "--out_dir", str(output), "--train_model", "true" if fit else "false"]
    for key in VALUE_OPTIONS:
        value = config[key]
        if isinstance(value, (tuple, list)):
            value = ",".join(map(str, value))
        args += ["--" + key, str(value)]
    for key in BOOL_OPTIONS:
        if type(config[key]) is not bool:
            raise AuditError(f"Expected recorded boolean setting for {key}")
        args += ["--" + key, str(config[key]).lower()]
    if config["use_cls"]:
        args.append("--use_cls")
    args += ["--num_workers", str(global_config["loader_workers"]),
             "--torch_num_threads", str(global_config["threads"]),
             "--torch_num_interop_threads", str(min(4, global_config["threads"])),
             "--evaluate_train", "true", "--export_predictions", "true", "--metrics_only", "true",
             "--fullgraph_attribution_pass", "false", "--emit_translational_figures", "false",
             "--require_pt_metadata", "true", "--fail_on_noncontiguous", "true"]
    # No --use_task_hparams: effective saved values have already been resolved.
    return args


def install_mean_pool(models: Any) -> Any:
    """Patch only the temporal module; keep encoder/head initialisation matched."""
    import torch
    import torch.nn.functional as F
    original = models.AMRDyGFormer

    class SevenDayMeanPool(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Construct the original model FIRST, consuming exactly its initial
            # random draws. Shared encoder and classifier start at the same
            # seeded values, rather than shifting classifier initialisation.
            super().__init__(*args, **kwargs)
            if self.T != 7:
                raise AuditError("Mean-pooling ablation requires seven daily embeddings.")
            required = ("gnn", "head", "pos_emb", "transformer")
            if not all(hasattr(self, key) for key in required):
                raise AuditError("Unexpected model structure; refusing a silent partial ablation.")
            self.reference_parameter_count = sum(p.numel() for p in self.parameters())
            for name in ("transformer", "pos_emb", "cls_token"):
                if hasattr(self, name):
                    delattr(self, name)
            self.use_cls_token = False

        def forward_from_day_embeddings(self, H: Any) -> Any:
            if H.ndim != 3 or H.shape[1] != self.T or H.shape[2] != self.hidden_channels:
                raise AuditError(f"Expected [batch,7,{self.hidden_channels}] embeddings, got {tuple(H.shape)}")
            out = self.head(H.mean(dim=1))
            act = getattr(self, "output_activation", "softplus" if self.use_softplus else "identity")
            if act == "softplus":
                return F.softplus(out)
            if act == "sigmoid":
                return torch.sigmoid(out)
            if act in ("identity", "none"):
                return out
            raise AuditError(f"Unknown output activation: {act}")

    models.AMRDyGFormer = SevenDayMeanPool
    return original


def protect_reference(root: Path) -> None:
    """Deny worker filesystem writes/deletions inside the original result tree."""
    def guarded(path: Any) -> bool:
        if isinstance(path, (str, bytes, os.PathLike)):
            return inside(Path(os.fsdecode(path)), root)
        return False

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        paths: Sequence[Any] = ()
        if event == "open":
            mode, flags = args[1], args[2]
            writable = (isinstance(mode, str) and any(c in mode for c in "wax+"))
            writable = writable or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            paths = args[:1] if writable else ()
        elif event in ("os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "shutil.rmtree"):
            paths = args[:1]
        elif event in ("os.rename", "os.replace"):
            paths = args[:2]
        elif event in ("os.link", "os.symlink"):
            paths = args[1:2]
        if any(guarded(path) for path in paths):
            raise PermissionError(f"Mean-pooling worker cannot modify reference files: {event} {paths}")
    sys.addaudithook(audit)


def validate_runtime(ref: Path) -> Dict[str, Any]:
    import torch
    from importlib import metadata
    observed = {"python": sys.version, "packages": {}, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    source = ref / "runtime.json"
    if not source.is_file():
        raise AuditError("Missing reference runtime.json; environment equivalence cannot be checked.")
    expected = read_json(source)
    relevant = ("numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "networkx", "PyYAML")
    differences = []
    for name in relevant:
        if name not in expected.get("packages", {}):
            continue
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = "not installed"
        observed["packages"][name] = version
        if version != expected["packages"][name]:
            differences.append(f"{name}: reference={expected['packages'][name]}, current={version}")
    if sys.version.split()[0] != expected["python"].split()[0]:
        differences.append("Python version differs")
    for name in ("device", "cuda"):
        if observed[name] != expected[name]:
            differences.append(f"{name}: reference={expected[name]}, current={observed[name]}")
    if differences:
        raise AuditError("Run in the original reference environment: " + "; ".join(differences))
    return observed


def worker(plan_path: Path) -> int:
    plan = read_json(plan_path)
    ref, sources, output = map(Path, (plan["reference_root"], plan["sources"], plan["output"]))
    owner = Path(plan["owner_root"]).resolve()
    if not inside(output, owner) or output.resolve() == owner or inside(output, ref):
        raise AuditError("Unsafe worker output path")
    if sources.resolve() != (owner / "sources").resolve() or not (owner / "meanpool_manifest.json").is_file():
        raise AuditError("Worker must use the owned output's private source copy.")
    # Check the actual trainer arguments as well as the plan's descriptive paths.
    for flag, expected in (("--out_dir", output), ("--data_folder", None), ("--test_folder", None)):
        locations = [i for i, value in enumerate(plan["cli"]) if value == flag]
        if len(locations) != 1 or locations[0] + 1 >= len(plan["cli"]):
            raise AuditError(f"Worker requires exactly one valid {flag} argument.")
        resolved = Path(plan["cli"][locations[0] + 1]).resolve()
        if (expected is not None and resolved != expected.resolve()) or (expected is None and not inside(resolved, ref)):
            raise AuditError(f"Unsafe or mismatched worker argument: {flag} {resolved}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sources))
    protect_reference(ref)
    runtime = validate_runtime(ref)
    import models_amr as models
    import temporal_graph_dataset as dataset_module
    install_mean_pool(models)
    # The dataset's vocabulary cache is an optional disk write. Keep its exact
    # construction/use, but never write a cache into the source dataset folder.
    dataset_module.TemporalGraphDataset._save_vocab_to_disk = lambda self: None
    import train_amr_dygformer as trainer
    if trainer.AMRDyGFormer is not models.AMRDyGFormer:
        raise AuditError("Trainer did not bind to the mean-pooling model.")
    records: List[Dict[str, Any]] = []
    original_class = trainer.AMRDyGFormer

    class RecordedMeanPool(original_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            records.append({"reference_parameter_count": self.reference_parameter_count,
                            "meanpool_parameter_count": sum(p.numel() for p in self.parameters()),
                            "trainable_parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
                            "gnn_trainable": all(p.requires_grad for p in self.gnn.parameters()),
                            "parameter_names": list(dict(self.named_parameters()))})

    trainer.AMRDyGFormer = RecordedMeanPool
    sys.argv = [str(sources / "train_amr_dygformer.py"), *plan["cli"]]
    trainer.main()
    if len(records) != 1:
        raise AuditError("Expected exactly one model construction per worker.")
    architecture = {"name": "seven_day_graph_mean_pool", "history_days": 7,
                    "temporal_attention": False, "positional_encoding": False,
                    "order_invariant_given_daily_embeddings": True,
                    "training": "end-to-end; GraphSAGE encoder is not frozen",
                    "change": "head(mean(H, dim=1)); original output activation retained",
                    "initialisation": "original constructor called before removing temporal parameters; shared initial weights matched by seed",
                    "sampling_note": "The same seed/settings are used. Removing temporal dropout changes subsequent training RNG consumption; training draw sequences are not asserted identical.",
                    **records[0], "runtime": runtime}
    atomic_json(output / "model_architecture.json", architecture)
    summary_path = output / "metrics_summary.json"
    summary = read_json(summary_path)
    summary["architecture"] = architecture
    atomic_json(summary_path, summary)
    return 0


def run_worker(plan: Dict[str, Any], plan_path: Path, log_path: Path) -> None:
    atomic_json(plan_path, plan)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(plan["config"]["model_seed"]), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    cache = Path(plan["owner_root"]) / "runtime_cache"
    for name in ("matplotlib", "tmp"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(cache / "matplotlib")
    env["TMPDIR"] = str(cache / "tmp")
    for key in ("DT_PT_OUT_DIR", "DT_STATE_MODE", "DT_SIM_EXTRA_ARGS", "DT_CONVERT_EXTRA_ARGS"):
        env.pop(key, None)
    if plan["config"]["deterministic"]:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-plan", str(plan_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {'TRAIN' if plan['fit'] else 'EVALUATE ONLY'} {plan['track']} repeat {plan['repeat']} {plan['step']}\n  Log: {log_path}", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + json.dumps({"command": command, "trainer_arguments": plan["cli"]}) + "\n")
        f.flush()
        process = subprocess.Popen(command, cwd=plan["sources"], env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"  Running; follow {log_path}", flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code:
        raise AuditError(f"Worker exited with code {code}. See {log_path}. No reference outputs were changed.")


def check_task(folder: Path, fingerprint: str) -> bool:
    marker = folder / "meanpool_complete.json"
    if not marker.exists():
        return False
    saved = read_json(marker)
    if saved["fingerprint"] != fingerprint:
        raise AuditError(f"Completed mean-pooling task has different inputs/settings: {folder}")
    for name in TASK_FILES:
        if not (folder / name).is_file() or saved["files"].get(name) != file_hash(folder / name):
            raise AuditError(f"Completed mean-pooling artifact changed: {folder / name}")
    return True


def finish_task(folder: Path, fingerprint: str) -> None:
    missing = [name for name in TASK_FILES if not (folder / name).is_file()]
    if missing:
        raise AuditError(f"Worker did not produce required files: {missing}")
    atomic_json(folder / "meanpool_complete.json", {"fingerprint": fingerprint,
                "files": {name: file_hash(folder / name) for name in TASK_FILES}})


def keyed_predictions(stats: Any, path: Path) -> Tuple[Any, Any, Any, Any, Dict[Tuple[str, int], Tuple[int, int]]]:
    rows, y, p, point = stats.read_predictions(path)
    keys = {(r["sim_id"], int(r["day"])): (int(r["y"]), int(r["start_day"])) for r in rows}
    if any(int(r["day"]) - int(r["start_day"]) != 6 for r in rows):
        raise AuditError(f"Non-seven-day window in {path}")
    return rows, y, p, point, keys


def validate_predictions(stats: Any, own: Path, reference: Optional[Path], canonical: Optional[Path]) -> None:
    import numpy as np
    seen = {}
    for split in SPLITS:
        rows, y, p, point, keys = keyed_predictions(stats, own / f"{split}_predictions.csv")
        stats.verify_saved_metrics(own, "GNN", split, point)
        seen[split] = {r["sim_id"] for r in rows}
        if reference is not None:
            refrows, _, _, refpoint, refkeys = keyed_predictions(stats, reference / f"{split}_predictions.csv")
            stats.verify_saved_metrics(reference, "GNN", split, refpoint)
            if keys != refkeys:
                raise AuditError(f"MeanPool/Transformer labels or window IDs differ: {own} {split}")
        if canonical is not None and split in ("train", "validation"):
            cr, _, cp, _, ck = keyed_predictions(stats, canonical / f"{split}_predictions.csv")
            lookup = {(r["sim_id"], int(r["day"])): float(prob) for r, prob in zip(cr, cp)}
            ordered = [lookup[(r["sim_id"], int(r["day"]))] for r in rows] if ck == keys else []
            if ck != keys or not np.allclose(p, ordered, rtol=1e-6, atol=1e-8):
                raise AuditError("Step 9 changed canonical train/validation predictions.")
    if seen["train"] & seen["validation"] or (seen["train"] | seen["validation"]) & seen["test"]:
        raise AuditError(f"Trajectory leakage in {own}")


def paired_bootstrap(stats: Any, rows: List[Dict[str, str]], y: Any, p_reference: Any,
                     p_mean: Any, repetitions: int, seed: int) -> Dict[str, Any]:
    """Same family-stratified trajectory draws as the reference helper; paired differences."""
    import numpy as np
    groups: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["sim_id"], []).append(i)
    families: Dict[str, List[str]] = {}
    for sid in sorted(groups):
        families.setdefault(sid.split("__sim_", 1)[0], []).append(sid)
    rng = np.random.default_rng(seed)
    values: Dict[str, List[float]] = {key: [] for key in METRICS}
    if len(groups) >= 2:
        for _ in range(repetitions):
            selected = []
            for family in sorted(families):
                ids = families[family]
                selected.extend(rng.choice(ids, len(ids), replace=True).tolist())
            ix = np.asarray([i for sid in selected for i in groups[sid]], dtype=int)
            left = stats.binary_metrics(y[ix], p_reference[ix])
            right = stats.binary_metrics(y[ix], p_mean[ix])
            for key in METRICS:
                if left[key] is not None and right[key] is not None:
                    values[key].append(float(left[key] - right[key]))
    return {key: {"low": float(np.quantile(v, .025)) if v else None,
                  "high": float(np.quantile(v, .975)) if v else None,
                  "valid_bootstraps": len(v), "requested_bootstraps": repetitions,
                  "n_trajectories": len(groups), "n_families": len(families),
                  "method": "paired 95% percentile whole-trajectory bootstrap within simulator family; Transformer minus MeanPool; conditional on fitted models"}
            for key, v in values.items()}


def cached_calculation(path: Path, identity: Dict[str, Any], calculate: Any) -> Any:
    fingerprint = digest(identity)
    if path.is_file():
        obj = read_json(path)
        if obj.get("fingerprint") == fingerprint:
            return obj["result"]
    result = calculate()
    atomic_json(path, {"fingerprint": fingerprint, "result": result})
    return result


def export_table(path: Path, rows: List[Dict[str, Any]], repeat_intervals: bool) -> None:
    def escape(value: Any) -> str:
        return str(value).replace("_", r"\_").replace("%", r"\%")
    grouped = {}
    for row in rows:
        if row["split"] == "test":
            key = (row.get("repeat", "all"), row["mode"], row["step"], row["model"])
            grouped.setdefault(key, {})[row["metric"]] = row
    title = ("Means and 95\\% Student-$t$ intervals across matched repetitions" if repeat_intervals
             else "Test metrics and conditional 95\\% whole-trajectory bootstrap intervals")
    lines = [r"% Requires booktabs,longtable,pdflscape. T=history length; H=outcome horizon.",
             r"\begin{landscape}\begingroup\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{lll l ccccc}",
             r"\caption{Seven-day mean-pooling comparison. " + title + r".}"
             + (r"\label{tab:predictive-meanpool-repeat}" if repeat_intervals else r"\label{tab:predictive-meanpool-conditional}") + r"\\",
             r"\toprule Repeat & Track & Test setting & Model & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\",
             r"\midrule\endfirsthead",
             r"\toprule Repeat & Track & Test setting & Model & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\",
             r"\midrule\endhead\bottomrule\endfoot"]
    for key, items in sorted(grouped.items(), key=lambda x: tuple(map(str, x[0]))):
        cells = []
        for metric in METRICS:
            row = items[metric]
            point = row.get("mean") if repeat_intervals else row.get("point")
            lo = row.get("ci95_low") if repeat_intervals else row.get("low")
            hi = row.get("ci95_high") if repeat_intervals else row.get("high")
            cell = "--" if point is None else f"{point:.3f}"
            if lo is not None and hi is not None:
                cell += f" ({lo:.3f}, {hi:.3f})"
            cells.append(cell)
        r, mode, step, model = key
        rlabel = str(min(row["n_valid"] for row in items.values())) + " repeats" if repeat_intervals else str(r)
        lines.append(" & ".join([escape(rlabel), escape(mode), "Canonical" if step == STEP4 else "Zero-shot", escape(model), *cells]) + r" \\")
    lines += [r"\end{longtable}",
              r"\noindent The same seven daily graphs, labels and trajectory splits are used. MeanPool replaces positional encoding and the Transformer with an equal mean of daily GraphSAGE embeddings; the encoder remains trainable. No shifted training is performed. "
              + (r"Across-repeat intervals are undefined for one fit and are not clipped to metric bounds. " if repeat_intervals else r"These intervals condition on each fitted model and do not measure variation across training runs. "),
              r"\endgroup\end{landscape}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def export_paired_table(path: Path, rows: List[Dict[str, Any]], repeat_intervals: bool) -> None:
    """Publication table of matched differences; no additional model fitting."""
    grouped: Dict[Tuple[Any, str, str], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if row["split"] == "test":
            key = (row.get("repeat", "all"), row["mode"], row["step"])
            grouped.setdefault(key, {})[row["metric"]] = row
    title = ("95\\% Student-$t$ intervals across paired repetitions" if repeat_intervals
             else "conditional 95\\% paired whole-trajectory bootstrap intervals")
    label = "tab:predictive-meanpool-paired-repeat" if repeat_intervals else "tab:predictive-meanpool-paired-conditional"
    lines = [r"% Requires booktabs,longtable,pdflscape. No experimental values are filled until results exist.",
             r"\begin{landscape}\begingroup\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{longtable}{lllccccc}",
             r"\caption{Transformer minus seven-day mean-pooling: test-metric differences with " + title + r".}\label{" + label + r"}\\",
             r"\toprule Repeat & Track & Test setting & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\",
             r"\midrule\endfirsthead",
             r"\toprule Repeat & Track & Test setting & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\",
             r"\midrule\endhead\bottomrule\endfoot"]
    for (number, mode, step), items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        cells = []
        for metric in METRICS:
            row = items[metric]
            point = row.get("mean") if repeat_intervals else row.get("point")
            low = row.get("ci95_low") if repeat_intervals else row.get("low")
            high = row.get("ci95_high") if repeat_intervals else row.get("high")
            cell = "--" if point is None else f"{point:.3f}"
            if low is not None and high is not None:
                cell += f" ({low:.3f}, {high:.3f})"
            cells.append(cell)
        repeat_label = str(min(row["n_available"] for row in items.values())) + " repeats" if repeat_intervals else str(number)
        track_label = "Ground truth" if mode == "ground_truth" else "Partial observation"
        setting = "Canonical" if step == STEP4 else "Zero-shot"
        lines.append(" & ".join([repeat_label, track_label, setting, *cells]) + r" \\")
    lines += [r"\end{longtable}",
              r"\noindent Positive differences favour the Transformer. Both models use matched decision windows and labels. "
              + (r"Intervals use paired metric differences across independent joint repetitions and are undefined for one available repetition. "
                 if repeat_intervals else r"Whole trajectories are resampled within simulator families, with the same draws for both fitted models. These intervals do not measure variability across training repetitions. "),
              r"\endgroup\end{landscape}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def summarize(root: Path, ref: Path, manifest: Dict[str, Any], sources: Path) -> Dict[str, Any]:
    import numpy as np
    stats = load_statistics(sources)
    identity, cfg = manifest["identity"], manifest["identity"]["config"]
    destination, cache = root / "summary", root / "ci_cache"
    destination.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    per_fit, conditional, pair_conditional, pending = [], [], [], []
    paired_points: List[Dict[str, Any]] = []
    refs_checked: Dict[str, Any] = {}
    for rep in identity["repeats"]:
        number = rep["repeat"]
        for track in cfg["tracks"]:
            for step in STEPS:
                own = root / f"repeat_{number:03d}" / track / step
                marker = own / "meanpool_complete.json"
                if not marker.exists():
                    continue
                complete = read_json(marker)
                check_task(own, complete["fingerprint"])
                reference = reference_task(ref, number, track, step)
                ready = verify_reference_task(reference)
                if ready is None:
                    pending.append({"repeat": number, "mode": track, "step": step,
                                    "reason": "reference task not yet complete; run --summarize-only after it finishes"})
                else:
                    refs_checked[str(reference)] = ready
                    copied = root / "reference_comparisons" / f"repeat_{number:03d}" / track / step
                    copied.mkdir(parents=True, exist_ok=True)
                    for name in ("metrics_summary.json", *[f"{part}_predictions.csv" for part in SPLITS]):
                        source_file, destination_file = reference / name, copied / name
                        if not destination_file.is_file() or file_hash(destination_file) != ready["files"][name]:
                            shutil.copy2(source_file, destination_file)
                    atomic_json(copied / "source_completion_record.json", ready)
                for split in SPLITS:
                    rows, y, p, point, keys = keyed_predictions(stats, own / f"{split}_predictions.csv")
                    stats.verify_saved_metrics(own, "GNN", split, point)
                    meta = {"repeat": number, "mode": track, "step": step, "split": split,
                            "model_seed": rep["model_seed"], "eval_seed": rep["eval_seed"],
                            "parameter_seed": rep["parameter_seed"],
                            **{f"multiplier_{k}": v for k, v in rep["parameter_multipliers"].items()},
                            "n_trajectories": len({r["sim_id"] for r in rows})}
                    per_fit.append({**meta, "model": "MeanPool", **point})
                    refpoint, rp = None, None
                    if ready is not None:
                        rr, _, rawp, refpoint, rk = keyed_predictions(stats, reference / f"{split}_predictions.csv")
                        stats.verify_saved_metrics(reference, "GNN", split, refpoint)
                        if rk != keys:
                            raise AuditError(f"Unmatched windows or labels: {reference} {split}")
                        lookup = {(r["sim_id"], int(r["day"])): prob for r, prob in zip(rr, rawp)}
                        rp = np.asarray([lookup[(r["sim_id"], int(r["day"]))] for r in rows], dtype=float)
                        per_fit.append({**meta, "model": "Transformer", **refpoint})
                        paired_points.append({**meta, **{key: (refpoint[key] - point[key])
                                              if refpoint[key] is not None and point[key] is not None else None for key in METRICS}})
                    if split != "test":
                        continue
                    count, seed = int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]) + number
                    stem = f"r{number:03d}_{track}_{step}"
                    ownhash = file_hash(own / "test_predictions.csv")
                    for label, probs, metrics in (("MeanPool", p, point), ("Transformer", rp, refpoint)):
                        if probs is None:
                            continue
                        phash = ownhash if label == "MeanPool" else file_hash(reference / "test_predictions.csv")
                        ci = cached_calculation(cache / f"{stem}_{label}.json",
                            {"prediction": phash, "bootstrap_replicates": count, "seed": seed,
                             "summary_source": file_hash(sources / "summarize_predictive_repeats.py"), "version": 1},
                            lambda probs=probs: stats.trajectory_bootstrap(rows, y, probs, count, seed))
                        conditional.extend({**meta, "model": label, "metric": key, "point": metrics[key], **item}
                                           for key, item in ci.items())
                    if rp is not None:
                        pci = cached_calculation(cache / f"{stem}_paired.json",
                            {"meanpool": ownhash, "transformer": file_hash(reference / "test_predictions.csv"),
                             "bootstrap_replicates": count, "seed": seed, "version": VERSION},
                            lambda: paired_bootstrap(stats, rows, y, rp, p, count, seed))
                        pair_conditional.extend({**meta, "comparison": "Transformer minus MeanPool", "metric": key,
                            "point": refpoint[key] - point[key] if refpoint[key] is not None and point[key] is not None else None,
                            **item} for key, item in pci.items())
    expected = len(identity["repeats"])
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for row in per_fit:
        grouped.setdefault(tuple(row[k] for k in ("mode", "step", "model", "split")), []).append(row)
    marginal = []
    for key, rows in grouped.items():
        for metric in METRICS:
            marginal.append({**dict(zip(("mode", "step", "model", "split"), key)), "metric": metric,
                             "reference_repeats_expected": expected, "all_reference_repeats_included": len(rows) == expected,
                             "repeat_ids": ",".join(str(r["repeat"]) for r in rows),
                             **stats.mean_interval([r[metric] for r in rows])})
    grouped_pairs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in paired_points:
        grouped_pairs.setdefault(tuple(row[k] for k in ("mode", "step", "split")), []).append(row)
    pairs = []
    for key, rows in grouped_pairs.items():
        for metric in METRICS:
            pairs.append({**dict(zip(("mode", "step", "split"), key)), "comparison": "Transformer minus MeanPool",
                          "metric": metric, "reference_repeats_expected": expected,
                          "all_reference_repeats_included": len(rows) == expected,
                          "repeat_ids": ",".join(str(r["repeat"]) for r in rows),
                          **stats.mean_interval([r[metric] for r in rows])})
    write_csv(destination / "per_fit_metrics.csv", per_fit)
    write_csv(destination / "conditional_test_ci95.csv", conditional)
    write_csv(destination / "paired_conditional_test_ci95.csv", pair_conditional)
    write_csv(destination / "mean_metrics_ci95.csv", marginal)
    write_csv(destination / "paired_transformer_minus_meanpool_ci95.csv", pairs)
    export_table(destination / "conditional_test_table.tex", conditional, False)
    export_table(destination / "repeat_comparison_table.tex", marginal, True)
    export_paired_table(destination / "paired_conditional_test_table.tex", pair_conditional, False)
    export_paired_table(destination / "paired_repeat_comparison_table.tex", pairs, True)
    status = {"reference_repeats_expected": expected,
              "reference_comparisons_pending": pending,
              "completed_meanpool_fits": sum(1 for r in per_fit if r["model"] == "MeanPool" and r["step"] == STEP4 and r["split"] == "test"),
              "completed_meanpool_step9_evaluations": sum(1 for r in per_fit if r["model"] == "MeanPool" and r["step"] == STEP9 and r["split"] == "test"),
              "conditional_ci": "Exact reference trajectory_bootstrap function: whole trajectories sampled within simulator family; overlapping windows kept together; conditional on fitted model.",
              "repeat_ci": "Exact reference mean_interval function: untruncated two-sided Student-t interval over independent matched joint repetitions; undefined for one repetition.",
              "pairing": "Transformer minus MeanPool; paired by repeat, observation track, trajectory, decision day, history start and label. Same bootstrap draws for both models.",
              "scope": "Only canonical fitting and zero-shot evaluation. Not a single-day static graph model. No causal or Step 8 training changes.",
              "limitations": "One repeat cannot establish variation across fitting seeds/parameter draws. Selected-repeat summaries must not be compared as if they covered all reference repetitions. Removing Transformer dropout can change subsequent training RNG draws despite using the same seed.",
              "source_hashes": {name: file_hash(sources / name) for name in REQUIRED_SOURCES}}
    atomic_json(destination / "summary.json", status)
    for path, saved in refs_checked.items():
        if verify_reference_task(Path(path)) != saved:
            raise AuditError(f"Reference task changed during summary: {path}")
    print(json.dumps(status, indent=2), flush=True)
    return status


def bundle(root: Path) -> Path:
    target = root / "mean_pool_review_bundle.zip"
    candidates = list((root / "summary").glob("*")) + [root / "meanpool_manifest.json"]
    candidates += list((root / "plans").glob("*.json"))
    candidates += list((root / "reference_comparisons").rglob("*.csv"))
    candidates += list((root / "reference_comparisons").rglob("*.json"))
    for folder in sorted(root.glob("repeat_*/*/step*")):
        for name in (*[f"{s}_predictions.csv" for s in SPLITS], "metrics_summary.json", "model_architecture.json", "meanpool_complete.json"):
            candidates.append(folder / name)
    candidates += list((root / "logs").glob("*.log"))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(set(candidates)):
            if path.is_file():
                z.write(path, str(path.relative_to(root)))
        z.write(Path(__file__), "run_predictive_mean_pool.py")
        # Include exactly the uncertainty definitions used, for independent audit.
        for name in ("summarize_predictive_repeats.py", "predictive_repeats.py"):
            z.write(root / "sources" / name, "reference_statistics/" + name)
    return target


def select_repeats(value: str, reps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    allowed = {int(r["repeat"]) for r in reps}
    chosen = allowed if value.lower() == "all" else {int(x.strip()) for x in value.split(",") if x.strip()}
    if not chosen or not chosen <= allowed:
        raise AuditError(f"--repeat-ids must be all or IDs from {sorted(allowed)}")
    return [r for r in reps if r["repeat"] in chosen]


def preflight(ref: Path, output: Path, manifest: Dict[str, Any], reps: List[Dict[str, Any]], tracks: List[str], full_hash: bool) -> Dict[str, Any]:
    tasks = []
    for rep in reps:
        for track in tracks:
            ref4 = reference_task(ref, rep["repeat"], track, STEP4)
            ready = verify_reference_task(ref4)
            task = {"repeat": rep["repeat"], "track": track, "model_seed": rep["model_seed"], "state": "pending_reference_step4"}
            if ready is not None:
                summary = read_json(ref4 / "metrics_summary.json")
                config = effective_config(summary, manifest["identity"], rep, track)
                try:
                    datasets = {kind: verify_dataset(ref, rep, track, kind, dataset_location(ref, rep, track, kind, config), full_hash)
                                for kind in ("canonical_train", "canonical_test", "shift_test")}
                except FileNotFoundError as exc:
                    task.update(state="pending_reference_data", reason=str(exc))
                else:
                    own = output / f"repeat_{rep['repeat']:03d}" / track
                    task.update(state="ready", datasets=datasets, config=config,
                                reference_step4_sha256=file_hash(ref4 / "metrics_summary.json"),
                                existing_canonical_fit=(own / STEP4 / "meanpool_complete.json").is_file(),
                                existing_zero_shot_evaluation=(own / STEP9 / "meanpool_complete.json").is_file(),
                                reference_step9_ready=verify_reference_task(reference_task(ref, rep["repeat"], track, STEP9)) is not None)
            tasks.append(task)
    missing = [name for name in ("numpy", "scipy", "sklearn", "torch", "torch_geometric", "matplotlib", "networkx", "yaml")
               if importlib.util.find_spec(name) is None]
    return {"tasks": tasks, "selected_repeat_ids": [r["repeat"] for r in reps], "tracks": tracks,
            "maximum_canonical_fits": len(reps) * len(tracks),
            "canonical_fits_ready_to_run": sum(t["state"] == "ready" and not t["existing_canonical_fit"] for t in tasks),
            "additional_step8_fits": 0, "additional_simulations": 0, "transformer_refits": 0,
            "missing_python_modules": missing,
            "ci_scope": "One selected repeat: conditional trajectory-bootstrap CIs only. Two or more: also Student-t CIs across selected matched repetitions. Use all for the reference's full repeat design."}


def self_test(sources: Path, root: Path) -> None:
    import numpy as np
    import torch
    stats = load_statistics(sources)
    models = import_source(sources / "models_amr.py", "_meanpool_selftest_models")
    original = models.AMRDyGFormer
    kwargs = dict(in_channels=6, hidden_channels=8, edge_dim=2, heads=2, T=7, dropout=0.2,
                  use_cls_token=False, n_outputs=2, n_layers=2, sage_layers=2,
                  use_softplus=False, output_activation="identity")
    torch.manual_seed(91)
    baseline = original(**kwargs)
    original_rng = torch.get_rng_state().clone()
    install_mean_pool(models)
    torch.manual_seed(91)
    model = models.AMRDyGFormer(**kwargs)
    assert torch.equal(torch.get_rng_state(), original_rng), "Constructor RNG is not matched"
    b = baseline.state_dict()
    for name, value in model.state_dict().items():
        assert torch.equal(value, b[name]), name
    assert not any(n.startswith(("pos_emb", "transformer", "cls_token")) for n, _ in model.named_parameters())
    H = torch.randn(4, 7, 8, requires_grad=True)
    model.eval()
    out = model.forward_from_day_embeddings(H)
    assert torch.allclose(out, model.head(H.mean(1)))
    assert torch.allclose(out, model.forward_from_day_embeddings(H[:, [6, 1, 5, 2, 4, 0, 3]]), atol=1e-6)
    out.sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    # Exercise real graph encoding and confirm end-to-end encoder gradients.
    from torch_geometric.data import Data, Batch
    graphs = []
    for _ in range(7):
        items = [Data(x=torch.randn(4, 6), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]), edge_attr=torch.ones(4, 2)) for _ in range(2)]
        graphs.append(Batch.from_data_list(items))
    model.zero_grad()
    graph_out = model(graphs)
    graph_out.square().sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gnn.parameters())
    assert torch.allclose(graph_out.detach(), model(list(reversed(graphs))).detach(), atol=1e-6)
    rows = [{"sim_id": f"f{j // 2}__sim_{j:03d}", "day": str(7 + i)} for j in range(4) for i in range(3)]
    y = np.asarray([0, 1, 0] * 4)
    p = np.asarray([.2, .7, .3] * 4)
    marginal = stats.trajectory_bootstrap(rows, y, p, 50, 99)
    paired = paired_bootstrap(stats, rows, y, p, p, 50, 99)
    assert all(v["low"] == 0 and v["high"] == 0 for v in paired.values())
    assert all(v["valid_bootstraps"] == 50 for v in marginal.values())
    assert stats.mean_interval([.5])["ci95_low"] is None
    assert stats.mean_interval([.5, .6, .7])["ci95_low"] is not None
    atomic_json(root / "self_test.json", {"passed": True,
                "checks": ["shared seeded initial weights", "constructor RNG", "temporal parameter removal",
                           "mean-pooling equality", "permutation invariance", "end-to-end graph gradients",
                           "reference CI functions", "zero paired difference", "one-fit t-CI undefined"]})
    print("Self-test passed.", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, help="Existing predictive_repeat_results containing run_manifest.json and source_snapshot")
    parser.add_argument("--out-root", type=Path, help="New separate output directory; never the original results directory")
    parser.add_argument("--repeat-ids", default="1", help="Comma-separated reference repeat IDs or all. Default: 1")
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, help="Default: both tracks present in the reference manifest")
    parser.add_argument("--preflight", action="store_true", help="Verify source/settings/data and show fit count, without training")
    parser.add_argument("--summarize-only", action="store_true", help="Refresh summaries/comparisons, without fitting or evaluating graphs")
    parser.add_argument("--verify-data-hashes", action="store_true", help="Hash every daily tensor rather than reuse verified size/mtime records; more disk I/O")
    parser.add_argument("--self-test", action="store_true", help="Small real-graph architecture/CI tests in the existing conda environment; no experiments")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_plan:
        return worker(args.worker_plan.resolve())
    if args.reference_root is None or args.out_root is None:
        parser.error("--reference-root and --out-root are required")
    ref, root = args.reference_root.expanduser().resolve(), args.out_root.expanduser().resolve()
    if inside(root, ref) or inside(ref, root):
        raise AuditError("Reference and output directories must be separate, non-nested trees.")
    manifest, source_hashes = read_reference(ref)
    validate_output_target(root, ref, manifest)
    # Summarisation and preflight are read-only with respect to the reference too.
    protect_reference(ref)
    cfg = manifest["identity"]["config"]
    tracks = list(dict.fromkeys(args.tracks or cfg["tracks"]))
    if not set(tracks) <= set(cfg["tracks"]):
        raise AuditError("Selected observation track is absent from the reference run.")
    reps = select_repeats(args.repeat_ids, manifest["identity"]["repeats"])
    if not args.summarize_only and not args.self_test:
        plan = preflight(ref, root, manifest, reps, tracks, args.verify_data_hashes)
        print(json.dumps(plan, indent=2), flush=True)
        if args.preflight:
            return 0
        if plan["missing_python_modules"]:
            raise AuditError("Use the reference conda environment. Missing: " + ", ".join(plan["missing_python_modules"]))
    with output_lock(root):
        sources = prepare_output(root, ref, manifest, source_hashes)
        if args.self_test:
            self_test(sources, root)
            return 0
        if args.summarize_only:
            summarize(root, ref, manifest, sources)
            print(f"Review bundle: {bundle(root)}", flush=True)
            return 0
        stats = load_statistics(sources)
        plans = root / "plans"
        plans.mkdir(exist_ok=True)
        for task in plan["tasks"]:
            if task["state"] != "ready":
                print(f"SKIPPED (not yet ready): repeat {task['repeat']} {task['track']}: {task['state']}", flush=True)
                continue
            number, track, config = task["repeat"], task["track"], task["config"]
            rep = next(r for r in reps if r["repeat"] == number)
            datasets = task["datasets"]
            ref4 = reference_task(ref, number, track, STEP4)
            protected_before = verify_reference_task(ref4)
            canonical = root / f"repeat_{number:03d}" / track / STEP4
            for step in STEPS:
                own = root / f"repeat_{number:03d}" / track / step
                train = Path(datasets["canonical_train"]["path"])
                test_kind = "canonical_test" if step == STEP4 else "shift_test"
                test = Path(datasets[test_kind]["path"])
                cli = build_cli(config, cfg, train, test, own, step == STEP4)
                fp = digest({"version": VERSION, "reference_identity": manifest["identity_sha256"],
                             "reference_step4_summary": task["reference_step4_sha256"],
                             "config": config, "cli": cli,
                             "dataset_identity": {k: v["identity"] for k, v in datasets.items()},
                             "canonical_checkpoint": file_hash(canonical / "trained_model.pt") if step == STEP9 else None})
                if check_task(own, fp):
                    print(f"REUSE completed: repeat {number} {track} {step}", flush=True)
                    continue
                if step == STEP9:
                    own.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(canonical / "trained_model.pt", own / "trained_model.pt")
                job = {"reference_root": str(ref), "owner_root": str(root), "sources": str(sources),
                       "output": str(own), "config": config, "cli": cli, "fit": step == STEP4,
                       "repeat": number, "track": track, "step": step, "dataset_audit": datasets,
                       "reference_model_seed": rep["model_seed"], "reference_eval_seed": rep["eval_seed"]}
                stem = f"repeat_{number:03d}_{track}_{step}"
                run_worker(job, plans / (stem + ".json"), root / "logs" / (stem + ".log"))
                reference = reference_task(ref, number, track, step)
                reference_ready = verify_reference_task(reference)
                validate_predictions(stats, own, reference if reference_ready else None,
                                     canonical if step == STEP9 else None)
                if step == STEP9 and file_hash(own / "trained_model.pt") != file_hash(canonical / "trained_model.pt"):
                    raise AuditError("Step 9 changed the canonical mean-pooling checkpoint.")
                for kind in ("canonical_train", test_kind):
                    after = verify_dataset(ref, rep, track, kind, Path(datasets[kind]["path"]))
                    if after["file_stat_digest"] != datasets[kind]["file_stat_digest"]:
                        raise AuditError("Reference graph files changed during the experiment.")
                finish_task(own, fp)
            if verify_reference_task(ref4) != protected_before:
                raise AuditError("Reference Step 4 artifacts changed during fitting.")
            summarize(root, ref, manifest, sources)
            print(f"Review bundle: {bundle(root)}", flush=True)
        status = summarize(root, ref, manifest, sources)
        atomic_json(root / "last_launch_status.json", {"selected_repeat_ids": [r["repeat"] for r in reps],
                    "tracks": tracks, "skipped_reference_tasks": [t for t in plan["tasks"] if t["state"] != "ready"],
                    "summary": status})
        print(f"Review bundle: {bundle(root)}", flush=True)
        if any(t["state"] != "ready" for t in plan["tasks"]):
            print("Some requested reference tasks are not ready. Re-run the same command after they finish; completed fits are skipped.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
