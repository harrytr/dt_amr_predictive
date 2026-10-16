#!/usr/bin/env python3
"""Original-data seven-day mean-pooling GNN comparator: Steps 4 and 9 ONLY.

Read original experiments_results and saved xgboost_original_results. Train one
canonical GraphSAGE + seven-day mean model per selected observation track, then
apply its checkpoint to the original shifted TEST set without retraining.
No simulation, conversion, Step 8 training, Transformer refit or XGBoost fit.

Default: ground truth, matching the original XGBoost experiment. Both original
XGBoost variants and the original Transformer predictions are reused. Optional
partial observation is supported, but the supplied original XGBoost code has
no partial-observation comparator. This is NOT a single-day static graph model.

Conditional 95% trajectory-bootstrap intervals are recalculated for ALL saved
models with the current predictive-repeat summarizer. Paired differences use
identical resampled trajectories. This original single-cohort analysis does NOT
produce five-independent-repeat Student-t intervals.

Save in AMR/addons/, not as a new top-level Python file in the AMR repository.
Example (run from addons):
  python run_original_mean_pool.py --repo .. \\
    --data-root ../experiments_results \\
    --xgboost-root ../xgboost_original_results \\
    --out-root ../original_mean_pool_results --preflight
Remove --preflight to run. See --help for optional paths and self-test.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
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
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

VERSION = '1.0.0'
STEP4 = 'step4_baseline'
STEP9 = 'step9_baseline_to_shift_test'
STEPS = (STEP4, STEP9)
TRACKS = ('ground_truth', 'partial_observation')
SPLITS = ('train', 'validation', 'test')
METRICS = ('auroc', 'accuracy', 'macro_f1', 'sensitivity', 'specificity')
TASK = 'endogenous_importation_majority_h7'
OWNER = 'original_meanpool_manifest.json'
SCIENTIFIC_VALUES = (
    'T', 'sliding_step', 'hidden', 'heads', 'dropout', 'transformer_layers',
    'sage_layers', 'batch_size', 'epochs', 'lr', 'max_neighbors',
    'num_neighbors', 'seed_count', 'seed_strategy', 'seed_batch_size',
    'max_sub_batches', 'split_seed', 'patience', 'min_delta',
    'lr_scheduler_factor', 'lr_scheduler_patience', 'lr_scheduler_min_lr',
)
SCIENTIFIC_BOOLS = ('neighbor_sampling', 'early_stopping', 'save_best_only',
                    'lr_scheduler_on_plateau')
REQUIRED = ('models_amr.py', 'train_amr_dygformer.py', 'temporal_graph_dataset.py',
            'tasks.py', 'experiments_pbc7.py', 'experiments_xgboost.py',
            'predictive_repeats.py', 'summarize_predictive_repeats.py')
TASK_FILES = ('trained_model.pt', 'metrics_summary.json', 'model_architecture.json',
              'train_predictions.csv', 'validation_predictions.csv', 'test_predictions.csv')
XGB_TAGS = {STEP4: ('step4_baseline', 'step5_no_admission_turnover'),
            STEP9: ('step9_baseline_to_shift_test', 'step9_no_admission_turnover')}
MODEL_LABELS = {'Transformer': 'GraphSAGE + Transformer',
                'MeanPool': 'GraphSAGE + seven-day mean',
                'XGB_all': 'XGBoost: all summaries',
                'XGB_no_counts': 'XGBoost: without counts'}


class AuditError(RuntimeError):
    """The inputs or outputs cannot safely be treated as a matched comparison."""


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as handle:
        if rows:
            fields = list(dict.fromkeys(k for row in rows for k in row))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temp, path)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def natural_key(value: Any) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]


def import_source(path: Path, name: str) -> Any:
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AuditError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_helpers(source: Path) -> Tuple[Any, Any, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    driver = import_source(source / 'experiments_pbc7.py', '_original_pool_driver')
    xgb = import_source(source / 'experiments_xgboost.py', '_original_pool_xgb')
    import_source(source / 'predictive_repeats.py', 'predictive_repeats')
    stats = import_source(source / 'summarize_predictive_repeats.py', '_original_pool_statistics')
    for name in ('binary_metrics', 'trajectory_bootstrap', 'read_predictions'):
        if not callable(getattr(stats, name, None)):
            raise AuditError(f'The repository summarizer lacks {name}.')
    return driver, xgb, stats


def protect_inputs(repo: Path, roots: Sequence[Path], output: Optional[Path]) -> None:
    """Block Python filesystem mutations to the source repo and reference trees.

    The owned output is allowed inside the repo, but never inside a data/results
    input tree. This complements the trainer's disabled vocabulary-cache writer.
    It is not an operating-system sandbox for arbitrary untrusted native code.
    """
    repo = repo.resolve()
    protected = [p.resolve() for p in roots]
    allowed = output.resolve() if output is not None else None

    def forbidden(raw: Any) -> bool:
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return False  # stdout/stderr file descriptors are not filesystem paths.
        p = Path(os.fsdecode(raw))
        if any(inside(p, root) for root in protected):
            return True
        return inside(p, repo) and not (allowed is not None and inside(p, allowed))

    def audit(event: str, args: Tuple[Any, ...]) -> None:
        paths: Sequence[Any] = ()
        if event == 'open':
            mode, flags = args[1], args[2]
            write = isinstance(mode, str) and any(c in mode for c in 'wax+')
            write = write or (isinstance(flags, int) and bool(flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            paths = args[:1] if write else ()
        elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'shutil.rmtree'):
            paths = args[:1]
        elif event in ('os.rename', 'os.replace'):
            paths = args[:2]
        elif event in ('os.link', 'os.symlink'):
            paths = args[1:2]
        if any(forbidden(p) for p in paths):
            raise PermissionError(f'Original-mean-pool runner refuses to modify inputs: {event} {paths}')
    sys.addaudithook(audit)


def validate_output(root: Path, repo: Path, protected: Sequence[Path], identity: Any = None) -> None:
    if root == repo or inside(repo, root):
        raise AuditError('Output cannot be the repository itself or an ancestor of it.')
    if any(inside(root, p) or inside(p, root) for p in protected):
        raise AuditError('Output and existing data/comparator results must be separate, non-nested trees.')
    if not root.parent.is_dir():
        raise AuditError(f'Output parent does not exist: {root.parent}')
    if root.exists():
        if not root.is_dir():
            raise AuditError(f'Output is not a directory: {root}')
        for p in root.rglob('*'):
            if p.is_symlink() or (p.is_file() and p.stat().st_nlink > 1):
                raise AuditError(f'Output contains a linked file/directory; choose a new folder: {p}')
        marker = root / OWNER
        if marker.is_file():
            if identity is not None and read_json(marker).get('identity') != identity:
                raise AuditError('This output belongs to another source/input/configuration. Use a new --out-root.')
        elif any(p.name != '.original_meanpool.lock' for p in root.iterdir()):
            raise AuditError('Refusing an unrelated nonempty output folder. Choose a new --out-root.')


@contextmanager
def output_lock(root: Path):
    import fcntl  # The supplied workflow runs in the macOS/Linux conda environment.
    root.mkdir(exist_ok=True)
    handle = (root / '.original_meanpool.lock').open('a+b')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise AuditError(f'Another runner owns {root}') from exc
    try:
        yield
    finally:
        handle.close()


def validate_original_root(root: Path) -> None:
    if (root / 'run_manifest.json').exists() or re.fullmatch(r'repeat_\d+', root.name):
        raise AuditError('--data-root must be ORIGINAL experiments_results, not a predictive-repeat directory.')
    if any(p.name == 'predictive_repeat_results' for p in (root, *root.parents)):
        raise AuditError('Predictive-repeat data are deliberately excluded by this original-benchmark runner.')


def rebase_saved_path(raw: Any, data_root: Path, track: str) -> Optional[Path]:
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    if path.exists() and inside(path, data_root):
        return path.resolve()
    parts = path.parts
    anchor = f'TRACK_{track}'
    if anchor in parts:
        candidate = data_root.joinpath(*parts[parts.index(anchor):])
        if candidate.exists():
            return candidate.resolve()
    return None


def select_existing(paths: Sequence[Optional[Path]], label: str, xgb: Any) -> Path:
    candidates = list(dict.fromkeys(p.resolve() for p in paths if p is not None))
    for candidate in candidates:
        if xgb.has_pt(candidate):
            return candidate
    raise FileNotFoundError(f'Missing {label}; no data will be generated. Checked: {candidates}')


def checked_xgb_artifact(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved XGBoost artifact: {path}')
    sha = file_hash(path)
    marker = path.parent / '_task.json'
    if marker.is_file():
        value = read_json(marker)
        expected = value.get('artifacts', {}).get(path.name)
        if value.get('status') != 'complete' or expected != sha:
            raise AuditError(f'Incomplete or changed XGBoost artifact: {path}')
    return sha


def discover(args: argparse.Namespace, driver: Any, xgb: Any) -> List[Dict[str, Any]]:
    root, xroot = args.data_root, args.xgboost_root
    plans_path = xroot / '01_inputs' / 'dataset_plan.json'
    if not plans_path.is_file():
        raise FileNotFoundError(f'Expected original XGBoost results with {plans_path}; not just 04_tables.')
    checked_xgb_artifact(plans_path)
    xp = read_json(plans_path)
    if not isinstance(xp, list):
        raise AuditError('XGBoost dataset_plan.json must contain the original list of experiment plans.')
    plan_lookup = {(p['mode'], p['tag']): p for p in xp}
    found = []
    for track in args.tracks:
        work = root / f'TRACK_{track}' / 'work'
        archive = work / 'repro_artifacts_steps_1_9'
        if not work.is_dir():
            raise FileNotFoundError(f'Original observation track not found: {work}')
        four, nine = plan_lookup.get((track, STEP4), {}), plan_lookup.get((track, STEP9), {})
        metrics = {}
        for tag, prior in ((STEP4, four), (STEP9, nine)):
            saved = rebase_saved_path(prior.get('original_metrics'), root, track)
            metrics[tag] = saved if saved is not None else xgb.original_metrics_path(archive, tag)
            if metrics[tag] is None:
                raise FileNotFoundError(f'Missing original GNN metrics_summary.json for {track}/{tag} in {archive}.')
        summaries = {tag: read_json(path) for tag, path in metrics.items()}
        datasets = {
            'canonical_train': select_existing([
                rebase_saved_path(four.get('train_folder'), root, track),
                rebase_saved_path(summaries[STEP4]['config'].get('data_folder'), root, track),
                work / driver.BASELINE_TRAIN_REL,
                *xgb.archive_candidates(archive, STEP4, 'train_folder')], f'{track} canonical train', xgb),
            'canonical_test': select_existing([
                rebase_saved_path(four.get('test_folder'), root, track),
                rebase_saved_path(summaries[STEP4]['config'].get('test_folder'), root, track),
                work / driver.BASELINE_TEST_REL,
                *xgb.archive_candidates(archive, STEP4, 'test_folder')], f'{track} canonical test', xgb),
            'shift_test': select_existing([
                rebase_saved_path(nine.get('test_folder'), root, track),
                rebase_saved_path(summaries[STEP9]['config'].get('test_folder'), root, track),
                work / driver.STEP8_TEST_REL,
                *xgb.archive_candidates(archive, STEP9, 'test_folder'),
                *xgb.archive_candidates(archive, 'step8_distribution_shift', 'test_folder')], f'{track} shifted TEST', xgb),
        }
        # No discovery of STEP8_TRAIN_REL: its existence is deliberately irrelevant.
        for folder in datasets.values():
            if not inside(folder, root):
                raise AuditError(f'Dataset outside the specified ORIGINAL data root: {folder}')
        xb = {}
        if track == 'ground_truth':
            for step in STEPS:
                for model, tag in zip(('XGB_all', 'XGB_no_counts'), XGB_TAGS[step]):
                    directory = xroot / '03_experiments' / track / tag
                    for name in ('metrics.json', 'train_predictions.csv', 'validation_predictions.csv', 'test_predictions.csv'):
                        checked_xgb_artifact(directory / name)
                    record = read_json(directory / 'metrics.json')
                    if record.get('step') != tag or record.get('mode') != track:
                        raise AuditError(f'Wrong original XGBoost experiment: {directory}')
                    expected_width = 28 if model == 'XGB_all' else 14
                    if int(record.get('feature_width', -1)) != expected_width:
                        raise AuditError(f'Unexpected original XGBoost feature width: {directory}')
                    if step == STEP9 and record.get('fit_status') != 'zero_shot_baseline_model':
                        raise AuditError(f'XGBoost Step 9 is not a saved zero-shot evaluation: {directory}')
                    for field, kind in (('training_source', 'canonical_train'),):
                        mapped = rebase_saved_path(record.get(field), root, track)
                        if mapped is not None and mapped != datasets[kind]:
                            # Identical archive copies are checked by content below, not silently assumed.
                            if not xgb.has_pt(mapped):
                                raise AuditError(f'Unavailable XGBoost training source: {mapped}')
                    xb[f'{step}/{model}'] = str(directory)
            # Require the same canonical XGBoost model to be used for zero-shot evaluation.
            for model in ('XGB_all', 'XGB_no_counts'):
                a = Path(xb[f'{STEP4}/{model}']) / 'model.json'
                b = Path(xb[f'{STEP9}/{model}']) / 'model.json'
                if a.is_file() and b.is_file() and file_hash(a) != file_hash(b):
                    raise AuditError(f'{model}: original Step 9 model differs from its Step 4 model.')
        # Where both archived checkpoints remain, verify that the historical
        # Step 9 really used the canonical fitted parameters. Different container
        # serialization is allowed only if every state-dict tensor is identical.
        cp4, cp9 = (metrics[tag].parent / 'trained_model.pt' for tag in STEPS)
        checkpoint_check = 'checkpoints absent; original archived evaluation records used'
        if cp4.is_file() and cp9.is_file():
            if file_hash(cp4) == file_hash(cp9):
                checkpoint_check = 'original Step 4 and Step 9 checkpoints are byte-identical'
            else:
                import torch
                left = torch.load(cp4, map_location='cpu', weights_only=False)
                right = torch.load(cp9, map_location='cpu', weights_only=False)
                if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
                    raise AuditError(f'{track}: original Step 4/9 checkpoint structures differ.')
                if any(not torch.is_tensor(left[k]) or not torch.is_tensor(right[k]) or
                       not torch.equal(left[k], right[k]) for k in left):
                    raise AuditError(f'{track}: original Step 9 parameters differ from Step 4.')
                checkpoint_check = 'original Step 4/9 parameters identical; serialization differs'
                del left, right
        found.append({'track': track, 'reference_checkpoint_check': checkpoint_check,
                      'datasets': {k: str(v) for k, v in datasets.items()},
                      'original_metrics': {k: str(v) for k, v in metrics.items()},
                      'original_metric_sha256': {k: file_hash(v) for k, v in metrics.items()},
                      'xgboost': xb})
    return found


def stat_inventory(folder: Path) -> List[Dict[str, Any]]:
    result = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.name.endswith('.pt') and entry.is_file():
                st = entry.stat()
                result.append({'name': entry.name, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns})
    result.sort(key=lambda x: natural_key(x['name']))
    if not result:
        raise AuditError(f'No saved PT graphs: {folder}')
    return result


def normalize_samples(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result, seen = [], set()
    for r in rows:
        row = {k: r[k] for k in ('sim_id', 'day', 'start_day', 'last_file', 'y')}
        row.update(sim_id=str(row['sim_id']), day=int(row['day']), start_day=int(row['start_day']), y=int(row['y']))
        key = (row['sim_id'], row['day'])
        if not row['sim_id'] or key in seen or row['day'] - row['start_day'] != 6 or row['y'] not in (0, 1):
            raise AuditError('Duplicate/invalid seven-day sample metadata.')
        if Path(row['last_file']).name != row['last_file']:
            raise AuditError('Unexpected non-flat graph filename in sample metadata.')
        seen.add(key)
        result.append(row)
    if not result:
        raise AuditError('No eligible original history windows.')
    return result


def inspect_dataset(folder: Path, track: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Reuse verified XGBoost metadata; otherwise read each existing graph once."""
    inventory = stat_inventory(folder)
    candidates = []
    feature_root = args.xgboost_root / '02_features' / track
    for audit in feature_root.glob('*/feature_audit.json'):
        source = rebase_saved_path(read_json(audit).get('source'), args.data_root, track)
        if source == folder:
            candidates.append(audit.parent)
    local = args.out_root / 'input_audit' / f'{track}_{digest(str(folder))[:16]}.json'
    if local.is_file():
        old = read_json(local)
        if old.get('stat_digest') == digest(inventory) and not args.verify_data_hashes:
            old['metadata_origin'] = 'previously verified original-data index in this output folder'
            return old
    if not candidates and track == 'ground_truth':
        # A dataset may have been moved from the live work directory into its
        # original archive. Recover the cache by the EXACT saved file inventory,
        # never merely by coincident window counts or matching binary labels.
        names = {item['name'] for item in inventory}
        for files in feature_root.glob('*/source_files.json'):
            stored_files = read_json(files)
            if {item['name'] for item in stored_files} == names:
                candidates.append(files.parent)
    if len(candidates) > 1:
        raise AuditError(f'Ambiguous XGBoost feature caches for {folder}: {candidates}')
    if not candidates and track == 'ground_truth':
        raise AuditError(f'No saved XGBoost file-inventory cache matches {folder}. '
                         'The original full XGBoost results are required, not only its tables. '
                         'Refusing to infer equal input data from labels alone.')
    if candidates:
        cache = candidates[0]
        for name in ('source_files.json', 'samples.json', 'feature_audit.json'):
            checked_xgb_artifact(cache / name)
        stored = read_json(cache / 'source_files.json')
        lookup = {r['name']: r for r in stored}
        if len(lookup) != len(stored) or set(lookup) != {r['name'] for r in inventory}:
            raise AuditError(f'Original PT file set no longer matches the XGBoost benchmark: {folder}')
        checked = []
        for item in inventory:
            prior = lookup[item['name']]
            if item['bytes'] != prior['bytes']:
                raise AuditError(f'Graph size differs from the XGBoost input: {folder / item["name"]}')
            sha = prior.get('sha256')
            if not sha or args.verify_data_hashes or item['mtime_ns'] != prior.get('mtime_ns'):
                observed = file_hash(folder / item['name'])
                if not sha or sha != observed:
                    raise AuditError(f'Graph content differs from the XGBoost input: {folder / item["name"]}')
                sha = observed
            checked.append({**item, 'sha256': sha})
        samples = normalize_samples(read_json(cache / 'samples.json'))
        origin = 'XGBoost cache; exact filenames/sizes/mtime checks and hashes for changed timestamps'
    else:
        import torch
        groups: Dict[str, List[Dict[str, Any]]] = {}
        checked = []
        for i, item in enumerate(inventory, 1):
            p = folder / item['name']
            data_bytes = p.read_bytes()
            g = torch.load(io.BytesIO(data_bytes), map_location='cpu', weights_only=False)
            sid = str(getattr(g, 'sim_id', ''))
            if not sid or not hasattr(g, 'day') or not hasattr(g, 'y_h7_endog_majority'):
                raise AuditError(f'Original graph lacks required metadata/label: {p}')
            if int(g.x.shape[1]) != 6:
                raise AuditError(f'Expected the original predictive six-column graph, not causal data: {p}')
            day = int(g.day)
            y = int(g.y_h7_endog_majority.item())
            groups.setdefault(sid, []).append({'day': day, 'name': item['name'], 'y': y})
            checked.append({**item, 'sha256': hashlib.sha256(data_bytes).hexdigest()})
            if i % 1000 == 0:
                print(f'  Indexed {i}/{len(inventory)} existing graphs: {track}/{folder.name}', flush=True)
            del g, data_bytes
        samples = []
        for sid, days in groups.items():
            days.sort(key=lambda r: r['day'])
            for a, b in zip(days, days[1:]):
                if b['day'] != a['day'] + 1:
                    raise AuditError(f'Duplicate or noncontiguous original graph days: {sid}')
            for start in range(len(days) - 6):
                last = days[start + 6]
                samples.append({'sim_id': sid, 'day': last['day'], 'start_day': days[start]['day'],
                                'last_file': last['name'], 'y': last['y']})
        samples = normalize_samples(samples)
        origin = 'original PT metadata scanned in the trainer/XGBoost natural-file, trajectory and day order'
    if stat_inventory(folder) != inventory:
        raise AuditError(f'Input changed during inspection: {folder}')
    return {'path': str(folder), 'track': track, 'samples': samples, 'source_files': checked,
            'stat_digest': digest(inventory), 'metadata_origin': origin,
            'content_digest': digest([{k: r[k] for k in ('name', 'bytes', 'sha256')} for r in checked]),
            'n_daily_graphs': len(checked), 'n_windows': len(samples),
            'n_trajectories': len({r['sim_id'] for r in samples})}


def scientific_config(summary: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(summary.get('config', {}))
    required = set(SCIENTIFIC_VALUES) | set(SCIENTIFIC_BOOLS) | {'use_cls'}
    missing = required - cfg.keys()
    if missing:
        raise AuditError(f'Original metrics lack effective scientific settings: {sorted(missing)}. '
                         'Supply the original complete metrics_summary.json; settings are not guessed.')
    if summary.get('task') != TASK or int(cfg['T']) != 7 or int(cfg['sliding_step']) != 1:
        raise AuditError('Expected the ORIGINAL seven-day endogenous/importation task, stride 1.')
    cfg['task'] = TASK
    cfg['original_model_seed'] = cfg.get('model_seed')
    cfg['original_eval_seed'] = cfg.get('eval_seed')
    cfg['model_seed'] = int(cfg['model_seed']) if cfg.get('model_seed') is not None else args.unrecorded_model_seed
    cfg['eval_seed'] = int(cfg['eval_seed']) if cfg.get('eval_seed') is not None else args.unrecorded_eval_seed
    cfg['deterministic'] = bool(cfg.get('deterministic', False))
    cfg['seed_note'] = ('Original model seed recorded and reused.' if cfg['original_model_seed'] is not None else
        f'Original model seed was not recorded. New comparator seed={cfg["model_seed"]}; '
        'identical historical initialization cannot be claimed.')
    cfg['evaluation_note'] = ('Original evaluation seed recorded and reused.' if cfg['original_eval_seed'] is not None else
        f'Original evaluation seed was not recorded. New comparator evaluation seed={cfg["eval_seed"]}; '
        'original saved GNN probabilities are reused, not resampled.')
    return cfg


def expected_partitions(samples: List[Dict[str, Any]], test: List[Dict[str, Any]],
                        seed: int, xgb: Any) -> Dict[str, List[Dict[str, Any]]]:
    train, validation = xgb.original_split(samples, seed)
    result = {'train': [samples[int(i)] for i in train],
              'validation': [samples[int(i)] for i in validation], 'test': test}
    ids = {s: {r['sim_id'] for r in rows} for s, rows in result.items()}
    if ids['train'] & ids['validation'] or (ids['train'] | ids['validation']) & ids['test']:
        raise AuditError('Original training/validation/test trajectories overlap.')
    return result


def prediction_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return str(row['sim_id']), int(row['day'])


def verify_rows(rows: Sequence[Dict[str, Any]], expected: Sequence[Dict[str, Any]],
                stats: Any, label: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import numpy as np
    lookup = {prediction_key(r): r for r in rows}
    keys = [prediction_key(r) for r in expected]
    if len(lookup) != len(rows) or set(lookup) != set(keys):
        raise AuditError(f'{label}: trajectory/day identities differ from the original benchmark.')
    ordered = []
    for e in expected:
        r = lookup[prediction_key(e)]
        if int(r['y']) != e['y'] or int(r['start_day']) != e['start_day']:
            raise AuditError(f'{label}: labels/history windows differ from the original benchmark.')
        if r.get('last_file') and Path(str(r['last_file'])).name != e['last_file']:
            raise AuditError(f'{label}: terminal source-graph filename differs.')
        p = float(r['probability'])
        pred = int(r.get('prediction', int(p > .5)))
        if pred != int(p > .5):
            raise AuditError(f'{label}: saved prediction disagrees with the fixed 0.5 rule.')
        ordered.append({**e, 'probability': p, 'prediction': pred})
    y = np.asarray([r['y'] for r in ordered], dtype=int)
    p = np.asarray([r['probability'] for r in ordered], dtype=float)
    return ordered, stats.binary_metrics(y, p)


def verify_points(calculated: Dict[str, Any], saved: Dict[str, Any], label: str,
                  gnn: bool = False) -> None:
    aliases = {'auroc': 'roc_auc', 'macro_f1': 'f1_macro'} if gnn else {}
    for key in METRICS:
        original = saved.get(aliases.get(key, key))
        if aliases.get(key, key) not in saved:
            # Older GNN JSONs expose these two via their confusion matrix instead.
            if gnn and key in ('sensitivity', 'specificity'):
                continue
            raise AuditError(f'{label}: missing original saved {key}.')
        new = calculated[key]
        if original is None and new is None:
            continue
        if original is None or new is None or not math.isclose(float(original), float(new), rel_tol=1e-7, abs_tol=1e-9):
            raise AuditError(f'{label}: saved metric {key}={original}, recalculated={new}.')


def gnn_reference(path: Path, expected: Dict[str, List[Dict[str, Any]]], stats: Any) -> Dict[str, Any]:
    import numpy as np
    summary = read_json(path)
    result = {}
    for split in SPLITS:
        block = summary.get(split)
        if block is None and split == 'train':
            continue  # Never relabel historical validation as fitted training data.
        if block is None:
            raise AuditError(f'Missing original GNN {split} results: {path}')
        rows_path = path.parent / f'{split}_predictions.csv'
        if rows_path.is_file():
            rows, point = verify_rows(read_csv(rows_path), expected[split], stats, f'GNN/{split}')
        else:
            y = block.get('y_true')
            probs = np.asarray(block.get('probabilities', []), dtype=float)
            if y != [r['y'] for r in expected[split]] or probs.shape != (len(y or []), 2):
                raise AuditError(f'{path}: original GNN arrays do not match reconstructed {split} windows.')
            predictions = block.get('y_pred')
            if predictions is None or len(predictions) != len(y):
                raise AuditError(f'{path}: original GNN {split} predictions missing.')
            raw = [{**r, 'probability': float(p[1]), 'prediction': int(pred)}
                   for r, p, pred in zip(expected[split], probs, predictions)]
            rows, point = verify_rows(raw, expected[split], stats, f'GNN arrays/{split}')
        verify_points(point, block['metrics'], f'{path}/{split}', gnn=True)
        result[split] = rows
    return result


def xgb_reference(directory: Path, expected: Dict[str, List[Dict[str, Any]]], stats: Any) -> Dict[str, Any]:
    original = read_json(directory / 'metrics.json')
    rows = {}
    for split in SPLITS:
        rows[split], point = verify_rows(read_csv(directory / f'{split}_predictions.csv'),
                                         expected[split], stats, f'{directory}/{split}')
        verify_points(point, original[split], f'{directory}/{split}')
    return rows
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



def runtime_versions() -> Dict[str, Any]:
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'torch-geometric', 'networkx', 'PyYAML'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'packages': packages}


def build_cli(cfg: Dict[str, Any], train: Path, test: Path, output: Path,
              fit: bool, threads: int, workers: int) -> List[str]:
    cli = ['--data_folder', str(train), '--test_folder', str(test), '--out_dir', str(output),
           '--task', TASK, '--train_model', str(fit).lower()]
    for key in SCIENTIFIC_VALUES:
        value = cfg[key]
        if key == 'num_neighbors' and isinstance(value, list):
            value = ','.join(str(int(v)) for v in value)
        cli += [f'--{key}', str(value)]
    for key in SCIENTIFIC_BOOLS:
        cli += [f'--{key}', str(bool(cfg[key])).lower()]
    if cfg['use_cls']:
        cli.append('--use_cls')  # Original constructor remains seed-matched before token removal.
    for key in ('model_seed', 'eval_seed'):
        cli += [f'--{key}', str(cfg[key])]
    cli += ['--deterministic', str(cfg['deterministic']).lower(),
            '--num_workers', str(workers), '--torch_num_threads', str(threads),
            '--torch_num_interop_threads', str(min(4, threads)),
            '--evaluate_train', 'true', '--export_predictions', 'true', '--metrics_only', 'true',
            '--emit_translational_figures', 'false', '--fullgraph_attribution_pass', 'false',
            '--require_pt_metadata', 'true', '--fail_on_noncontiguous', 'true']
    # Do not pass --use_task_hparams: cfg already contains effective post-override settings.
    return cli


def worker(plan_path: Path) -> int:
    plan = read_json(plan_path)
    owner, repo, sources, output = [Path(plan[k]).resolve() for k in ('owner', 'repo', 'sources', 'output')]
    manifest = read_json(owner / OWNER)
    if not inside(output, owner) or output == owner or sources != owner / 'sources':
        raise AuditError('Unsafe worker paths.')
    if plan['step'] not in STEPS or bool(plan['fit']) != (plan['step'] == STEP4):
        raise AuditError('Only canonical fitting and Step 9 evaluation are permitted.')
    for name, expected in manifest['identity']['source_sha256'].items():
        if file_hash(sources / name) != expected:
            raise AuditError(f'Private source changed: {sources / name}')
    for flag, target in (('--out_dir', output), ('--data_folder', Path(plan['train'])),
                         ('--test_folder', Path(plan['test']))):
        positions = [i for i, item in enumerate(plan['cli']) if item == flag]
        if len(positions) != 1 or Path(plan['cli'][positions[0] + 1]).resolve() != target.resolve():
            raise AuditError(f'Worker CLI does not match its declared {flag}.')
    protect_inputs(repo, [Path(p) for p in plan['protected_roots']], owner)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(sources))
    import models_amr as models
    import temporal_graph_dataset as dataset_module
    install_mean_pool(models)
    dataset_module.TemporalGraphDataset._save_vocab_to_disk = lambda self: None
    import train_amr_dygformer as trainer
    if trainer.AMRDyGFormer is not models.AMRDyGFormer:
        raise AuditError('Trainer did not import the seven-day mean-pooling model.')
    base = trainer.AMRDyGFormer
    records = []

    class RecordedMeanPool(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            records.append({'original_parameter_count': self.reference_parameter_count,
                            'new_parameter_count': sum(p.numel() for p in self.parameters()),
                            'gnn_trainable': all(p.requires_grad for p in self.gnn.parameters()),
                            'parameter_names': list(dict(self.named_parameters()))})

    trainer.AMRDyGFormer = RecordedMeanPool
    sys.argv = [str(sources / 'train_amr_dygformer.py'), *plan['cli']]
    trainer.main()
    if len(records) != 1 or not records[0]['gnn_trainable']:
        raise AuditError('Expected one end-to-end trainable mean-pooling model.')
    if any(n.startswith(('transformer', 'pos_emb', 'cls_token')) for n in records[0]['parameter_names']):
        raise AuditError('A temporal parameter unexpectedly survived the ablation.')
    architecture = {'model': 'seven-day graph mean pooling', 'history_days': 7,
                    'single_day_static_graph': False, 'temporal_attention': False,
                    'positional_encoding': False, 'GraphSAGE_frozen': False,
                    'change': 'Equal mean of seven daily graph embeddings followed by original classifier.',
                    'original_model_seed': plan['config']['original_model_seed'],
                    'model_seed': plan['config']['model_seed'], 'seed_note': plan['config']['seed_note'],
                    'evaluation_note': plan['config']['evaluation_note'],
                    'sampling_note': 'Same sampling settings. Removing Transformer dropout changes training RNG consumption; identical sampling draws are not claimed.',
                    **records[0]}
    atomic_json(output / 'model_architecture.json', architecture)
    summary = read_json(output / 'metrics_summary.json')
    summary['architecture'] = architecture
    atomic_json(output / 'metrics_summary.json', summary)
    return 0


def run_worker(plan: Dict[str, Any], root: Path, stem: str) -> None:
    plan_path = root / 'plans' / f'{stem}.json'
    log_path = root / 'logs' / f'{stem}.log'
    atomic_json(plan_path, plan)
    log_path.parent.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONHASHSEED=str(plan['config']['model_seed']), PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg', OMP_NUM_THREADS=str(plan['threads']),
               OPENBLAS_NUM_THREADS=str(plan['threads']), MKL_NUM_THREADS=str(plan['threads']))
    if plan['config']['deterministic']:
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for variable, subdir in (('MPLCONFIGDIR', 'matplotlib'), ('TORCH_HOME', 'torch'),
                             ('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp')):
        target = root / 'runtime_cache' / subdir
        target.mkdir(parents=True, exist_ok=True)
        env[variable] = str(target)
    env['TMP'] = env['TEMP'] = env['TMPDIR']
    command = [sys.executable, str(Path(__file__).resolve()), '--worker-plan', str(plan_path)]
    print(f'RUN {stem}: {"canonical training" if plan["fit"] else "zero-shot evaluation only"}\n  log: {log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as handle:
        proc = subprocess.Popen(command, cwd=plan['sources'], env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    status = proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f'  Running {stem}; log: {log_path}', flush=True)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if status:
        tail = '\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-35:])
        raise AuditError(f'{stem} failed (exit {status}). Existing inputs remain unchanged.\n{tail}\nFull log: {log_path}')


def task_complete(folder: Path, fingerprint: str) -> bool:
    path = folder / 'meanpool_complete.json'
    if not path.exists():
        return False
    record = read_json(path)
    if record.get('fingerprint') != fingerprint:
        raise AuditError(f'Completed mean-pooling fit has different settings: {folder}')
    for name in TASK_FILES:
        if not (folder / name).is_file() or file_hash(folder / name) != record['files'].get(name):
            raise AuditError(f'Completed output changed: {folder / name}')
    return True


def finish_task(folder: Path, fingerprint: str) -> None:
    missing = [name for name in TASK_FILES if not (folder / name).is_file()]
    if missing:
        raise AuditError(f'Trainer output is incomplete at {folder}: {missing}')
    atomic_json(folder / 'meanpool_complete.json', {'fingerprint': fingerprint,
                'files': {name: file_hash(folder / name) for name in TASK_FILES}})


def joint_bootstrap(stats: Any, rows: Sequence[Dict[str, Any]], probabilities: Dict[str, Any],
                    repetitions: int, seed: int) -> Dict[str, Any]:
    """Same trajectory/family sampling and percentiles as the repository helper.

    Compute each model's metrics once per draw, then reuse those values for all
    paired differences. No row-wise resampling or independent draws for models.
    A numerical contract test below compares marginals to trajectory_bootstrap.
    """
    import numpy as np
    y = np.asarray([r['y'] for r in rows], dtype=int)
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r['sim_id']), []).append(i)
    families: Dict[str, List[str]] = {}
    for sid in sorted(groups):
        families.setdefault(sid.split('__sim_', 1)[0], []).append(sid)
    values = {name: {key: [] for key in METRICS} for name in probabilities}
    pairs = [(a, b) for a, b in (('Transformer', 'MeanPool'), ('MeanPool', 'XGB_all'),
                                 ('MeanPool', 'XGB_no_counts')) if a in probabilities and b in probabilities]
    pair_values = {f'{a} minus {b}': {key: [] for key in METRICS} for a, b in pairs}
    rng = np.random.default_rng(seed)
    if len(groups) >= 2:
        for _ in range(repetitions):
            selected = []
            for family in sorted(families):
                ids = families[family]
                selected.extend(rng.choice(ids, len(ids), replace=True).tolist())
            idx = np.asarray([i for sid in selected for i in groups[sid]], dtype=int)
            computed = {name: stats.binary_metrics(y[idx], np.asarray(p)[idx]) for name, p in probabilities.items()}
            for name, point in computed.items():
                for key in METRICS:
                    if point[key] is not None:
                        values[name][key].append(float(point[key]))
            for a, b in pairs:
                for key in METRICS:
                    if computed[a][key] is not None and computed[b][key] is not None:
                        pair_values[f'{a} minus {b}'][key].append(computed[a][key] - computed[b][key])
    def pack(numbers: Dict[str, List[float]]) -> Dict[str, Any]:
        return {key: {'low': float(np.quantile(v, .025)) if v else None,
                      'high': float(np.quantile(v, .975)) if v else None,
                      'valid_bootstraps': len(v), 'requested_bootstraps': repetitions,
                      'n_trajectories': len(groups), 'n_families': len(families)} for key, v in numbers.items()}
    return {'marginal': {name: pack(item) for name, item in values.items()},
            'paired': {name: pack(item) for name, item in pair_values.items()},
            'method': '95% percentile bootstrap of whole trajectories within simulator family; identical draws for all models; conditional on fixed fitted models',
            'seed': seed, 'repetitions': repetitions,
            'family_sizes': {name: len(ids) for name, ids in families.items()}}


def check_statistics_contract(stats: Any) -> None:
    import numpy as np
    rows = [{'sim_id': f'family_{j // 2}__sim_{j:03d}', 'day': 7 + i,
             'start_day': 1 + i, 'y': int((i + j) % 3 == 0)} for j in range(6) for i in range(4)]
    y = np.asarray([r['y'] for r in rows])
    p = np.asarray([.15 + .7 * ((i * 7) % 11) / 10 for i in range(len(rows))])
    reference = stats.trajectory_bootstrap(rows, y, p, 37, 719)
    actual = joint_bootstrap(stats, rows, {'Transformer': p, 'MeanPool': p}, 37, 719)
    for metric in METRICS:
        for field in ('low', 'high', 'valid_bootstraps', 'requested_bootstraps', 'n_trajectories', 'n_families'):
            a, b = reference[metric][field], actual['marginal']['MeanPool'][metric][field]
            if a != b:
                raise AuditError(f'Bootstrap no longer matches the repository helper: {metric}/{field}: {a} vs {b}')
        if actual['paired']['Transformer minus MeanPool'][metric]['low'] != 0:
            raise AuditError('Paired-bootstrap identity self-test failed.')


def latex_escape(value: Any) -> str:
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(char, char) for char in str(value))


def export_tables(destination: Path, metric_rows: List[Dict[str, Any]], paired_rows: List[Dict[str, Any]]) -> None:
    def cell(row: Dict[str, Any]) -> str:
        if row['point'] is None:
            return '--'
        point = f'{row["point"]:.4f}'
        if row['low'] is None:
            return point + r' (CI not estimable)'
        return r'\shortstack{' + point + r'\\(' + f'{row["low"]:.4f}, {row["high"]:.4f}' + ')}'
    for paired, rows, filename in ((False, metric_rows, 'original_benchmark_test_table.tex'),
                                   (True, paired_rows, 'original_benchmark_paired_differences.tex')):
        grouped = {}
        for row in rows:
            grouped.setdefault((row['track'], row['step'], row['comparison'] if paired else row['model']), {})[row['metric']] = row
        caption = ('Paired performance differences on the original predictive test datasets. ' if paired else
                   'Predictive model comparison on the original test datasets. ')
        caption += ('The seven-day graph mean-pooling model is trained only on the original canonical data; '
                    'Step 9 applies that checkpoint to the original shifted test data without retraining. '
                    'Values are ' + ('differences' if paired else 'point estimates') + r' with 95\% whole-trajectory bootstrap intervals.')
        lines = [r'% Requires: \usepackage{booktabs,adjustbox}', r'\begin{table}[!htbp]',
                 r'\centering\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.2}',
                 r'\caption{' + caption + '}',
                 r'\label{tab:original-meanpool-' + ('paired' if paired else 'performance') + '}',
                 r'\begin{adjustbox}{max width=\textwidth}', r'\begin{tabular}{llccccc}',
                 r'\toprule Test & ' + ('Comparison' if paired else 'Model') + r' & AUROC & Accuracy & Macro F1 & Sensitivity & Specificity \\', r'\midrule']
        for track in TRACKS:
            if not any(k[0] == track for k in grouped):
                continue
            lines += [r'\multicolumn{7}{l}{\textbf{' + track.replace('_', ' ').title() + r'}} \\']
            for step in STEPS:
                for key, group in grouped.items():
                    if key[:2] != (track, step):
                        continue
                    name = key[2]
                    if paired:
                        a, b = name.split(' minus ')
                        name = MODEL_LABELS[a] + ' minus ' + MODEL_LABELS[b]
                    else:
                        name = MODEL_LABELS[name]
                    title = 'Step 4: canonical' if step == STEP4 else 'Step 9: zero-shot'
                    lines.append(' & '.join([title, latex_escape(name), *[cell(group[k]) for k in METRICS]]) + r' \\')
                lines.append(r'\addlinespace')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\par\smallskip',
                  r'\begin{minipage}{\textwidth}\scriptsize ',
                  ('Positive differences favour the first named model. ' if paired else '') +
                  'All models in a comparison use identical test windows and labels. Resampling retains complete '
                  'trajectories and their overlapping windows within each simulator family. These intervals are '
                  'conditional on one fitted model per method, not intervals across independent training repetitions. '
                  'The mean-pooling model retains seven daily graphs but has no positional encoding or Transformer. '
                  'Existing Transformer and XGBoost predictions are reused; their point estimates are not replaced by new fits.',
                  r'\end{minipage}', r'\end{table}']
        (destination / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (destination / 'review_tables.tex').write_text(
        '\\documentclass{article}\n\\usepackage[a4paper,landscape,margin=15mm]{geometry}\n'
        '\\usepackage{booktabs,adjustbox}\n\\begin{document}\n'
        '\\input{original_benchmark_test_table.tex}\n\\clearpage\n'
        '\\input{original_benchmark_paired_differences.tex}\n\\end{document}\n', encoding='utf-8')


def summarize(root: Path, stats: Any) -> Dict[str, Any]:
    import numpy as np
    manifest = read_json(root / OWNER)
    ci = manifest['identity']['ci']
    evidence_index = read_json(root / 'reference_evidence_index.json')
    for relative, sha in evidence_index.items():
        if not (root / relative).is_file() or file_hash(root / relative) != sha:
            raise AuditError(f'Owned reference/partition evidence changed: {relative}')
    destination = root / 'summary'
    destination.mkdir(exist_ok=True)
    all_metrics, interval_rows, paired_rows, pending = [], [], [], []
    for track in manifest['identity']['tracks']:
        for step in STEPS:
            folder = root / track / step
            marker = folder / 'meanpool_complete.json'
            if not marker.is_file():
                pending.append(f'{track}/{step}')
                continue
            if not task_complete(folder, read_json(marker)['fingerprint']):
                raise AuditError(f'Incomplete owned result: {folder}')
            expected = read_json(root / 'input_audit' / f'{track}_{step}_partitions.json')
            all_models = {'MeanPool': {s: read_csv(folder / f'{s}_predictions.csv') for s in SPLITS}}
            references = root / 'reference_predictions' / track / step
            for model in ('Transformer', 'XGB_all', 'XGB_no_counts'):
                if (references / model).is_dir():
                    all_models[model] = {s: read_csv(references / model / f'{s}_predictions.csv')
                                         for s in SPLITS if (references / model / f'{s}_predictions.csv').is_file()}
            # Consistent table order, without manufacturing absent PO XGBoost results.
            all_models = {model: all_models[model] for model in MODEL_LABELS if model in all_models}
            probabilities, points, hashes = {}, {}, {}
            for model, splits in all_models.items():
                for split, records in splits.items():
                    ordered, point = verify_rows(records, expected[split], stats, f'{track}/{step}/{model}/{split}')
                    all_metrics.append({'track': track, 'step': step, 'model': model, 'split': split,
                                        'n_trajectories': len({r['sim_id'] for r in ordered}), **point})
                    if split == 'test':
                        probabilities[model] = np.asarray([r['probability'] for r in ordered], dtype=float)
                        points[model] = point
                        hashes[model] = digest(ordered)
            fp = digest({'prediction_hashes': hashes, 'ci': ci, 'method_version': VERSION})
            cache = destination / f'{track}_{step}_bootstrap.json'
            old = read_json(cache) if cache.is_file() else {}
            if old.get('fingerprint') == fp:
                result = old['result']
            else:
                print(f'Confidence intervals: {track}/{step}, {ci["repetitions"]} matched trajectory resamples', flush=True)
                result = joint_bootstrap(stats, expected['test'], probabilities, ci['repetitions'], ci['seed'])
                atomic_json(cache, {'fingerprint': fp, 'result': result})
            for model, metrics in result['marginal'].items():
                for key, info in metrics.items():
                    interval_rows.append({'track': track, 'step': step, 'model': model, 'metric': key,
                                          'point': points[model][key], **info})
            for comparison, metrics in result['paired'].items():
                a, b = comparison.split(' minus ')
                for key, info in metrics.items():
                    point = points[a][key] - points[b][key] if points[a][key] is not None and points[b][key] is not None else None
                    paired_rows.append({'track': track, 'step': step, 'comparison': comparison,
                                        'metric': key, 'point': point, **info})
    write_csv(destination / 'per_model_metrics.csv', all_metrics)
    write_csv(destination / 'conditional_test_ci95.csv', interval_rows)
    write_csv(destination / 'paired_differences_ci95.csv', paired_rows)
    export_tables(destination, interval_rows, paired_rows)
    status = {'complete': not pending, 'pending': pending, 'new_simulations': 0, 'Step8_training_runs': 0,
              'Transformer_refits': 0, 'XGBoost_refits': 0,
              'maximum_canonical_meanpool_fits': len(manifest['identity']['tracks']),
              'uncertainty': 'Conditional on the fixed original datasets and one fit per method; no independent-repeat t interval.',
              'ci': ci, 'original_seed_notes': manifest['identity']['seed_notes'],
              'table_rows_with_metrics': len(interval_rows), 'paired_rows_with_metrics': len(paired_rows)}
    atomic_json(destination / 'summary.json', status)
    return status


def make_bundle(root: Path) -> Path:
    destination = root / 'original_mean_pool_review_bundle.zip'
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p in (destination, temporary):
                continue
            relative = p.relative_to(root)
            if any(part in ('runtime_cache', '__pycache__') for part in relative.parts):
                continue
            if p.suffix not in ('.py', '.json', '.csv', '.tex', '.txt', '.log'):
                continue
            z.write(p, relative.as_posix())
    os.replace(temporary, destination)
    return destination


def self_test(repo: Path) -> None:
    import numpy as np
    import torch
    sys.path.insert(0, str(repo))
    _, _, stats = load_helpers(repo)
    check_statistics_contract(stats)
    import models_amr as models
    from torch_geometric.data import Batch, Data
    kwargs = {'in_channels': 6, 'hidden_channels': 8, 'edge_dim': 2, 'heads': 2,
              'T': 7, 'dropout': .2, 'n_outputs': 2, 'n_layers': 1,
              'sage_layers': 2, 'use_cls_token': False, 'use_softplus': False}
    torch.manual_seed(113)
    original = models.AMRDyGFormer(**kwargs)
    state = torch.get_rng_state().clone()
    original_class = install_mean_pool(models)
    try:
        torch.manual_seed(113)
        model = models.AMRDyGFormer(**kwargs)
        assert torch.equal(state, torch.get_rng_state())
        for key, value in model.state_dict().items():
            assert torch.equal(value, original.state_dict()[key]), key
        model.eval()
        H = torch.randn(3, 7, 8, requires_grad=True)
        assert torch.allclose(model.forward_from_day_embeddings(H), model.head(H.mean(1)))
        assert torch.allclose(model.forward_from_day_embeddings(H), model.forward_from_day_embeddings(H.flip(1)), atol=1e-6)
        graphs = []
        for _ in range(7):
            items = [Data(x=torch.randn(5, 6), edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
                          edge_attr=torch.ones(5, 2)) for _ in range(2)]
            graphs.append(Batch.from_data_list(items))
        output = model(graphs)
        output.square().sum().backward()
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.gnn.parameters())
        assert torch.allclose(output.detach(), model(list(reversed(graphs))).detach(), atol=1e-6)
        assert not any(name.startswith(('transformer', 'pos_emb', 'cls_token')) for name, _ in model.named_parameters())
        # Same-seed paired bootstrap must have zero width for identical predictors.
        assert stats.mean_interval([.5])['ci95_low'] is None
    finally:
        models.AMRDyGFormer = original_class
    print('Self-test passed: graph gradients, shared seeded initial weights, temporal removal, seven-day averaging, permutation invariance and reference-bootstrap equality.', flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=None, help='Original predictive code folder; inferred from addons/ or current directory.')
    parser.add_argument('--data-root', type=Path, help='ORIGINAL experiments_results with TRACK_ground_truth/work; not predictive_repeat_results.')
    parser.add_argument('--xgboost-root', type=Path, help='Original XGBoost output root containing 01_inputs, 02_features, 03_experiments.')
    parser.add_argument('--out-root', type=Path, help='New separate output folder, e.g. AMR/original_mean_pool_results.')
    parser.add_argument('--tracks', nargs='+', choices=TRACKS, default=['ground_truth'], help='Default ground truth for direct comparison with original XGBoost; optional PO.')
    parser.add_argument('--ci-config', type=Path, help='Predictive-repeat JSON configuration for bootstrap count/seed only; default repo/predictive_repeats.json.')
    parser.add_argument('--bootstrap', type=int, help='Override bootstrap count from CI configuration; no impact on model/data.')
    parser.add_argument('--bootstrap-seed', type=int, help='Override bootstrap seed from CI configuration; same seed used for every compared model.')
    parser.add_argument('--unrecorded-model-seed', type=int, default=0, help='New comparator seed ONLY if original model_seed is absent; explicitly recorded as unmatched (default 0).')
    parser.add_argument('--unrecorded-eval-seed', type=int, default=202609, help='New evaluation seed ONLY if original eval_seed is absent (default 202609).')
    parser.add_argument('--threads', type=int, default=4, help='Runtime CPU threads; scientific training settings are read from original metrics.')
    parser.add_argument('--loader-workers', type=int, default=0, help='Runtime DataLoader workers (default 0).')
    parser.add_argument('--preflight', action='store_true', help='Verify original data, saved settings and comparator alignment; no training or output writes.')
    parser.add_argument('--summarize-only', action='store_true', help='Regenerate comparison tables/bundle from completed owned results; no training.')
    parser.add_argument('--verify-data-hashes', action='store_true', help='Rehash every original PT file; slower than verified size/mtime reuse.')
    parser.add_argument('--self-test', action='store_true', help='Small architecture/CI test using the real repository dependencies; only --repo needed.')
    parser.add_argument('--worker-plan', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_plan:
        return worker(args.worker_plan.expanduser().resolve())
    if args.repo is None:
        options = (Path.cwd(), Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent)
        args.repo = next((p for p in options if (p / 'experiments_pbc7.py').is_file()), None)
        if args.repo is None:
            parser.error('Provide --repo pointing to the original AMR predictive code.')
    args.repo = args.repo.expanduser().resolve()
    for name in REQUIRED:
        p = args.repo / name
        if not p.is_file():
            raise FileNotFoundError(f'Required original code file missing: {p}')
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    if args.self_test:
        protect_inputs(args.repo, [], None)
        self_test(args.repo)
        return 0
    if args.data_root is None or args.xgboost_root is None or args.out_root is None:
        parser.error('--data-root, --xgboost-root and --out-root are required.')
    # Reject output symlinks before resolve() could hide their nature.
    if args.out_root.expanduser().is_symlink():
        raise AuditError('--out-root must not be a symbolic link.')
    for key in ('data_root', 'xgboost_root', 'out_root'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.tracks = list(dict.fromkeys(args.tracks))
    if args.threads < 1 or args.loader_workers < 0 or args.unrecorded_model_seed < 0 or args.unrecorded_eval_seed < 0:
        parser.error('Thread counts/seeds must be valid nonnegative values; threads must be positive.')
    validate_original_root(args.data_root)
    protected = [args.data_root, args.xgboost_root]
    validate_output(args.out_root, args.repo, protected)
    protect_inputs(args.repo, protected, args.out_root)
    driver, xgb, stats = load_helpers(args.repo)
    check_statistics_contract(stats)
    ci_path = (args.ci_config or args.repo / 'predictive_repeats.json').expanduser().resolve()
    ci_cfg = read_json(ci_path)
    ci = {'repetitions': args.bootstrap if args.bootstrap is not None else int(ci_cfg['bootstrap_replicates']),
          'seed': args.bootstrap_seed if args.bootstrap_seed is not None else int(ci_cfg['bootstrap_seed']),
          'config_source': str(ci_path), 'config_sha256': file_hash(ci_path),
          'method': 'original-data conditional whole-trajectory percentile bootstrap within simulator family',
          'not_a_five_repeat_interval': True}
    if ci['repetitions'] < 1 or ci['seed'] < 0:
        raise AuditError('Bootstrap count must be positive and seed nonnegative.')
    if args.summarize_only:
        if not (args.out_root / OWNER).is_file():
            raise AuditError('No owned original-meanpool run exists to summarize.')
        saved = read_json(args.out_root / OWNER)
        if saved['identity']['data_root'] != str(args.data_root) or saved['identity']['xgboost_root'] != str(args.xgboost_root):
            raise AuditError('Summary inputs differ from the owned run.')
        for name, sha in saved['identity']['source_sha256'].items():
            if file_hash(args.out_root / 'sources' / name) != sha:
                raise AuditError(f'Owned frozen source changed: {name}')
        _, _, frozen_stats = load_helpers(args.out_root / 'sources')
        check_statistics_contract(frozen_stats)
        with output_lock(args.out_root):
            summarize(args.out_root, frozen_stats)
            print(f'Review bundle: {make_bundle(args.out_root)}', flush=True)
        return 0
    plans = discover(args, driver, xgb)
    indices, references = {}, {}
    seed_notes = {}
    for plan in plans:
        track = plan['track']
        for kind, folder in plan['datasets'].items():
            indices[(track, kind)] = inspect_dataset(Path(folder), track, args)
        configs = {step: scientific_config(read_json(Path(plan['original_metrics'][step])), args) for step in STEPS}
        architecture_fields = ('T', 'hidden', 'heads', 'dropout', 'transformer_layers', 'sage_layers', 'use_cls')
        if any(configs[STEP4][key] != configs[STEP9][key] for key in architecture_fields):
            raise AuditError(f'{track}: original Step 9 architecture differs from the canonical source model.')
        if configs[STEP4]['split_seed'] != configs[STEP9]['split_seed']:
            raise AuditError(f'{track}: original Step 4 and Step 9 trajectory splits differ.')
        plan['configs'] = configs
        seed_notes[track] = {step: {k: configs[step][k] for k in ('seed_note', 'evaluation_note')} for step in STEPS}
        for step, testkind in ((STEP4, 'canonical_test'), (STEP9, 'shift_test')):
            expected = expected_partitions(indices[(track, 'canonical_train')]['samples'], indices[(track, testkind)]['samples'],
                                            int(configs[step]['split_seed']), xgb)
            references[(track, step)] = {'expected': expected, 'models': {
                'Transformer': gnn_reference(Path(plan['original_metrics'][step]), expected, stats)}}
            if track == 'ground_truth':
                for model in ('XGB_all', 'XGB_no_counts'):
                    references[(track, step)]['models'][model] = xgb_reference(Path(plan['xgboost'][f'{step}/{model}']), expected, stats)
    sources = {p.name: file_hash(p) for p in sorted(args.repo.glob('*.py'))}
    reference_hashes = {str(Path(p)): file_hash(Path(p)) for plan in plans for p in plan['original_metrics'].values()}
    for plan in plans:
        for path in plan['xgboost'].values():
            for name in ('metrics.json', *[f'{s}_predictions.csv' for s in SPLITS]):
                p = Path(path) / name
                reference_hashes[str(p)] = file_hash(p)
    identity = {'version': VERSION, 'script_sha256': file_hash(Path(__file__)), 'repo': str(args.repo),
                'data_root': str(args.data_root), 'xgboost_root': str(args.xgboost_root),
                'tracks': args.tracks, 'ci': ci, 'source_sha256': sources,
                'dataset_content_digests': {f'{track}/{kind}': item['content_digest'] for (track, kind), item in indices.items()},
                'reference_sha256': reference_hashes, 'effective_configs': {p['track']: p['configs'] for p in plans},
                'seed_notes': seed_notes, 'threads': args.threads, 'loader_workers': args.loader_workers,
                'runtime': runtime_versions()}
    validate_output(args.out_root, args.repo, protected, identity)
    modules = ('numpy', 'scipy', 'sklearn', 'torch', 'torch_geometric', 'networkx', 'matplotlib', 'yaml')
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    preflight = {'data_basis': 'ORIGINAL Step 4 and Step 9 datasets and original saved GNN/XGBoost predictions',
                 'tracks': args.tracks, 'maximum_new_canonical_fits': len(args.tracks),
                 'new_simulations': 0, 'new_tensor_conversions': 0, 'Step8_training_runs': 0,
                 'Transformer_refits': 0, 'XGBoost_refits': 0, 'confidence_intervals': ci,
                 'seed_notes': seed_notes, 'missing_python_modules': missing,
                 'tasks': [{'track': p['track'], 'datasets': {
                     kind: {k: indices[(p['track'], kind)][k] for k in ('path', 'n_daily_graphs', 'n_windows', 'n_trajectories', 'metadata_origin')}
                     for kind in p['datasets']}, 'original_metrics': p['original_metrics'],
                     'effective_Step4_config': p['configs'][STEP4]} for p in plans],
                 'output_root': str(args.out_root), 'preflight_performs_no_writes': True,
                 'historical_prediction_alignment': 'Original prediction CSV identifiers are checked when available; otherwise original JSON probabilities are mapped through the unchanged natural graph/window order and checked against all saved labels.'}
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight:
        return 0
    if missing:
        raise AuditError('Use the existing idp_ns environment. Missing packages: ' + ', '.join(missing))
    root = args.out_root
    with output_lock(root):
        if not (root / OWNER).exists():
            atomic_json(root / OWNER, {'identity': identity, 'plans': plans})
        source_dir = root / 'sources'
        source_dir.mkdir(exist_ok=True)
        for name, sha in sources.items():
            p, dest = args.repo / name, source_dir / name
            if file_hash(p) != sha:
                raise AuditError(f'Source changed during inspection: {p}')
            if dest.exists() and file_hash(dest) != sha:
                raise AuditError(f'Owned source copy changed: {dest}')
            if not dest.exists():
                shutil.copy2(p, dest)
        if not (root / 'run_original_mean_pool.py').exists():
            shutil.copy2(Path(__file__), root / 'run_original_mean_pool.py')
        atomic_json(root / 'preflight.json', preflight)
        for (track, kind), item in indices.items():
            name = f'{track}_{digest(item["path"])[:16]}.json'
            atomic_json(root / 'input_audit' / name, item)
        for (track, step), info in references.items():
            atomic_json(root / 'input_audit' / f'{track}_{step}_partitions.json', info['expected'])
            for model, splits in info['models'].items():
                for split, rows in splits.items():
                    write_csv(root / 'reference_predictions' / track / step / model / f'{split}_predictions.csv', rows)
        # Freeze a copy of the unchanged source reference files for the reviewer bundle.
        for plan in plans:
            for step, path in plan['original_metrics'].items():
                target = root / 'reference_records' / plan['track'] / step / 'gnn_metrics_summary.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(path, target)
        evidence = {p.relative_to(root).as_posix(): file_hash(p)
                    for base in (root / 'reference_predictions', root / 'reference_records')
                    for p in sorted(base.rglob('*')) if p.is_file()}
        evidence.update({p.relative_to(root).as_posix(): file_hash(p)
                         for p in sorted((root / 'input_audit').glob('*_partitions.json'))})
        evidence_path = root / 'reference_evidence_index.json'
        if evidence_path.exists() and read_json(evidence_path) != evidence:
            raise AuditError('Owned reference evidence changed unexpectedly.')
        atomic_json(evidence_path, evidence)
        for plan in plans:
            track = plan['track']
            canonical = root / track / STEP4
            for step, testkind in ((STEP4, 'canonical_test'), (STEP9, 'shift_test')):
                own = root / track / step
                cfg = plan['configs'][step]
                train, test = Path(plan['datasets']['canonical_train']), Path(plan['datasets'][testkind])
                cli = build_cli(cfg, train, test, own, step == STEP4, args.threads, args.loader_workers)
                fingerprint = digest({'identity': digest(identity), 'track': track, 'step': step, 'cli': cli,
                                      'canonical_checkpoint': file_hash(canonical / 'trained_model.pt') if step == STEP9 else None})
                if task_complete(own, fingerprint):
                    print(f'REUSE completed {track}/{step}', flush=True)
                    continue
                own.mkdir(parents=True, exist_ok=True)
                if step == STEP9:
                    shutil.copy2(canonical / 'trained_model.pt', own / 'trained_model.pt')
                job = {'owner': str(root), 'repo': str(args.repo), 'sources': str(source_dir), 'output': str(own),
                       'protected_roots': [str(p) for p in protected], 'track': track, 'step': step,
                       'fit': step == STEP4, 'train': str(train), 'test': str(test), 'config': cfg,
                       'cli': cli, 'threads': args.threads}
                run_worker(job, root, f'{track}_{step}')
                expected = references[(track, step)]['expected']
                new_summary = read_json(own / 'metrics_summary.json')
                for split in SPLITS:
                    _, point = verify_rows(read_csv(own / f'{split}_predictions.csv'), expected[split], stats, f'MeanPool/{step}/{split}')
                    verify_points(point, new_summary[split]['metrics'], f'MeanPool/{step}/{split}', gnn=True)
                for key in (*SCIENTIFIC_VALUES, *SCIENTIFIC_BOOLS, 'use_cls', 'model_seed', 'eval_seed', 'deterministic'):
                    a, b = new_summary['config'][key], cfg[key]
                    if key == 'num_neighbors':
                        a = list(a)
                        b = [int(v) for v in b.split(',')] if isinstance(b, str) else list(b)
                    if a != b:
                        raise AuditError(f'Effective trainer setting changed unexpectedly: {key}: {a} vs {b}')
                if step == STEP9 and file_hash(own / 'trained_model.pt') != file_hash(canonical / 'trained_model.pt'):
                    raise AuditError('Step 9 changed the newly trained canonical checkpoint.')
                for kind in ('canonical_train', testkind):
                    if digest(stat_inventory(Path(plan['datasets'][kind]))) != indices[(track, kind)]['stat_digest']:
                        raise AuditError('Original input graph files changed during execution.')
                finish_task(own, fingerprint)
                summarize(root, stats)
                make_bundle(root)
        for raw, expected_sha in reference_hashes.items():
            if file_hash(Path(raw)) != expected_sha:
                raise AuditError(f'Original reference predictions changed during execution: {raw}')
        for name, sha in sources.items():
            if file_hash(args.repo / name) != sha:
                raise AuditError(f'Original repository source changed during execution: {name}')
        status = summarize(root, stats)
        print(f'Complete: {status["complete"]}\nTables: {root / "summary"}\nReview bundle: {make_bundle(root)}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AuditError, FileNotFoundError, KeyError, ValueError, ImportError, PermissionError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(2)
