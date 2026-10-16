#!/usr/bin/env python3
"""Read-only audit of adding IS->IR selection to endogenous-majority labels.

Examples (run in the existing AMR environment):
  python audit_is_ir_labels.py --input canonical=/path/to/canonical_pt \
      --format pt --output audit_canonical
  python audit_is_ir_labels.py --input canonical=/path/to/canonical_graphml \
      --format graphml --output audit_canonical_graphml

Repeat --input NAME=PATH for separate datasets. Each path is searched recursively.
Use the original predictive data, not causal branches or checkpoints. Default:
7-day history, 7-day future, stride 1, ties endogenous (>= 0.5), event-free label 0.
Only windows with a complete history are counted; terminal futures are retained.

PT mode uses saved horizon counts and validates the old labels. By default it
bounds missing IS->IR counts between zero and ALL new resistant-infection events.
That upper bound also includes CR->IR progression; it is NOT an IS->IR estimate.
--pt-track ground_truth tightens bounds using original, unmodified scalar-six
or one-hot GT node features and stable node_names (continuing patients only).
The scalar-six encoding is [role, state, antibiotic, isolation, new_CR, new_IR].
New admissions and
staff remain uncertain because PT does not retain their full provenance.
Use the default aggregate mode for PO, feature ablations, or older encodings.

GraphML mode reconstructs IS->IR from daily event flags and the state BEFORE
selection, handling new admissions and forced superspreader states. Recovery
on the same day does not erase the event flag. Missing provenance yields bounds,
never a guessed exact count. Reconstruction assumes the supplied October 2026
simulator's update order. Use its GraphML outputs, not a different simulator.

No inputs are changed. No models are fitted. Outputs go into a NEW directory.
No inference about performance or the need for retraining follows from label
changes alone. An exact binary-label result can occur even with bounded counts.
PT deserialization requires torch and torch_geometric; load only trusted files.
"""
import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

STATES = {'U': 0, 'CS': 1, 'CR': 2, 'IS': 3, 'IR': 4}
PATTERN = re.compile(r'^(.*?)_t(\d+)(?:_L\d+)?$')


def integer(value, name):
    if hasattr(value, 'numel'):
        if value.numel() != 1:
            raise ValueError(f'{name}: expected a scalar')
        value = value.item()
    x = float(value)
    if not math.isfinite(x) or x < 0 or abs(x - round(x)) > 1e-6:
        raise ValueError(f'{name}: expected a non-negative integer, got {value}')
    return int(round(x))


def share(endogenous, imported):
    return endogenous / (endogenous + imported) if endogenous + imported else 0.0


def label(endogenous, imported):
    return int(endogenous + imported > 0 and endogenous >= imported)


def attrs(element, keys):
    return {keys[d.attrib['key']]: d.text or ''
            for d in element if d.tag.rsplit('}', 1)[-1] == 'data'}


def graphml_record(path):
    root = ET.parse(path).getroot()
    keys = {k.attrib['id']: k.attrib.get('attr.name', k.attrib['id'])
            for k in root if k.tag.rsplit('}', 1)[-1] == 'key'}
    graph = next(g for g in root if g.tag.rsplit('}', 1)[-1] == 'graph')
    meta = attrs(graph, keys)
    match = PATTERN.match(path.stem)
    if not match:
        raise ValueError(f'Unrecognised daily filename: {path}')
    day = integer(meta.get('day', match[2]), 'day')
    if day != int(match[2]):
        raise ValueError(f'Day metadata disagrees with filename: {path}')
    nodes = {n.attrib['id']: attrs(n, keys) for n in graph
             if n.tag.rsplit('}', 1)[-1] == 'node'}
    fields = ('new_trans_cr_total', 'new_select_cr_total',
              'new_import_cr_total', 'new_import_ir_total', 'new_ir_inf_total')
    counts = {k: integer(meta[k], k) for k in fields}
    return {'path': str(path), 'group': str(path.parent) + '/' + match[1],
            'day': day, 'meta': meta, 'nodes': nodes, 'counts': counts}


def event_bounds(current, previous, graphml):
    """Bound daily IS->IR events; distinguish progression before summing futures."""
    lo = hi = total = 0
    prior = previous['nodes'] if previous else {}
    meta = current.get('meta', {})
    for name, node in current['nodes'].items():
        event = integer(node['new_ir_inf_today'], 'new_ir_inf_today')
        if event not in (0, 1):
            raise ValueError('Expected binary node event flags')
        total += event
        if not event:
            continue
        if 'abx_class' in node and integer(node['abx_class'], 'abx_class') == 0:
            continue
        before = None
        if graphml:
            if 'superspreader_active' in meta:
                forced = (integer(meta['superspreader_active'], 'superspreader_active') == 1
                          and name == meta.get('superspreader_staff'))
                if forced:
                    raw = meta.get('superspreader_state', '').upper()
                    before = STATES.get(raw)
                    if before is None:
                        raise ValueError(f'Unknown forced superspreader state: {raw}')
                elif name in prior:
                    before = integer(prior[name]['amr_state'], 'amr_state')
            elif node.get('role') == 'patient' and name in prior:
                before = integer(prior[name]['amr_state'], 'amr_state')
            if before is None and node.get('role') == 'patient':
                admission_day = node.get('admission_day')
                if admission_day is not None and integer(admission_day, 'admission_day') == current['day']:
                    flags = [node.get('new_import_' + s.lower() + '_today')
                             for s in ('CS', 'CR', 'IS', 'IR')]
                    if all(v is not None for v in flags):
                        flags = [integer(v, 'import flag') for v in flags]
                        if any(v > 1 for v in flags) or sum(flags) > 1:
                            raise ValueError('Invalid admission import flags')
                        before = next((i + 1 for i, v in enumerate(flags) if v), 0)
        elif node.get('role') == 'patient' and name in prior:
            before = integer(prior[name]['amr_state'], 'amr_state')
        if before == 3:
            lo += 1
            hi += 1
        elif before is None:
            hi += 1
    return lo, hi, total


def pt_record(path, horizon, track, torch):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        data = torch.load(path, map_location='cpu')
    def get(key):
        value = data.get(key) if isinstance(data, dict) else getattr(data, key, None)
        if value is None:
            raise ValueError(f'{path}: missing {key}; use original daily data, not model checkpoints')
        return value
    match = PATTERN.match(path.stem)
    sim = data.get('sim_id') if isinstance(data, dict) else getattr(data, 'sim_id', None)
    day = data.get('day') if isinstance(data, dict) else getattr(data, 'day', None)
    if sim is None or day is None:
        if not match:
            raise ValueError(f'No trajectory/day metadata or recognised filename: {path}')
        sim, day = match[1], int(match[2])
    prefix = f'y_h{horizon}_'
    fields = ('trans_res', 'select_res', 'import_res', 'ir_inf', 'endog_majority')
    values = {k: integer(get(prefix + k), prefix + k) for k in fields}
    endogenous = values['trans_res'] + values['select_res']
    if values['endog_majority'] != label(endogenous, values['import_res']):
        raise ValueError(f'{path}: saved label does not match expected endpoint; audit stopped')
    result = {'path': str(path), 'group': str(path.parent) + '/' + str(sim),
              'day': integer(day, 'day'), 'values': values}
    if track == 'ground_truth':
        names = list(get('node_names'))
        x = get('x').tolist()
        if len(names) != len(x) or len(set(names)) != len(names):
            raise ValueError(f'{path}: invalid node identities')
        nodes = {}
        for name, row in zip(names, x):
            if len(row) == 6:
                state = integer(row[1], 'GT state')
                if state not in range(5):
                    raise ValueError(f'{path}: GT state must be in 0..4')
                antibiotic, ir_event = row[2], row[5]
            elif len(row) in (10, 22):
                if any(v not in (0.0, 1.0) for v in row[1:6]) or sum(row[1:6]) != 1:
                    raise ValueError(f'{path}: invalid GT one-hot states; use --pt-track aggregate')
                state = row[1:6].index(1.0)
                antibiotic, ir_event = row[6], row[9]
            else:
                raise ValueError(f'{path}: unsupported GT feature width {len(row)}; use --pt-track aggregate')
            antibiotic = integer(antibiotic, 'antibiotic class')
            ir_event = integer(ir_event, 'IR event flag')
            if ir_event not in (0, 1):
                raise ValueError(f'{path}: IR event flag must be binary')
            if row[0] not in (0, 1):
                raise ValueError('Unrecognised role encoding')
            nodes[str(name)] = {'role': 'staff' if row[0] else 'patient',
                                'amr_state': state,
                                'abx_class': antibiotic, 'new_ir_inf_today': ir_event}
        result['nodes'] = nodes
    return result


def audit_group(records, history, horizon, stride, mode, track):
    records.sort(key=lambda r: r['day'])
    days = [r['day'] for r in records]
    if any(b != a + 1 for a, b in zip(days, days[1:])):
        raise ValueError(f"Duplicate or missing days in {records[0]['group']}; supply a complete trajectory")
    previous = None
    for r in records:
        if (mode == 'graphml' or track == 'ground_truth') and 'bounds' not in r:
            lo, hi, total = event_bounds(r, previous, mode == 'graphml')
            r['bounds'] = (lo, hi, total)
            if mode == 'graphml' and total != r['counts']['new_ir_inf_total']:
                raise ValueError(f"Node and graph infection counters disagree: {r['path']}")
        previous = r
    rows = []
    for start in range(0, len(records) - history + 1, stride):
        i = start + history - 1
        r = records[i]
        future = records[i + 1:i + 1 + horizon]
        if mode == 'graphml':
            trans = sum(f['counts']['new_trans_cr_total'] for f in future)
            selection = sum(f['counts']['new_select_cr_total'] for f in future)
            imported = sum(f['counts']['new_import_cr_total'] + f['counts']['new_import_ir_total'] for f in future)
            ir = sum(f['counts']['new_ir_inf_total'] for f in future)
        else:
            v = r['values']
            trans, selection, imported, ir = [v[k] for k in ('trans_res', 'select_res', 'import_res', 'ir_inf')]
        if mode == 'graphml' or track == 'ground_truth':
            lo = sum(f['bounds'][0] for f in future)
            hi = sum(f['bounds'][1] for f in future)
            if sum(f['bounds'][2] for f in future) != ir:
                raise ValueError(f"Saved future infection count disagrees with GT node events: {r['path']}")
        else:
            lo, hi = 0, ir
        old_e = trans + selection
        old = label(old_e, imported)
        new_lo, new_hi = label(old_e + lo, imported), label(old_e + hi, imported)
        rows.append({'trajectory': r['group'], 'day': r['day'], 'source_file': r['path'],
                     'future_days': len(future), 'horizon_status': 'complete' if len(future) == horizon else 'truncated',
                     'transmission': trans, 'carriage_selection': selection, 'importation': imported,
                     'all_new_resistant_infections': ir, 'is_ir_lower': lo, 'is_ir_upper': hi,
                     'old_share': share(old_e, imported), 'new_share_lower': share(old_e + lo, imported),
                     'new_share_upper': share(old_e + hi, imported), 'old_label': old,
                     'new_label_lower': new_lo, 'new_label_upper': new_hi,
                     'definitely_changed': int(new_lo != old), 'possibly_changed': int(new_hi != old),
                     'label_resolved': int(new_lo == new_hi)})
    return rows


def summarize(rows):
    n = len(rows)
    lower = sum(r['definitely_changed'] for r in rows)
    upper = sum(r['possibly_changed'] for r in rows)
    return {'windows': n, 'old_endogenous': sum(r['old_label'] for r in rows),
            'changed_lower': lower, 'changed_upper': upper,
            'changed_percent_lower': 100 * lower / n if n else None,
            'changed_percent_upper': 100 * upper / n if n else None,
            'unresolved_labels': sum(1 - r['label_resolved'] for r in rows),
            'windows_with_exact_is_ir_count': sum(r['is_ir_lower'] == r['is_ir_upper'] for r in rows),
            'binary_label_change_count_exact': lower == upper}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--format', choices=('pt', 'graphml'), required=True)
    parser.add_argument('--pt-track', choices=('aggregate', 'ground_truth'), default='aggregate')
    parser.add_argument('--history', type=int, default=7)
    parser.add_argument('--horizon', type=int, default=7)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.history, args.horizon, args.stride) < 1:
        parser.error('History, horizon and stride must be positive')
    if args.output.exists():
        parser.error('Output directory already exists; choose a new name (nothing is overwritten)')
    torch = None
    if args.format == 'pt':
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Run PT mode in your AMR environment with torch and torch_geometric installed') from exc
    all_rows, summaries, trajectories = [], [], []
    names = set()
    roots = []
    for spec in args.input:
        if '=' not in spec:
            parser.error('--input must be NAME=PATH')
        name, directory = spec.split('=', 1)
        root = Path(directory).expanduser().resolve()
        if not name or name in names or not root.is_dir():
            parser.error(f'Invalid/duplicate input name or missing directory: {spec}')
        if any(root == p or root in p.parents or p in root.parents for p in roots):
            parser.error('Input directories overlap; supply separate cohorts to avoid double counting')
        names.add(name)
        roots.append(root)
        paths = sorted(root.rglob('*.' + args.format))
        if not paths:
            raise ValueError(f'No {args.format} files under {root}')
        print(f'{name}: reading {len(paths)} files', flush=True)
        groups = defaultdict(list)
        # Discover identities without retaining all graphs in memory.
        for j, path in enumerate(paths, 1):
            if args.format == 'graphml':
                match = PATTERN.match(path.stem)
                if not match:
                    raise ValueError(f'Unrecognised daily filename: {path}')
                group = str(path.parent) + '/' + match[1]
                groups[group].append(path)
            else:
                record = pt_record(path, args.horizon, 'aggregate', torch)
                groups[record['group']].append(record)
            if j % 1000 == 0:
                print(f'  indexed {j}/{len(paths)}', flush=True)
        cohort = []
        for group, entries in sorted(groups.items()):
            records = ([graphml_record(p) for p in entries] if args.format == 'graphml' else entries)
            if args.format == 'pt' and args.pt_track == 'ground_truth':
                previous = None
                for record in sorted(records, key=lambda r: r['day']):
                    full = pt_record(Path(record['path']), args.horizon, 'ground_truth', torch)
                    record['bounds'] = event_bounds(full, previous, False)
                    previous = full
            rows = audit_group(records, args.history, args.horizon, args.stride, args.format, args.pt_track)
            for row in rows:
                row['dataset'] = name
            cohort.extend(rows)
            trajectories.append({'dataset': name, 'trajectory': group, **summarize(rows)})
            for record in records:
                record.pop('nodes', None)
        if not cohort:
            raise ValueError(f'{name}: no complete history windows')
        for subset in ('all', 'complete', 'truncated'):
            selected = cohort if subset == 'all' else [r for r in cohort if r['horizon_status'] == subset]
            summaries.append({'dataset': name, 'horizon_status': subset, **summarize(selected)})
        result = summarize(cohort)
        print(f"{name}: {result['windows']} windows; changed labels "
              f"{result['changed_lower']} to {result['changed_upper']}; "
              f"unresolved {result['unresolved_labels']}", flush=True)
        all_rows.extend(cohort)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / 'windows.csv', all_rows)
    write_csv(args.output / 'summary.csv', summaries)
    write_csv(args.output / 'trajectories.csv', trajectories)
    with (args.output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump({'configuration': {**vars(args), 'output': str(args.output)},
                   'summary': summaries,
                   'interpretation': 'Changed counts are bounds, not estimates. Equal bounds give an exact binary-label change count. Overlapping windows are not independent. No training or performance audit was performed.'}, stream, indent=2)
    print(f'Reports saved to {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'AUDIT STOPPED: {error}', file=sys.stderr)
        sys.exit(1)
