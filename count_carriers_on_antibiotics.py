#!/usr/bin/env python3
"""Count antibiotic-exposed carrier-days or unique carriers in GT tensors.

Each saved daily graph is counted once, not once per overlapping window.
Percentage = exposed carrier-days / all carrier-days * 100.
With --unique, also count distinct people who ever carried (CS, CR, or either)
and the subset exposed to antibiotics on at least one day in that carrier state.
Each person is identified by (sim_id, node_name), never by tensor row position.
The supplied simulator assigns a fresh patient ID to each new admission.
Combined CS_and_CR counts are unions, not sums of the separate state counts.
This measures recorded antibiotic exposure, not effective treatment, and
describes states at the time each daily graph was saved, over the saved period
only. Antibiotics received solely outside carriage do not count as exposure.
Requires the project's torch and torch_geometric environment.
"""
import argparse
import csv
import math
from pathlib import Path


DEFAULT_ROOT = Path("experiments_results/TRACK_ground_truth/work")
DEFAULT_INPUTS = [
    f"canonical={DEFAULT_ROOT / 'synthetic_amr_graphs_test_frozen'}",
    f"shifted={DEFAULT_ROOT / 'synthetic_step8_shift_test'}",
]


def update_unique_counts(counts, person_roles, sid, node_names,
                         roles, states, antibiotics, source):
    """Accumulate sets of carriers and exposed carriers across daily snapshots."""
    if (not isinstance(node_names, (list, tuple))
            or len(node_names) != len(states)
            or any(not isinstance(node, str) or not node for node in node_names)):
        raise ValueError(
            f"Missing or invalid stable node_names: {source}. "
            "Unique patients cannot be recovered from tensor row positions; "
            "use GT tensors retaining the original node identities."
        )
    if len(set(node_names)) != len(node_names):
        raise ValueError(f"Duplicate node_names within one graph: {source}")
    if len(roles) != len(states) or len(antibiotics) != len(states):
        raise ValueError(f"Inconsistent node metadata lengths: {source}")
    for node, role, state, antibiotic in zip(node_names, roles, states, antibiotics):
        person = (str(sid), node)
        role_name = "staff" if role == 1 else "patients"
        previous_role = person_roles.setdefault(person, role_name)
        if previous_role != role_name:
            raise ValueError(f"Person changes role across daily graphs: {person}")
        if state not in (1, 2):
            continue
        state_name = "CS" if state == 1 else "CR"
        for population in (role_name, "all"):
            for carrier_state in (state_name, "CS_and_CR"):
                carriers, exposed_carriers = counts[population, carrier_state]
                carriers.add(person)
                if antibiotic > 0:
                    exposed_carriers.add(person)


def count_dataset(name, folder, unique=False):
    import torch

    files = sorted(folder.glob("*.pt"))
    if not files:
        raise ValueError(f"No daily .pt graphs found in {folder}")
    counts = {(role, state): [0, 0] for role in ("patients", "staff", "all")
              for state in ("CS", "CR", "CS_and_CR")}
    unique_counts = {key: [set(), set()] for key in counts} if unique else {}
    person_roles = {}
    seen = set()
    observed_states = set()
    for index, path in enumerate(files, 1):
        try:
            graph = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            graph = torch.load(path, map_location="cpu")
        sid, day = getattr(graph, "sim_id", None), getattr(graph, "day", None)
        if sid is None or day is None:
            raise ValueError(f"Missing trajectory/day metadata: {path}")
        key = (str(sid), int(day))
        if key in seen:
            raise ValueError(f"Duplicate trajectory/day {key}")
        seen.add(key)
        mode = str(getattr(graph, "state_mode", "")).lower()
        if "partial" in mode:
            raise ValueError(f"Partial-observation tensor cannot identify carriers: {path}")
        x = graph.x
        if x.ndim != 2 or x.shape[1] != 6 or not bool(torch.isfinite(x).all()):
            raise ValueError(f"Expected finite original six-feature GT tensor: {path}")
        role, state, antibiotic = x[:, 0], x[:, 1], x[:, 2]
        if not bool(((role == 0) | (role == 1)).all()):
            raise ValueError(f"Invalid role encoding: {path}")
        if not bool(((state == state.round()) & (state >= 0) & (state <= 4)).all()):
            raise ValueError(f"Invalid numeric AMR state encoding: {path}")
        if not bool(((antibiotic >= 0) & (antibiotic == antibiotic.round())).all()):
            raise ValueError(f"Invalid antibiotic-class encoding: {path}")
        observed_states.update(int(v) for v in state.unique().tolist())
        if unique:
            update_unique_counts(
                unique_counts, person_roles, sid, getattr(graph, "node_names", None),
                role.tolist(), state.tolist(), antibiotic.tolist(), path,
            )
        exposed = antibiotic > 0
        for role_name, role_mask in (("patients", role == 0), ("staff", role == 1),
                                      ("all", torch.ones_like(role, dtype=torch.bool))):
            for state_name, state_mask in (("CS", state == 1), ("CR", state == 2),
                                          ("CS_and_CR", (state == 1) | (state == 2))):
                mask = role_mask & state_mask
                counts[role_name, state_name][0] += int(mask.sum())
                counts[role_name, state_name][1] += int((mask & exposed).sum())
        if index % 1000 == 0:
            print(f"{name}: read {index}/{len(files)} graphs", flush=True)
    if not observed_states.intersection({2, 3, 4}):
        raise ValueError(f"{name}: cannot distinguish GT encoding from observed-positive encoding; stopped")
    rows = []
    for (role, state), (total, exposed) in counts.items():
        rows.append({"dataset": name, "population": role, "carrier_state": state,
                     "daily_graphs": len(files), "carrier_days": total,
                     "carrier_days_on_antibiotics": exposed,
                     "percent_on_antibiotics": 100 * exposed / total if total else ""})
        if unique:
            carriers, exposed_carriers = unique_counts[role, state]
            rows[-1].update({
                "unique_carriers": len(carriers),
                "unique_carriers_ever_on_antibiotics_while_carrying": len(exposed_carriers),
                "percent_unique_carriers_ever_on_antibiotics_while_carrying": (
                    100 * len(exposed_carriers) / len(carriers) if carriers else ""
                ),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", metavar="NAME=GT_FOLDER",
                        help="Repeat for each GT dataset; defaults to canonical and shifted test folders")
    parser.add_argument("--unique", action="store_true",
                        help="Also count distinct carriers using saved simulation and node identities")
    parser.add_argument("--output", default=None,
                        help="Default: carrier_antibiotic_summary.csv, or "
                             "carrier_antibiotic_unique_summary.csv with --unique")
    args = parser.parse_args()
    output = Path(args.output or (
        "carrier_antibiotic_unique_summary.csv" if args.unique else "carrier_antibiotic_summary.csv"
    ))
    if output.exists():
        raise ValueError(f"Output exists; choose another --output: {output}")
    rows, names = [], set()
    for entry in args.input or DEFAULT_INPUTS:
        name, separator, folder = entry.partition("=")
        if not separator or not name or not folder or name in names:
            raise ValueError("Each input must have a unique NAME=GT_FOLDER")
        names.add(name)
        rows.extend(count_dataset(name, Path(folder), unique=args.unique))
    with output.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        value = row["percent_unique_carriers_ever_on_antibiotics_while_carrying"
                    if args.unique else "percent_on_antibiotics"]
        percentage = f"{value:.2f}%" if isinstance(value, (int, float)) and math.isfinite(value) else "not applicable"
        detail = ""
        if args.unique:
            detail = (f" ({row['unique_carriers_ever_on_antibiotics_while_carrying']}/"
                      f"{row['unique_carriers']} unique carriers)")
        print(f"{row['dataset']} | {row['population']} | {row['carrier_state']}: {percentage}{detail}")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"COUNT STOPPED: {error}") from error
