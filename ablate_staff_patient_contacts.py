#!/usr/bin/env python3
"""Create a clinically relevant ablation hiding staff-patient contact detail.

The output keeps graph topology sufficient for temporal ordering but masks the
staff-patient mechanism by setting staff-patient edge weights to zero and adding
a graph-level marker. This is intended for Step 5-style tests of whether the
model relies on explicit staff-patient contact intensity.
"""
from pathlib import Path
import shutil
import networkx as nx

SRC_CANDIDATES = [Path("pt_data"), Path("step2_pt"), Path("data_pt")]
DST = Path("step5_ablation") / "hide_staff_patient_contacts"


def _find_src() -> Path:
    for src in SRC_CANDIDATES:
        if src.exists():
            return src
    raise FileNotFoundError("Could not find a source folder among: " + ", ".join(str(p) for p in SRC_CANDIDATES))


def main() -> None:
    src = _find_src()
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(src, DST)

    for path in DST.rglob("*.graphml"):
        graph = nx.read_graphml(path)
        for u, v, attrs in graph.edges(data=True):
            edge_type = int(float(attrs.get("edge_type", 0)))
            if edge_type in (1, 3):
                attrs["weight_original"] = float(attrs.get("weight", 1.0))
                attrs["weight"] = 0.0
                attrs["staff_patient_contact_hidden"] = 1
        graph.graph["ablation"] = "hide_staff_patient_contacts"
        nx.write_graphml(graph, path)


if __name__ == "__main__":
    main()
