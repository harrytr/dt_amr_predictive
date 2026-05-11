#!/usr/bin/env python3
"""
graph_folder_figures.py

Includes:
  - Single-folder summaries:
      * graph_summary_<label>.csv
      * figure_microgrid_<label>.png/.pdf
      * figure_distributions_<label>.png/.pdf
      * figure_communities_and_centrality_<label>.png/.pdf
  - Optional Train vs Test comparison (distribution shift):
      * graph_summary_combined.csv
      * figure_train_vs_test_shift.png/.pdf
      * figure_train_vs_test_ecdf.png/.pdf
      * train_vs_test_shift_stats.csv
  - Sankey-style network flow figure (aggregated across graphs):
      * figure_flow_sankey_<label>.png/.pdf
      * flow_matrix_<label>.csv
  - Timeline / dynamics figures (if day index can be parsed from filenames):
      * figure_timeline_nodes_edges_<label>.png/.pdf
      * figure_timeline_diff_test_minus_train.png/.pdf   (only when --compare_dir is provided)
  - State-composition timeline figures:
      * figure_state_percentages_<label>.png/.pdf
        (percentage of each node state per day/graph)
  - NEW: latex.txt
      * A LaTeX snippet that includes all PNGs produced in --out_dir with captions.

Community reporting:
  - Per-graph: n_communities, modularity, largest_comm_frac (in CSVs)
  - Plots: histogram of n_communities in figure_communities_and_centrality_<label>
  - Console: prints median/mean/min/max #communities per folder.

Sankey / flow:
  - Aggregates edge flow between node categories (e.g., ward->ward, role->role).
  - Category attribute chosen via --flow_attr:
        auto (default): ward if present else node_type/role else state
        ward | node_type | state
  - Edge weight uses first available among: weight, viral_load, w, edge_weight, vl
    otherwise edge count.

Timeline / dynamics:
  - Parses a day/time index from graph filename (stem) using patterns:
        *_day12_*, *_t12_*, *_d12_*
  - Plots mean/median nodes and edges per day and their day-to-day deltas (net change proxy).

Dependencies:
  pip install networkx matplotlib pandas numpy scipy

Parallel processing:
  - Per-file GraphML read + summary extraction can run across CPU cores.
  - Use --workers 1 to force serial execution.

Usage (single folder):
  python graph_folder_figures.py --graph_dir synthetic_amr_graphs_train --out_dir figs --identity "Harry Triantafyllidis"

Usage (train vs test):
  python graph_folder_figures.py --graph_dir synthetic_amr_graphs_train --compare_dir synthetic_amr_graphs_test --out_dir figs --identity "Harry Triantafyllidis"
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import random

try:
    from scipy.stats import ks_2samp
except Exception:
    ks_2samp = None


class ProgressBar:
    def __init__(self, total: int, width: int = 32, prefix: str = "PROGRESS") -> None:
        self.total = max(int(total), 1)
        self.width = max(int(width), 10)
        self.prefix = prefix
        self.current = 0

    def show(self, message: str = "") -> None:
        filled = int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        msg = f"{self.prefix} [{bar}] {self.current}/{self.total}"
        if message:
            msg += f" | {message}"
        print(msg, flush=True)

    def advance(self, step: int = 1, message: str = "") -> None:
        self.current = min(self.total, self.current + int(step))
        self.show(message)


# -------------------------- Color utilities (identity-hashed) --------------------------

def _sha256_int(seed_text: str) -> int:
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0.0:
            t += 1.0
        if t > 1.0:
            t -= 1.0
        if t < 1.0 / 6.0:
            return p + (q - p) * 6.0 * t
        if t < 1.0 / 2.0:
            return q
        if t < 2.0 / 3.0:
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0
        return p

    if s == 0.0:
        return (l, l, l)

    q = l * (1.0 + s) if l < 0.5 else (l + s - l * s)
    p = 2.0 * l - q
    r = hue_to_rgb(p, q, h + 1.0 / 3.0)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1.0 / 3.0)
    return (r, g, b)


def make_identity_palette(identity: str, n: int = 10) -> Dict[str, Any]:
    seed_int = _sha256_int(identity)
    rng = np.random.default_rng(seed_int)

    base_h = (seed_int % 360) / 360.0
    hues = (base_h + np.linspace(0.0, 0.90, n, endpoint=False)) % 1.0

    sat_base = 0.64 + 0.10 * (rng.random() - 0.5)
    lig_base = 0.48 + 0.10 * (rng.random() - 0.5)

    colors: List[Tuple[float, float, float]] = []
    for h in hues:
        s = float(np.clip(sat_base + 0.06 * (rng.random() - 0.5), 0.45, 0.82))
        l = float(np.clip(lig_base + 0.06 * (rng.random() - 0.5), 0.34, 0.64))
        colors.append(_hsl_to_rgb(float(h), s, l))

    primary = colors[0]
    accent = colors[2] if len(colors) > 2 else colors[-1]
    muted = _hsl_to_rgb(base_h, 0.20, 0.55)
    return {"colors": colors, "primary": primary, "accent": accent, "muted": muted, "seed_int": seed_int}


# -------------------------- Graph loading & candidates --------------------------

_WEIGHT_CANDIDATES = ("weight", "viral_load", "w", "edge_weight", "vl")
_NODETYPE_CANDIDATES = ("node_type", "type", "role")
_STATE_CANDIDATES = ("amr_state", "state", "status")
_WARD_CANDIDATES = ("ward", "ward_id", "location", "unit")

# Timeline parsing from filenames
_DAY_PATTERNS = [
    re.compile(r"(?:^|[_\-])day[_\-]?(\d+)(?:$|[_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])t[_\-]?(\d+)(?:$|[_\-])", re.IGNORECASE),
    re.compile(r"(?:^|[_\-])d[_\-]?(\d+)(?:$|[_\-])", re.IGNORECASE),
]


def parse_day_from_filename(fp: Path) -> Optional[int]:
    """
    Try to parse a day/time index from filename stem.
    Supports patterns like: *_day12_*, *_t12_*, *_d12_*.
    Returns int day or None if not found.
    """
    stem = fp.stem
    for pat in _DAY_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def find_graphml_files(graph_dir: Path) -> List[Path]:
    return sorted([p for p in graph_dir.rglob("*.graphml") if p.is_file()])


def _resolve_max_graphs_count(max_graphs: Any, n_files: int) -> int:
    if n_files <= 0:
        return 0
    if max_graphs is None:
        return n_files

    spec = str(max_graphs).strip().lower()
    if spec in {"", "0", "0%", "all", "100%"}:
        return n_files

    if spec.endswith("%"):
        try:
            pct = float(spec[:-1].strip())
        except Exception as exc:
            raise ValueError(f"Invalid --max_graphs percentage: {max_graphs}") from exc
        if pct <= 0.0:
            raise ValueError("--max_graphs percentage must be > 0")
        if pct >= 100.0:
            return n_files
        return max(1, min(n_files, int(math.ceil((pct / 100.0) * n_files))))

    try:
        count = int(spec)
    except Exception as exc:
        raise ValueError(f"Invalid --max_graphs value: {max_graphs}") from exc
    if count <= 0:
        return n_files
    return min(n_files, count)


def _sample_graph_files(files: List[Path], max_graphs: Any, seed: int) -> List[Path]:
    if not files:
        return files
    keep_n = _resolve_max_graphs_count(max_graphs, len(files))
    if keep_n >= len(files):
        return list(files)

    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(files), size=keep_n, replace=False))
    return [files[int(i)] for i in idx]


def _to_simple_graph(G: nx.Graph) -> nx.Graph:
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return G

    is_directed = G.is_directed()
    H = nx.DiGraph() if is_directed else nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        if H.has_edge(u, v):
            existing = H[u][v]
            for k, val in data.items():
                if k not in existing:
                    existing[k] = val
                else:
                    if isinstance(existing[k], (int, float)) and isinstance(val, (int, float)):
                        existing[k] = float(existing[k]) + float(val)
                    else:
                        existing[k] = val
        else:
            H.add_edge(u, v, **data)
    return H


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _detect_edge_weight_attr(G: nx.Graph) -> Optional[str]:
    for _, _, d in G.edges(data=True):
        for cand in _WEIGHT_CANDIDATES:
            if cand in d:
                return cand
    return None


def extract_edge_weights(G: nx.Graph) -> Optional[np.ndarray]:
    attr_name = _detect_edge_weight_attr(G)
    if attr_name is None:
        return None
    vals: List[float] = []
    for _, _, d in G.edges(data=True):
        v = _safe_float(d.get(attr_name))
        if v is not None and math.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return np.asarray(vals, dtype=float)


def extract_node_attr_counts(G: nx.Graph, candidates: Tuple[str, ...]) -> Optional[Dict[str, int]]:
    attr_name = None
    for _, d in G.nodes(data=True):
        for cand in candidates:
            if cand in d:
                attr_name = cand
                break
        if attr_name is not None:
            break
    if attr_name is None:
        return None

    counts: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        v = d.get(attr_name, None)
        key = "NA" if v is None else str(v)
        counts[key] = counts.get(key, 0) + 1
    return counts


def largest_weak_component_subgraph(G: nx.Graph) -> nx.Graph:
    comps = nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    comps_list = list(comps)
    if not comps_list:
        return G
    biggest = max(comps_list, key=len)
    return G.subgraph(biggest).copy()


# -------------------------- Community + centrality helpers --------------------------

def compute_communities_and_modularity(
    Gu: nx.Graph,
    max_nodes_for_full: int = 6000,
    rng_seed: int = 123
) -> Tuple[Optional[List[set]], float]:
    try:
        H = Gu
        if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
            return None, float("nan")

        if H.number_of_nodes() > max_nodes_for_full:
            rng = np.random.default_rng(rng_seed)
            nodes = list(H.nodes())
            keep = set(rng.choice(nodes, size=max_nodes_for_full, replace=False).tolist())
            H = H.subgraph(keep).copy()

        communities = list(nx.algorithms.community.greedy_modularity_communities(H))
        mod = float(nx.algorithms.community.modularity(H, communities))
        return communities, mod
    except Exception:
        return None, float("nan")


def betweenness_centrality_safe(
    G: nx.Graph,
    exact_max_nodes: int = 2500,
    sample_k: int = 300,
    rng_seed: int = 123
) -> Optional[Dict[Any, float]]:
    try:
        n = G.number_of_nodes()
        if n == 0:
            return None
        if n <= exact_max_nodes:
            return nx.betweenness_centrality(G, normalized=True)
        return nx.betweenness_centrality(G, k=min(sample_k, n), normalized=True, seed=rng_seed)
    except Exception:
        return None


def pagerank_safe(G: nx.DiGraph) -> Optional[Dict[Any, float]]:
    try:
        if G.number_of_nodes() == 0:
            return None
        return nx.pagerank(G, alpha=0.85)
    except Exception:
        return None


def eigenvector_centrality_safe(Gu: nx.Graph, max_nodes: int = 4000) -> Optional[Dict[Any, float]]:
    try:
        n = Gu.number_of_nodes()
        if n == 0:
            return None
        if n > max_nodes:
            return None
        return nx.eigenvector_centrality(Gu, max_iter=2000, tol=1e-06)
    except Exception:
        return None


# -------------------------- Stats per graph --------------------------

def compute_graph_stats(
    G_in: nx.Graph,
    bc_exact_max_nodes: int,
    bc_sample_k: int,
    comm_max_nodes: int,
    rng_seed: int
) -> Dict[str, Any]:
    G0 = _to_simple_graph(G_in)
    G = G0.copy()

    n = G.number_of_nodes()
    m = G.number_of_edges()
    is_directed = G.is_directed()

    density = nx.density(G) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if (n > 0 and not is_directed) else (m / n if n > 0 else 0.0)
    avg_in_degree = float(np.mean([d for _, d in G.in_degree()])) if (is_directed and n > 0) else np.nan
    avg_out_degree = float(np.mean([d for _, d in G.out_degree()])) if (is_directed and n > 0) else np.nan
    reciprocity = nx.reciprocity(G) if (is_directed and m > 0) else np.nan

    Gu_full = G.to_undirected(as_view=False) if is_directed else G
    avg_clustering = nx.average_clustering(Gu_full) if n > 2 else np.nan
    transitivity = nx.transitivity(Gu_full) if n > 2 else np.nan
    try:
        assort = nx.degree_assortativity_coefficient(Gu_full) if n > 3 else np.nan
    except Exception:
        assort = np.nan

    Gc = largest_weak_component_subgraph(G)
    gc_n = Gc.number_of_nodes()
    gc_m = Gc.number_of_edges()

    diam = np.nan
    asp = np.nan
    try:
        if gc_n >= 2:
            H = Gc.to_undirected(as_view=False) if is_directed else Gc
            if gc_n <= 2000 and nx.is_connected(H):
                diam = float(nx.diameter(H))
                asp = float(nx.average_shortest_path_length(H))
            elif gc_n > 2000:
                nodes = list(H.nodes())
                rng = np.random.default_rng(rng_seed)
                sample = rng.choice(nodes, size=min(200, len(nodes)), replace=False)
                eccs = []
                for s in sample:
                    lengths = nx.single_source_shortest_path_length(H, s)
                    if lengths:
                        eccs.append(max(lengths.values()))
                if eccs:
                    diam = float(np.percentile(eccs, 95))
    except Exception:
        pass

    w = extract_edge_weights(G)
    w_mean = float(np.mean(w)) if w is not None else np.nan
    w_std = float(np.std(w)) if w is not None else np.nan
    w_p95 = float(np.percentile(w, 95)) if (w is not None and len(w) > 1) else np.nan

    node_type_counts = extract_node_attr_counts(G, _NODETYPE_CANDIDATES)
    state_counts = extract_node_attr_counts(G, _STATE_CANDIDATES)
    ward_counts = extract_node_attr_counts(G, _WARD_CANDIDATES)

    deg_arr = np.array([d for _, d in G.degree()], dtype=float) if n > 0 else np.array([])
    deg_max = float(np.max(deg_arr)) if deg_arr.size else 0.0

    # Centralities on largest component
    Gc_u = Gc.to_undirected(as_view=False) if is_directed else Gc
    bc = betweenness_centrality_safe(Gc_u, exact_max_nodes=bc_exact_max_nodes, sample_k=bc_sample_k, rng_seed=rng_seed)
    pr = pagerank_safe(Gc if is_directed else nx.DiGraph(Gc_u))
    ev = eigenvector_centrality_safe(Gc_u)

    bc_vals = np.array(list(bc.values()), dtype=float) if bc else np.array([])
    pr_vals = np.array(list(pr.values()), dtype=float) if pr else np.array([])
    ev_vals = np.array(list(ev.values()), dtype=float) if ev else np.array([])

    def _summ(arr: np.ndarray) -> Tuple[float, float, float]:
        if arr.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        mean = float(np.mean(arr))
        p95 = float(np.percentile(arr, 95)) if arr.size > 1 else float("nan")
        mx = float(np.max(arr))
        return (mean, p95, mx)

    bc_mean, bc_p95, bc_max = _summ(bc_vals)
    pr_mean, pr_p95, pr_max = _summ(pr_vals)
    ev_mean, ev_p95, ev_max = _summ(ev_vals)

    communities, modularity = compute_communities_and_modularity(Gc_u, max_nodes_for_full=comm_max_nodes, rng_seed=rng_seed)
    n_communities = int(len(communities)) if communities else 0
    comm_sizes = [len(c) for c in communities] if communities else []
    largest_comm_frac = (max(comm_sizes) / float(sum(comm_sizes))) if comm_sizes else np.nan

    return {
        "n_nodes": int(n),
        "n_edges": int(m),
        "directed": bool(is_directed),
        "density": float(density),
        "avg_degree": float(avg_degree),
        "avg_in_degree": float(avg_in_degree) if is_directed else np.nan,
        "avg_out_degree": float(avg_out_degree) if is_directed else np.nan,
        "reciprocity": float(reciprocity) if is_directed else np.nan,
        "avg_clustering": float(avg_clustering) if not np.isnan(avg_clustering) else np.nan,
        "transitivity": float(transitivity) if not np.isnan(transitivity) else np.nan,
        "assortativity": float(assort) if not np.isnan(assort) else np.nan,
        "gc_nodes": int(gc_n),
        "gc_edges": int(gc_m),
        "diameter_or_p95_ecc": float(diam) if not np.isnan(diam) else np.nan,
        "avg_shortest_path": float(asp) if not np.isnan(asp) else np.nan,
        "deg_max": float(deg_max),
        "w_mean": w_mean,
        "w_std": w_std,
        "w_p95": w_p95,
        "node_type_counts": node_type_counts,
        "state_counts": state_counts,
        "ward_counts": ward_counts,
        "deg_arr": deg_arr,
        "w_arr": w if w is not None else np.array([]),
        "bc_arr": bc_vals,
        "pr_arr": pr_vals,
        "ev_arr": ev_vals,
        "bc_mean": bc_mean,
        "bc_p95": bc_p95,
        "bc_max": bc_max,
        "pr_mean": pr_mean,
        "pr_p95": pr_p95,
        "pr_max": pr_max,
        "ev_mean": ev_mean,
        "ev_p95": ev_p95,
        "ev_max": ev_max,
        "n_communities": n_communities,
        "modularity": float(modularity) if not np.isnan(modularity) else np.nan,
        "largest_comm_frac": float(largest_comm_frac) if not np.isnan(largest_comm_frac) else np.nan,
        "comm_sizes": np.array(comm_sizes, dtype=float) if comm_sizes else np.array([]),
    }




def _process_graph_file_worker(
    fp_str: str,
    bc_exact_max_nodes: int,
    bc_sample_k: int,
    comm_max_nodes: int,
    rng_seed: int,
) -> Dict[str, Any]:
    fp = Path(fp_str)
    day_idx = parse_day_from_filename(fp)

    try:
        G = nx.read_graphml(fp)
    except Exception as e:
        return {
            "row": {"file": str(fp), "read_error": str(e), "day": day_idx},
            "pooled": None,
            "error": True,
        }

    st = compute_graph_stats(
        G,
        bc_exact_max_nodes=bc_exact_max_nodes,
        bc_sample_k=bc_sample_k,
        comm_max_nodes=comm_max_nodes,
        rng_seed=rng_seed,
    )

    row = {
        "file": str(fp),
        "day": day_idx,
        "n_nodes": st["n_nodes"],
        "n_edges": st["n_edges"],
        "directed": st["directed"],
        "density": st["density"],
        "avg_degree": st["avg_degree"],
        "avg_in_degree": st["avg_in_degree"],
        "avg_out_degree": st["avg_out_degree"],
        "reciprocity": st["reciprocity"],
        "avg_clustering": st["avg_clustering"],
        "transitivity": st["transitivity"],
        "assortativity": st["assortativity"],
        "gc_nodes": st["gc_nodes"],
        "gc_edges": st["gc_edges"],
        "diameter_or_p95_ecc": st["diameter_or_p95_ecc"],
        "avg_shortest_path": st["avg_shortest_path"],
        "deg_max": st["deg_max"],
        "w_mean": st["w_mean"],
        "w_std": st["w_std"],
        "w_p95": st["w_p95"],
        "bc_mean": st["bc_mean"],
        "bc_p95": st["bc_p95"],
        "bc_max": st["bc_max"],
        "pr_mean": st["pr_mean"],
        "pr_p95": st["pr_p95"],
        "pr_max": st["pr_max"],
        "ev_mean": st["ev_mean"],
        "ev_p95": st["ev_p95"],
        "ev_max": st["ev_max"],
        "n_communities": st["n_communities"],
        "modularity": st["modularity"],
        "largest_comm_frac": st["largest_comm_frac"],
    }

    state_counts = st.get("state_counts") or {}
    total_states = int(sum(state_counts.values()))
    if total_states > 0:
        for state_name, state_count in state_counts.items():
            norm_state = _normalise_state_label(state_name)
            row[f"state_pct__{norm_state}"] = 100.0 * float(state_count) / float(total_states)

    pooled = {
        "n_communities_all": [float(st["n_communities"])],
        "deg_all": [float(x) for x in st["deg_arr"]] if isinstance(st.get("deg_arr"), np.ndarray) and st["deg_arr"].size > 0 else [],
        "w_all": [float(x) for x in st["w_arr"]] if isinstance(st.get("w_arr"), np.ndarray) and st["w_arr"].size > 0 else [],
        "comm_sizes_all": [float(x) for x in st["comm_sizes"]] if isinstance(st.get("comm_sizes"), np.ndarray) and st["comm_sizes"].size > 0 else [],
        "bc_all": [float(x) for x in st["bc_arr"]] if isinstance(st.get("bc_arr"), np.ndarray) and st["bc_arr"].size > 0 else [],
        "pr_all": [float(x) for x in st["pr_arr"]] if isinstance(st.get("pr_arr"), np.ndarray) and st["pr_arr"].size > 0 else [],
        "ev_all": [float(x) for x in st["ev_arr"]] if isinstance(st.get("ev_arr"), np.ndarray) and st["ev_arr"].size > 0 else [],
        "node_type_counts_all": st.get("node_type_counts") or {},
        "state_counts_all": st.get("state_counts") or {},
        "ward_counts_all": st.get("ward_counts") or {},
    }
    return {"row": row, "pooled": pooled, "error": False}


def _merge_pooled_accumulator(target: Dict[str, Any], piece: Optional[Dict[str, Any]]) -> None:
    if not piece:
        return
    for list_key in ("deg_all", "w_all", "comm_sizes_all", "bc_all", "pr_all", "ev_all", "n_communities_all"):
        vals = piece.get(list_key) or []
        if vals:
            target[list_key].extend(vals)
    for src_key, tgt_key in (
        ("node_type_counts_all", "node_type_counts_all"),
        ("state_counts_all", "state_counts_all"),
        ("ward_counts_all", "ward_counts_all"),
    ):
        d_src = piece.get(src_key) or {}
        d_tgt = target[tgt_key]
        for k, v in d_src.items():
            d_tgt[k] = d_tgt.get(k, 0) + int(v)


def _resolve_worker_count(requested: int, n_files: int) -> int:
    if n_files <= 1:
        return 1
    if requested is None or int(requested) <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(n_files, cpu))
    return max(1, min(n_files, int(requested)))

# -------------------------- Plotting basics --------------------------

def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _hist(ax, data: np.ndarray, bins: int, color: Tuple[float, float, float], title: str, xlabel: str) -> None:
    if data.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.hist(data, bins=bins, color=color, alpha=0.65, edgecolor="black", linewidth=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def _bar_topk(ax, counts: Dict[str, int], color: Tuple[float, float, float], title: str, k: int = 8) -> None:
    if not counts:
        ax.text(0.5, 0.5, "Attribute not present", ha="center", va="center")
        ax.set_axis_off()
        return
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    labels = [a for a, _ in items]
    vals = [b for _, b in items]
    ax.bar(range(len(vals)), vals, color=color, alpha=0.90, edgecolor="black", linewidth=0.3)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Nodes")


def make_microgrid_figure(
    df: pd.DataFrame,
    deg_all: np.ndarray,
    w_all: np.ndarray,
    node_type_counts_all: Dict[str, int],
    state_counts_all: Dict[str, int],
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    suptitle: str
) -> None:
    colors = palette["colors"]
    primary = palette["primary"]
    accent = palette["accent"]
    muted = palette["muted"]

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 4, wspace=0.28, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _hist(ax1, df["n_nodes"].to_numpy(dtype=float), bins=18, color=primary, title="Nodes per graph", xlabel="# nodes")

    ax2 = fig.add_subplot(gs[0, 1])
    _hist(ax2, df["n_edges"].to_numpy(dtype=float), bins=18, color=colors[1], title="Edges per graph", xlabel="# edges")

    ax3 = fig.add_subplot(gs[0, 2])
    _hist(ax3, df["density"].to_numpy(dtype=float), bins=18, color=colors[2], title="Density", xlabel="density")

    ax4 = fig.add_subplot(gs[0, 3])
    _hist(ax4, df["avg_degree"].to_numpy(dtype=float), bins=18, color=colors[3], title="Average degree", xlabel="avg degree")

    ax5 = fig.add_subplot(gs[1, 0])
    if deg_all.size > 0:
        kmax = max(1.0, float(np.max(deg_all)))
        bins = np.unique(np.round(np.logspace(0, math.log10(kmax + 1.0), 20))).astype(int)
        bins = bins[bins >= 1]
        ax5.hist(deg_all, bins=bins, color=accent, alpha=0.75, edgecolor="black", linewidth=0.25)
        ax5.set_xscale("log")
        ax5.set_title("Pooled degree distribution (log-x)")
        ax5.set_xlabel("degree")
        ax5.set_ylabel("Count")
    else:
        ax5.text(0.5, 0.5, "No degree data", ha="center", va="center")
        ax5.set_axis_off()

    ax6 = fig.add_subplot(gs[1, 1])
    if w_all.size > 0:
        _hist(ax6, w_all, bins=24, color=colors[4], title="Pooled edge weight distribution", xlabel="edge weight")
    else:
        ax6.text(0.5, 0.5, "No edge weights detected", ha="center", va="center")
        ax6.set_axis_off()

    ax7 = fig.add_subplot(gs[1, 2])
    _bar_topk(ax7, node_type_counts_all, color=muted, title="Node types (top)")

    ax8 = fig.add_subplot(gs[1, 3])
    _bar_topk(ax8, state_counts_all, color=colors[5], title="Node states (top)")

    fig.suptitle(f"{suptitle}\nIdentity palette seed={palette['seed_int']}  |  graphs={len(df)}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_distributions_figure(
    df: pd.DataFrame,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    suptitle: str
) -> None:
    colors = palette["colors"]
    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _hist(ax1, df["avg_clustering"].dropna().to_numpy(dtype=float), bins=18, color=colors[0],
          title="Avg clustering (undirected projection)", xlabel="avg clustering")

    ax2 = fig.add_subplot(gs[0, 1])
    _hist(ax2, df["transitivity"].dropna().to_numpy(dtype=float), bins=18, color=colors[1],
          title="Transitivity", xlabel="transitivity")

    ax3 = fig.add_subplot(gs[0, 2])
    _hist(ax3, df["assortativity"].dropna().to_numpy(dtype=float), bins=18, color=colors[2],
          title="Degree assortativity", xlabel="assortativity")

    ax4 = fig.add_subplot(gs[1, 0])
    _hist(ax4, df["reciprocity"].dropna().to_numpy(dtype=float), bins=18, color=colors[3],
          title="Reciprocity (directed only)", xlabel="reciprocity")

    ax5 = fig.add_subplot(gs[1, 1])
    _hist(ax5, df["gc_nodes"].to_numpy(dtype=float), bins=18, color=colors[4],
          title="Largest component size", xlabel="# nodes in largest component")

    ax6 = fig.add_subplot(gs[1, 2])
    _hist(ax6, df["diameter_or_p95_ecc"].dropna().to_numpy(dtype=float), bins=18, color=colors[5],
          title="Diameter (or ~p95 eccentricity)", xlabel="distance")

    fig.suptitle(f"{suptitle} (structural distributions)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_communities_and_centrality_figure(
    df: pd.DataFrame,
    comm_sizes_all: np.ndarray,
    bc_all: np.ndarray,
    pr_all: np.ndarray,
    ev_all: np.ndarray,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    suptitle: str
) -> None:
    colors = palette["colors"]
    primary = palette["primary"]

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 4, wspace=0.30, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _hist(ax1, df["n_communities"].to_numpy(dtype=float), bins=18, color=primary,
          title="# communities (per graph)", xlabel="# communities")

    ax2 = fig.add_subplot(gs[0, 1])
    _hist(ax2, df["modularity"].dropna().to_numpy(dtype=float), bins=18, color=colors[1],
          title="Modularity (largest component)", xlabel="modularity")

    ax3 = fig.add_subplot(gs[0, 2])
    _hist(ax3, df["largest_comm_frac"].dropna().to_numpy(dtype=float), bins=18, color=colors[2],
          title="Largest community fraction", xlabel="fraction")

    ax4 = fig.add_subplot(gs[0, 3])
    if comm_sizes_all.size > 0:
        kmax = max(2.0, float(np.max(comm_sizes_all)))
        bins = np.unique(np.round(np.logspace(0, math.log10(kmax + 1.0), 22))).astype(int)
        bins = bins[bins >= 1]
        ax4.hist(comm_sizes_all, bins=bins, color=colors[3], alpha=0.75, edgecolor="black", linewidth=0.25)
        ax4.set_xscale("log")
        ax4.set_title("Community sizes (pooled, log-x)")
        ax4.set_xlabel("community size")
        ax4.set_ylabel("Count")
    else:
        ax4.text(0.5, 0.5, "No community data", ha="center", va="center")
        ax4.set_axis_off()

    ax5 = fig.add_subplot(gs[1, 0])
    _hist(ax5, bc_all, bins=24, color=colors[4], title="Betweenness (pooled)", xlabel="betweenness") if bc_all.size else ax5.set_axis_off()

    ax6 = fig.add_subplot(gs[1, 1])
    _hist(ax6, pr_all, bins=24, color=colors[5], title="PageRank (pooled)", xlabel="pagerank") if pr_all.size else ax6.set_axis_off()

    ax7 = fig.add_subplot(gs[1, 2])
    _hist(ax7, ev_all, bins=24, color=colors[6], title="Eigenvector (pooled)", xlabel="eigenvector") if ev_all.size else ax7.set_axis_off()

    ax8 = fig.add_subplot(gs[1, 3])
    if df["modularity"].notna().any():
        x = df["n_nodes"].to_numpy(dtype=float)
        y = df["modularity"].to_numpy(dtype=float)
        ax8.scatter(x, y, s=18, alpha=0.85, color=colors[7] if len(colors) > 7 else colors[-1],
                    edgecolor="black", linewidth=0.2)
        ax8.set_xscale("log")
        ax8.set_title("Modularity vs size")
        ax8.set_xlabel("# nodes (log)")
        ax8.set_ylabel("modularity")
    else:
        ax8.set_axis_off()

    fig.suptitle(f"{suptitle} (communities + centrality)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------------- Train vs Test shift plots --------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    na = a.size
    nb = b.size
    pooled = ((na - 1) * va + (nb - 1) * vb) / float(max(na + nb - 2, 1))
    if pooled <= 0.0:
        return float("nan")
    return (float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled)


def _ks_pvalue(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if ks_2samp is None or a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    res = ks_2samp(a, b, alternative="two-sided", mode="auto")
    return (float(res.statistic), float(res.pvalue))


def _hist_overlay(ax, a: np.ndarray, b: np.ndarray, bins, ca, cb, title: str, xlabel: str, la: str, lb: str) -> None:
    if a.size == 0 and b.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return
    if a.size > 0:
        ax.hist(a, bins=bins, color=ca, alpha=0.55, edgecolor="black", linewidth=0.25, label=la)
    if b.size > 0:
        ax.hist(b, bins=bins, color=cb, alpha=0.45, edgecolor="black", linewidth=0.25, label=lb)
    title_map = {
        "assortativity": "Assortativity",
        "bc_p95": "Betweenness p95",
    }
    t = title_map.get(title, title)
    ax.set_title(t, fontsize=9, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(frameon=False)


def _ecdf(ax, x: np.ndarray, color, label: str) -> None:
    if x.size == 0:
        return
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    ax.plot(xs, ys, color=color, linewidth=1.7, label=label)


def make_train_vs_test_shift_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    title: str,
    label_train: str,
    label_test: str
) -> pd.DataFrame:
    colors = palette["colors"]
    c_train = palette["primary"]
    c_test = colors[4] if len(colors) > 4 else palette["accent"]

    metrics = [
        ("n_nodes", "# nodes"),
        ("n_edges", "# edges"),
        ("density", "density"),
        ("avg_degree", "avg degree"),
        ("avg_clustering", "avg clustering"),
        ("transitivity", "transitivity"),
        ("assortativity", "assortativity"),
        ("reciprocity", "reciprocity"),
        ("modularity", "modularity"),
        ("n_communities", "# communities"),
        ("bc_p95", "betweenness p95"),
        ("pr_p95", "pagerank p95"),
        ("w_mean", "mean edge weight"),
        ("w_p95", "edge weight p95"),
    ]

    fig = plt.figure(figsize=(16.0, 10.0))
    gs = fig.add_gridspec(4, 4, wspace=0.30, hspace=0.40)

    stats_rows: List[Dict[str, Any]] = []
    for i, (col, xlabel) in enumerate(metrics[:16]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.title.set_wrap(False)
        a = train_df[col].dropna().to_numpy(dtype=float) if col in train_df.columns else np.array([])
        b = test_df[col].dropna().to_numpy(dtype=float) if col in test_df.columns else np.array([])

        combined = np.concatenate([a, b]) if (a.size and b.size) else (a if a.size else b)
        if combined.size == 0:
            ax.set_axis_off()
            continue

        lo = float(np.percentile(combined, 1))
        hi = float(np.percentile(combined, 99))
        if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
            lo = float(np.min(combined))
            hi = float(np.max(combined))

        if hi > lo:
            bin_edges = np.linspace(lo, hi, 21)
        else:
            bin_edges = 20

        _hist_overlay(ax, a, b, bins=bin_edges, ca=c_train, cb=c_test,
                      title=col, xlabel=xlabel, la=label_train, lb=label_test)

        if col in ("n_edges", "avg_degree", "bc_p95"):
            ax.set_xscale("log")
            ax.tick_params(axis="x", labelrotation=30)
            ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(2.0, 5.0)))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())

        if col == "assortativity":
            ax.tick_params(axis="x", labelrotation=30)
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        d = _cohens_d(a, b)
        ks_stat, ks_p = _ks_pvalue(a, b)
        ax.text(
            0.02, 0.98,
            f"Δμ={np.mean(a) - np.mean(b):.3g}\nd={d:.3g}\nKS={ks_stat:.3g}\np={ks_p:.2g}",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.3, alpha=0.9)
        )

        stats_rows.append({
            "metric": col,
            "train_mean": float(np.mean(a)) if a.size else np.nan,
            "test_mean": float(np.mean(b)) if b.size else np.nan,
            "delta_mean_train_minus_test": float(np.mean(a) - float(np.mean(b))) if (a.size and b.size) else np.nan,
            "cohens_d_train_minus_test": d,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "train_n": int(a.size),
            "test_n": int(b.size),
        })

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(stats_rows)


def make_train_vs_test_ecdf_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    title: str,
    label_train: str,
    label_test: str
) -> None:
    colors = palette["colors"]
    c_train = palette["primary"]
    c_test = colors[4] if len(colors) > 4 else palette["accent"]

    metrics = [
        ("n_nodes", "# nodes"),
        ("n_edges", "# edges"),
        ("avg_degree", "avg degree"),
        ("density", "density"),
        ("modularity", "modularity"),
        ("bc_p95", "betweenness p95"),
        ("w_mean", "mean edge weight"),
        ("w_p95", "edge weight p95"),
    ]

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 4, wspace=0.30, hspace=0.38)

    for i, (col, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        a = train_df[col].dropna().to_numpy(dtype=float) if col in train_df.columns else np.array([])
        b = test_df[col].dropna().to_numpy(dtype=float) if col in test_df.columns else np.array([])

        if a.size == 0 and b.size == 0:
            ax.set_axis_off()
            continue

        _ecdf(ax, a, c_train, label_train)
        _ecdf(ax, b, c_test, label_test)
        ax.set_title(col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ECDF")
        ax.legend(frameon=False)

        if col in ("n_nodes", "n_edges", "avg_degree"):
            ax.set_xscale("log")
            ax.tick_params(axis="x", labelrotation=30)
            ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=(2.0, 5.0)))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------------- Timeline / dynamics figures --------------------------

def make_timeline_nodes_edges_figure(
    df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str
) -> None:
    if "day" not in df.columns:
        return
    dfx = df.dropna(subset=["day"]).copy()
    if dfx.empty:
        return

    dfx["day"] = dfx["day"].astype(int)

    g = dfx.groupby("day", as_index=False).agg(
        n_graphs=("file", "count"),
        nodes_mean=("n_nodes", "mean"),
        edges_mean=("n_edges", "mean"),
        nodes_med=("n_nodes", "median"),
        edges_med=("n_edges", "median"),
    ).sort_values("day")

    g["nodes_delta_mean"] = g["nodes_mean"].diff()
    g["edges_delta_mean"] = g["edges_mean"].diff()

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(2, 1, hspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(g["day"], g["nodes_mean"], linewidth=2.2, label="nodes (mean)")
    ax1.plot(g["day"], g["nodes_med"], linewidth=2.2, linestyle="--", label="nodes (median)")
    ax1b = ax1.twinx()
    ax1b.plot(g["day"], g["edges_mean"], linewidth=2.2, label="edges (mean)")
    ax1b.plot(g["day"], g["edges_med"], linewidth=2.2, linestyle="--", label="edges (median)")

    ax1.set_title("Network size over time")
    ax1.set_xlabel("day")
    ax1.set_ylabel("# nodes")
    ax1b.set_ylabel("# edges")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axhline(0.0, linewidth=1.0)
    ax2.plot(g["day"], g["nodes_delta_mean"], linewidth=2.2, label="Δ nodes (mean)")
    ax2.plot(g["day"], g["edges_delta_mean"], linewidth=2.2, label="Δ edges (mean)")
    ax2.set_title("Net change proxy (importation vs discharge)")
    ax2.set_xlabel("day")
    ax2.set_ylabel("delta")
    ax2.legend(frameon=False)

    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_timeline_train_test_diff_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
    title: str
) -> None:
    if "day" not in train_df.columns or "day" not in test_df.columns:
        return

    A = train_df.dropna(subset=["day"]).copy()
    B = test_df.dropna(subset=["day"]).copy()
    if A.empty or B.empty:
        return

    A["day"] = A["day"].astype(int)
    B["day"] = B["day"].astype(int)

    ga = A.groupby("day", as_index=False).agg(nodes_mean=("n_nodes", "mean"), edges_mean=("n_edges", "mean"))
    gb = B.groupby("day", as_index=False).agg(nodes_mean=("n_nodes", "mean"), edges_mean=("n_edges", "mean"))

    M = pd.merge(ga, gb, on="day", how="inner", suffixes=("_train", "_test")).sort_values("day")
    if M.empty:
        return

    M["nodes_diff"] = M["nodes_mean_test"] - M["nodes_mean_train"]
    M["edges_diff"] = M["edges_mean_test"] - M["edges_mean_train"]

    fig = plt.figure(figsize=(14.0, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(M["day"], M["nodes_diff"], linewidth=2.2, label="nodes (test - train)")
    ax.plot(M["day"], M["edges_diff"], linewidth=2.2, label="edges (test - train)")
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel("difference")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------------- Sankey-style flow --------------------------

def _detect_node_attr_name(G: nx.Graph, candidates: Tuple[str, ...]) -> Optional[str]:
    for _, d in G.nodes(data=True):
        for cand in candidates:
            if cand in d:
                return cand
    return None


def _choose_flow_attr(G: nx.Graph, flow_attr: str) -> Tuple[str, str]:
    if flow_attr not in ("auto", "ward", "node_type", "state"):
        return ("ward", "")

    if flow_attr in ("auto", "ward"):
        name = _detect_node_attr_name(G, _WARD_CANDIDATES)
        if name is not None:
            return ("ward", name)
        if flow_attr == "ward":
            return ("ward", "")

    if flow_attr in ("auto", "node_type"):
        name = _detect_node_attr_name(G, _NODETYPE_CANDIDATES)
        if name is not None:
            return ("node_type", name)
        if flow_attr == "node_type":
            return ("node_type", "")

    if flow_attr in ("auto", "state"):
        name = _detect_node_attr_name(G, _STATE_CANDIDATES)
        if name is not None:
            return ("state", name)
        if flow_attr == "state":
            return ("state", "")

    return ("ward", "")


def _node_category(d: Dict[str, Any], attr_name: str) -> str:
    v = d.get(attr_name, None)
    if v is None:
        return "NA"
    s = str(v).strip()
    return s if s else "NA"


def aggregate_flow_matrix(
    files: List[Path],
    flow_attr: str,
    top_k: int,
    rng_seed: int
) -> Tuple[pd.DataFrame, str]:
    flow: Dict[Tuple[str, str], float] = {}
    used_attr_name: Optional[str] = None
    used_kind: Optional[str] = None

    for fp in files:
        try:
            G0 = nx.read_graphml(fp)
        except Exception:
            continue
        G = _to_simple_graph(G0)

        kind, attr_name = _choose_flow_attr(G, flow_attr=flow_attr)
        if attr_name == "":
            continue

        if used_attr_name is None:
            used_attr_name = attr_name
            used_kind = kind

        w_attr = _detect_edge_weight_attr(G)

        if G.is_directed():
            for u, v, ed in G.edges(data=True):
                su = _node_category(G.nodes[u], attr_name)
                tv = _node_category(G.nodes[v], attr_name)
                w = _safe_float(ed.get(w_attr)) if w_attr else None
                val = float(w) if (w is not None and math.isfinite(w)) else 1.0
                flow[(su, tv)] = flow.get((su, tv), 0.0) + val
        else:
            for u, v, ed in G.edges(data=True):
                a = _node_category(G.nodes[u], attr_name)
                b = _node_category(G.nodes[v], attr_name)
                w = _safe_float(ed.get(w_attr)) if w_attr else None
                val = float(w) if (w is not None and math.isfinite(w)) else 1.0
                flow[(a, b)] = flow.get((a, b), 0.0) + val
                flow[(b, a)] = flow.get((b, a), 0.0) + val

    if used_attr_name is None:
        return pd.DataFrame(), ""

    totals: Dict[str, float] = {}
    for (s, t), v in flow.items():
        totals[s] = totals.get(s, 0.0) + v
        totals[t] = totals.get(t, 0.0) + v

    cats_sorted = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    keep = [c for c, _ in cats_sorted[: max(2, top_k)]]
    keep_set = set(keep)

    def _map_cat(x: str) -> str:
        return x if x in keep_set else "Other"

    mapped: Dict[Tuple[str, str], float] = {}
    for (s, t), v in flow.items():
        ms = _map_cat(s)
        mt = _map_cat(t)
        mapped[(ms, mt)] = mapped.get((ms, mt), 0.0) + v

    cats = sorted(set([s for s, _ in mapped.keys()] + [t for _, t in mapped.keys()]))
    df = pd.DataFrame(0.0, index=cats, columns=cats)
    for (s, t), v in mapped.items():
        df.loc[s, t] += float(v)

    out_tot = df.sum(axis=1).to_dict()
    order = sorted(df.index.tolist(), key=lambda c: out_tot.get(c, 0.0), reverse=True)
    df = df.loc[order, order]

    _ = rng_seed
    return df, f"{used_kind}:{used_attr_name}"


def _bezier_band_path(x0: float, x1: float, y0a: float, y0b: float, y1a: float, y1b: float) -> MplPath:
    cx0 = x0 + (x1 - x0) * 0.35
    cx1 = x0 + (x1 - x0) * 0.65

    verts = [
        (x0, y0a),
        (cx0, y0a),
        (cx1, y1a),
        (x1, y1a),

        (x1, y1b),
        (cx1, y1b),
        (cx0, y0b),
        (x0, y0b),

        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,

        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,

        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def plot_sankey_from_matrix(
    flow_df: pd.DataFrame,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    title: str,
    max_links: int = 40
) -> None:
    if flow_df.empty:
        return

    cats = flow_df.index.tolist()

    links: List[Tuple[str, str, float]] = []
    for s in cats:
        for t in cats:
            v = float(flow_df.loc[s, t])
            if v > 0.0:
                links.append((s, t, v))
    links.sort(key=lambda x: x[2], reverse=True)
    links = links[: max(1, max_links)]

    out_tot: Dict[str, float] = {c: 0.0 for c in cats}
    in_tot: Dict[str, float] = {c: 0.0 for c in cats}
    for s, t, v in links:
        out_tot[s] += v
        in_tot[t] += v

    for c in cats:
        out_tot[c] = max(out_tot[c], 1e-12)
        in_tot[c] = max(in_tot[c], 1e-12)

    y_min, y_max = 0.04, 0.96
    avail = y_max - y_min

    n_nodes = len(cats)
    pad = 0.012
    if n_nodes > 1:
        pad = min(pad, 0.25 * avail / float(n_nodes - 1))
    total_pad = pad * max(n_nodes - 1, 0)
    avail_for_bars = max(avail - total_pad, 1e-6)

    total_out = float(sum(out_tot.values()))
    total_in = float(sum(in_tot.values()))
    total = max(total_out, total_in, 1e-12)

    out_scale = avail_for_bars / total
    in_scale = avail_for_bars / total

    y0: Dict[str, Tuple[float, float]] = {}
    y1: Dict[str, Tuple[float, float]] = {}

    y = y_min
    for c in cats:
        h = out_tot[c] * out_scale
        y0[c] = (y, y + h)
        y = y + h + pad

    y = y_min
    for c in cats:
        h = in_tot[c] * in_scale
        y1[c] = (y, y + h)
        y = y + h + pad

    y0_cursor: Dict[str, float] = {c: y0[c][0] for c in cats}
    y1_cursor: Dict[str, float] = {c: y1[c][0] for c in cats}

    colors = palette["colors"]

    fig = plt.figure(figsize=(14.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title)

    xL, xR = 0.08, 0.92
    bar_w = 0.018

    for i, c in enumerate(cats):
        ccol = colors[i % len(colors)]
        a0, b0 = y0[c]
        a1, b1 = y1[c]

        ax.add_patch(plt.Rectangle((xL - bar_w, a0), bar_w, max(1e-6, b0 - a0),
                                   facecolor=ccol, edgecolor="black", linewidth=0.3, alpha=0.95))
        ax.add_patch(plt.Rectangle((xR, a1), bar_w, max(1e-6, b1 - a1),
                                   facecolor=ccol, edgecolor="black", linewidth=0.3, alpha=0.95))

        ax.text(xL - bar_w - 0.01, (a0 + b0) / 2, c, ha="right", va="center", fontsize=9)
        ax.text(xR + bar_w + 0.01, (a1 + b1) / 2, c, ha="left", va="center", fontsize=9)

    for s, t, v in links:
        frac = v / total
        y0a = y0_cursor[s]
        y0b = y0a + frac
        y0_cursor[s] = y0b

        y1a = y1_cursor[t]
        y1b = y1a + frac
        y1_cursor[t] = y1b

        i = cats.index(s)
        link_col = colors[i % len(colors)]
        path = _bezier_band_path(xL, xR, y0a, y0b, y1a, y1b)
        patch = PathPatch(path, facecolor=link_col, edgecolor="black", linewidth=0.15, alpha=0.35)
        ax.add_patch(patch)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)



# -------------------------- State-composition timeline figures --------------------------

def _normalise_state_label(raw_value: Any) -> str:
    """
    Map raw state encodings to readable labels where possible.
    Supports common AMR encodings:
        0/U, 1/CS, 2/CR, 3/IS, 4/IR
    """
    if raw_value is None:
        return "NA"

    s = str(raw_value).strip()
    if s == "":
        return "NA"

    mapping = {
        "0": "U",
        "1": "CS",
        "2": "CR",
        "3": "IS",
        "4": "IR",
        "U": "U",
        "CS": "CS",
        "CR": "CR",
        "IS": "IS",
        "IR": "IR",
    }
    return mapping.get(s.upper(), s)


def make_state_percentages_figure(
    df: pd.DataFrame,
    palette: Dict[str, Any],
    out_png: Path,
    out_pdf: Path,
    title: str
) -> None:
    """
    Plot the percentage of each node-state category across days/graphs.
    If multiple graphs share the same parsed day, percentages are averaged.
    Falls back to graph index when no day is parsed from filenames.
    """
    state_cols = [c for c in df.columns if c.startswith("state_pct__")]
    if not state_cols:
        return

    dfx = df.copy()
    dfx = dfx[dfx[state_cols].notna().any(axis=1)].copy()
    if dfx.empty:
        return

    x_col = "day"
    if "day" not in dfx.columns or not dfx["day"].notna().any():
        dfx["graph_index"] = np.arange(1, len(dfx) + 1, dtype=int)
        x_col = "graph_index"
    else:
        dfx["day"] = dfx["day"].astype(int)

    grouped = dfx.groupby(x_col, as_index=False)[state_cols].mean().sort_values(x_col)
    if grouped.empty:
        return

    fig = plt.figure(figsize=(14.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)

    colors = palette["colors"]
    for idx, col in enumerate(state_cols):
        state_label = col.replace("state_pct__", "")
        ax.plot(
            grouped[x_col],
            grouped[col],
            linewidth=2.4,
            label=state_label,
            color=colors[idx % len(colors)],
        )

    ax.set_title(title)
    ax.set_xlabel("day" if x_col == "day" else "graph index")
    ax.set_ylabel("percentage of nodes")
    ax.set_ylim(0.0, 100.0)
    ax.legend(frameon=False, ncol=1)
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -------------------------- LaTeX snippet generation --------------------------

def _latex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
              .replace("_", "\\_")
              .replace("%", "\\%")
              .replace("&", "\\&")
              .replace("#", "\\#")
              .replace("{", "\\{")
              .replace("}", "\\}")
              .replace("$", "\\$")
              .replace("^", "\\^{}")
              .replace("~", "\\~{}"))


def caption_for_png(filename: str) -> Tuple[str, str]:
    """
    Returns (title, caption) for a known PNG filename pattern.
    """
    f = filename.lower()

    if f.startswith("figure_microgrid_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} microgrid", "Basic dataset-level graph statistics: node/edge counts, density, mean degree, pooled degree distribution, pooled edge-weight distribution (if present), and categorical node attribute summaries (if present).")
    if f.startswith("figure_distributions_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} structural distributions", "Additional structural descriptors across graphs: clustering, transitivity, assortativity, reciprocity (directed), largest-component size, and diameter/eccentricity proxy.")
    if f.startswith("figure_communities_and_centrality_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} communities and centrality", "Community structure and node-centrality statistics: number of communities (greedy modularity on undirected projection of largest component), modularity, largest-community fraction, pooled community sizes, and pooled centralities (betweenness/PageRank/eigenvector where computed).")
    if f.startswith("figure_flow_sankey_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} flow (Sankey)", "Aggregated directed flow between node categories (default: ward if available, else node type/state). Ribbon width represents summed edge weights when present, otherwise edge counts. Categories outside top-K are pooled as ``Other''.")
    if f == "figure_train_vs_test_shift.png":
        return ("Train vs test shift (histograms)", "Distribution shift assessment using overlaid histograms for key graph metrics, annotated with effect sizes (Cohen's $d$) and two-sample Kolmogorov--Smirnov statistics.")
    if f == "figure_train_vs_test_ecdf.png":
        return ("Train vs test shift (ECDF)", "Empirical CDF overlays for selected metrics highlighting distribution shift beyond histogram binning.")
    if f.startswith("figure_timeline_nodes_edges_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} timeline (nodes/edges)", "Temporal evolution of average/median graph size (nodes/edges) over the parsed day index, with day-to-day deltas as a net change proxy (importation vs discharge intensity).")
    if f.startswith("figure_state_percentages_"):
        label = filename.split("_")[-1].split(".")[0]
        return (f"{label} state composition timeline", "Per-day node-state composition shown as the percentage of nodes in each state (for example U, CS, CR, IS, IR). When multiple graphs share the same parsed day, percentages are averaged across graphs.")
    if f == "figure_timeline_diff_test_minus_train.png":
        return ("Timeline difference (test $-$ train)", "Per-day difference between test and train mean nodes/edges (test minus train) to highlight temporal distribution shift.")
    return ("Dataset figure", "Automatically generated dataset summary figure.")


def write_latex_txt(out_dir: Path, main_title: str) -> Path:
    """
    Scans out_dir for produced PNGs and writes latex.txt with figure environments.
    """
    pngs = sorted([p for p in out_dir.glob("*.png") if p.is_file()])
    latex_path = out_dir / "latex.txt"

    lines: List[str] = []
    lines.append("% Auto-generated by graph_folder_figures.py")
    lines.append("% Preamble requirements:")
    lines.append("% \\usepackage{graphicx}")
    lines.append("% Optional: \\usepackage{float} (if you want [H])")
    lines.append("")
    lines.append("% If your figures are in a subfolder, set e.g.:")
    lines.append("% \\graphicspath{{figs/}}")
    lines.append("")
    lines.append(f"% Figures for: {_latex_escape(main_title)}")
    lines.append("")

    for p in pngs:
        title, cap = caption_for_png(p.name)
        label = "fig:" + re.sub(r"[^a-z0-9]+", "-", p.stem.lower()).strip("-")
        lines.append("\\begin{figure*}[t]")
        lines.append("  \\centering")
        lines.append(f"  \\includegraphics[width=0.95\\textwidth]{{{_latex_escape(p.name)}}}")
        lines.append(f"  \\caption{{\\textbf{{{_latex_escape(title)}}}. {_latex_escape(cap)}}}")
        lines.append(f"  \\label{{{_latex_escape(label)}}}")
        lines.append("\\end{figure*}")
        lines.append("")

    latex_path.write_text("\n".join(lines), encoding="utf-8")
    return latex_path


class ProgressBar:
    def __init__(self, total: int, prefix: str = "GRAPHML") -> None:
        self.total = max(int(total), 1)
        self.current = 0
        self.prefix = prefix

    def show(self, note: str = "") -> None:
        self._render(note)

    def advance(self, note: str = "") -> None:
        self.current = min(self.current + 1, self.total)
        self._render(note)

    def _render(self, note: str = "") -> None:
        width = 32
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"{self.prefix} [{bar}] {self.current}/{self.total}"
        if note:
            msg += f" | {note}"
        print(msg, flush=True)


def _resolve_max_graphs_spec(spec: str, total: int, seed: int) -> List[int]:
    spec = (spec or "100%").strip().lower()
    if total <= 0:
        return []
    if spec in {"all", "100%"}:
        return list(range(total))
    if spec.endswith("%"):
        pct = float(spec[:-1].strip())
        if pct <= 0 or pct > 100:
            raise ValueError("--max_graphs percentage must be in (0, 100].")
        k = max(1, int(math.ceil(total * pct / 100.0)))
    else:
        k = int(spec)
        if k <= 0:
            raise ValueError("--max_graphs count must be positive, or use all/100%.")
        k = min(k, total)
    rng = random.Random(seed)
    idx = list(range(total))
    if k < total:
        idx = rng.sample(idx, k)
    idx.sort()
    return idx


def _apply_max_graphs(files: List[Path], spec: str, seed: int) -> List[Path]:
    idx = _resolve_max_graphs_spec(spec, len(files), seed)
    return [files[i] for i in idx]


def _resolve_worker_count(workers: int, n_files: int) -> int:
    if n_files <= 1:
        return 1
    if workers <= 0:
        workers = max(1, os.cpu_count() or 1)
    return max(1, min(int(workers), n_files))


def _process_graph_file(
    fp_str: str,
    label: str,
    bc_exact_max_nodes: int,
    bc_sample_k: int,
    comm_max_nodes: int,
    rng_seed: int,
) -> Dict[str, Any]:
    fp = Path(fp_str)
    day_idx = parse_day_from_filename(fp)
    try:
        G = nx.read_graphml(fp)
    except Exception as e:
        return {"row": {"file": str(fp), "read_error": str(e), "day": day_idx, "set": label}, "st": None}

    st = compute_graph_stats(
        G,
        bc_exact_max_nodes=bc_exact_max_nodes,
        bc_sample_k=bc_sample_k,
        comm_max_nodes=comm_max_nodes,
        rng_seed=rng_seed,
    )
    row = {
        "file": str(fp),
        "set": label,
        "day": day_idx,
        "n_nodes": st["n_nodes"],
        "n_edges": st["n_edges"],
        "directed": st["directed"],
        "density": st["density"],
        "avg_degree": st["avg_degree"],
        "avg_in_degree": st["avg_in_degree"],
        "avg_out_degree": st["avg_out_degree"],
        "reciprocity": st["reciprocity"],
        "avg_clustering": st["avg_clustering"],
        "transitivity": st["transitivity"],
        "assortativity": st["assortativity"],
        "gc_nodes": st["gc_nodes"],
        "gc_edges": st["gc_edges"],
        "diameter_or_p95_ecc": st["diameter_or_p95_ecc"],
        "avg_shortest_path": st["avg_shortest_path"],
        "deg_max": st["deg_max"],
        "w_mean": st["w_mean"],
        "w_std": st["w_std"],
        "w_p95": st["w_p95"],
        "bc_mean": st["bc_mean"],
        "bc_p95": st["bc_p95"],
        "bc_max": st["bc_max"],
        "pr_mean": st["pr_mean"],
        "pr_p95": st["pr_p95"],
        "pr_max": st["pr_max"],
        "ev_mean": st["ev_mean"],
        "ev_p95": st["ev_p95"],
        "ev_max": st["ev_max"],
        "n_communities": st["n_communities"],
        "modularity": st["modularity"],
        "largest_comm_frac": st["largest_comm_frac"],
    }
    state_counts = st.get("state_counts") or {}
    total_states = int(sum(state_counts.values()))
    if total_states > 0:
        for state_name, state_count in state_counts.items():
            norm_state = _normalise_state_label(state_name)
            row[f"state_pct__{norm_state}"] = 100.0 * float(state_count) / float(total_states)
    return {"row": row, "st": st}

# -------------------------- Folder runner --------------------------

def run_folder(
    graph_dir: Path,
    args: argparse.Namespace,
    out_dir: Path,
    palette: Dict[str, Any],
    label: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    all_files = find_graphml_files(graph_dir)
    files = _apply_max_graphs(all_files, args.max_graphs, args.seed)

    if not files:
        raise SystemExit(f"No .graphml files found under: {graph_dir}")

    progress = ProgressBar(total=len(files), prefix=f"GRAPHML {label}")
    progress.show(f"start | dir={graph_dir.name} | sampled={len(files)}/{len(all_files)}")

    pooled: Dict[str, Any] = {
        "deg_all": [],
        "w_all": [],
        "comm_sizes_all": [],
        "bc_all": [],
        "pr_all": [],
        "ev_all": [],
        "node_type_counts_all": {},
        "state_counts_all": {},
        "ward_counts_all": {},
        "n_communities_all": [],
    }

    worker_count = _resolve_worker_count(args.workers, len(files))
    print(f"GRAPHML {label} | workers={worker_count} | sampled={len(files)}/{len(all_files)}", flush=True)

    rows_by_idx: Dict[int, Dict[str, Any]] = {}
    stats_by_idx: Dict[int, Optional[Dict[str, Any]]] = {}
    futures = {}

    if worker_count == 1:
        for idx, fp in enumerate(files):
            result = _process_graph_file(
                str(fp), label, args.bc_exact_max_nodes, args.bc_sample_k, args.comm_max_nodes, args.seed
            )
            rows_by_idx[idx] = result["row"]
            stats_by_idx[idx] = result["st"]
            progress.advance(Path(fp).name)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            for idx, fp in enumerate(files):
                fut = ex.submit(
                    _process_graph_file,
                    str(fp),
                    label,
                    args.bc_exact_max_nodes,
                    args.bc_sample_k,
                    args.comm_max_nodes,
                    args.seed,
                )
                futures[fut] = (idx, fp)
            for fut in as_completed(futures):
                idx, fp = futures[fut]
                result = fut.result()
                rows_by_idx[idx] = result["row"]
                stats_by_idx[idx] = result["st"]
                progress.advance(Path(fp).name)

    rows: List[Dict[str, Any]] = []
    for idx in range(len(files)):
        row = rows_by_idx[idx]
        st = stats_by_idx[idx]
        rows.append(row)
        if st is None:
            continue

        pooled["n_communities_all"].append(float(st["n_communities"]))

        if isinstance(st.get("deg_arr"), np.ndarray) and st["deg_arr"].size > 0:
            pooled["deg_all"].extend([float(x) for x in st["deg_arr"] if math.isfinite(float(x))])
        if isinstance(st.get("w_arr"), np.ndarray) and st["w_arr"].size > 0:
            pooled["w_all"].extend([float(x) for x in st["w_arr"] if math.isfinite(float(x))])
        if isinstance(st.get("comm_sizes"), np.ndarray) and st["comm_sizes"].size > 0:
            pooled["comm_sizes_all"].extend([float(x) for x in st["comm_sizes"] if math.isfinite(float(x))])
        if isinstance(st.get("bc_arr"), np.ndarray) and st["bc_arr"].size > 0:
            pooled["bc_all"].extend([float(x) for x in st["bc_arr"] if math.isfinite(float(x))])
        if isinstance(st.get("pr_arr"), np.ndarray) and st["pr_arr"].size > 0:
            pooled["pr_all"].extend([float(x) for x in st["pr_arr"] if math.isfinite(float(x))])
        if isinstance(st.get("ev_arr"), np.ndarray) and st["ev_arr"].size > 0:
            pooled["ev_all"].extend([float(x) for x in st["ev_arr"] if math.isfinite(float(x))])

        for key_src, key_tgt in [
            ("node_type_counts", "node_type_counts_all"),
            ("state_counts", "state_counts_all"),
            ("ward_counts", "ward_counts_all"),
        ]:
            d_src = st.get(key_src) or {}
            d_tgt = pooled[key_tgt]
            for k, v in d_src.items():
                d_tgt[k] = d_tgt.get(k, 0) + int(v)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"graph_summary_{label}.csv", index=False)

    ok_df = df[df.get("read_error").isna()] if "read_error" in df.columns else df

    deg_all_arr = np.asarray(pooled["deg_all"], dtype=float) if pooled["deg_all"] else np.array([])
    w_all_arr = np.asarray(pooled["w_all"], dtype=float) if pooled["w_all"] else np.array([])
    comm_sizes_all_arr = np.asarray(pooled["comm_sizes_all"], dtype=float) if pooled["comm_sizes_all"] else np.array([])
    bc_all_arr = np.asarray(pooled["bc_all"], dtype=float) if pooled["bc_all"] else np.array([])
    pr_all_arr = np.asarray(pooled["pr_all"], dtype=float) if pooled["pr_all"] else np.array([])
    ev_all_arr = np.asarray(pooled["ev_all"], dtype=float) if pooled["ev_all"] else np.array([])

    print(f"GRAPHML {label} | plotting microgrid", flush=True)
    make_microgrid_figure(
        df=ok_df, deg_all=deg_all_arr, w_all=w_all_arr,
        node_type_counts_all=pooled["node_type_counts_all"],
        state_counts_all=pooled["state_counts_all"],
        palette=palette,
        out_png=out_dir / f"figure_microgrid_{label}.png",
        out_pdf=out_dir / f"figure_microgrid_{label}.pdf",
        suptitle=f"{args.title} ({label})",
    )

    print(f"GRAPHML {label} | plotting distributions", flush=True)
    make_distributions_figure(
        df=ok_df, palette=palette,
        out_png=out_dir / f"figure_distributions_{label}.png",
        out_pdf=out_dir / f"figure_distributions_{label}.pdf",
        suptitle=f"{args.title} ({label})",
    )

    print(f"GRAPHML {label} | plotting communities_and_centrality", flush=True)
    make_communities_and_centrality_figure(
        df=ok_df, comm_sizes_all=comm_sizes_all_arr, bc_all=bc_all_arr, pr_all=pr_all_arr, ev_all=ev_all_arr,
        palette=palette,
        out_png=out_dir / f"figure_communities_and_centrality_{label}.png",
        out_pdf=out_dir / f"figure_communities_and_centrality_{label}.pdf",
        suptitle=f"{args.title} ({label})",
    )

    make_timeline_nodes_edges_figure(
        df=ok_df,
        out_png=out_dir / f"figure_timeline_nodes_edges_{label}.png",
        out_pdf=out_dir / f"figure_timeline_nodes_edges_{label}.pdf",
        title=f"{args.title} ({label}) timeline: nodes/edges",
    )

    make_state_percentages_figure(
        df=ok_df, palette=palette,
        out_png=out_dir / f"figure_state_percentages_{label}.png",
        out_pdf=out_dir / f"figure_state_percentages_{label}.pdf",
        title=f"{args.title} ({label}) state composition over time",
    )

    print(f"GRAPHML {label} | aggregating flow matrix", flush=True)
    flow_df, used_attr = aggregate_flow_matrix(
        files=files,
        flow_attr=args.flow_attr,
        top_k=args.flow_top_k,
        rng_seed=args.seed,
    )
    if not flow_df.empty:
        flow_df.to_csv(out_dir / f"flow_matrix_{label}.csv")
        plot_sankey_from_matrix(
            flow_df=flow_df, palette=palette,
            out_png=out_dir / f"figure_flow_sankey_{label}.png",
            out_pdf=out_dir / f"figure_flow_sankey_{label}.pdf",
            title=f"{args.title} ({label}) flow sankey\nnode_attr={used_attr}  |  links(top)={args.flow_max_links}",
            max_links=args.flow_max_links,
        )

    comms = ok_df["n_communities"].dropna().to_numpy(dtype=float) if "n_communities" in ok_df.columns else np.array([])
    if comms.size > 0:
        print(
            f"DT_COMMUNITY_SUMMARY {label} n_graphs={len(ok_df)} communities_median={np.median(comms):.3g} "
            f"mean={np.mean(comms):.3g} min={np.min(comms):.3g} max={np.max(comms):.3g}"
        )
    else:
        print(f"DT_COMMUNITY_SUMMARY {label} n_graphs={len(ok_df)} communities=NA")

    if "day" in ok_df.columns and ok_df["day"].notna().any():
        print(f"DT_TIMELINE_OK {label} days_parsed={int(ok_df['day'].notna().sum())}/{len(ok_df)}")
    else:
        print(f"DT_TIMELINE_OK {label} days_parsed=0/{len(ok_df)} (no filename match)")

    return df, pooled


# -------------------------- Main --------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", type=str, required=True, help="Primary folder containing .graphml files (recursively).")
    parser.add_argument("--compare_dir", type=str, default="", help="Optional second folder for train-vs-test comparison.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--identity", type=str, default="Harry", help='Palette identity string, e.g. "Harry Triantafyllidis".')
    parser.add_argument("--title", type=str, default="Graph dataset summary", help="Figure title.")
    parser.add_argument("--label", type=str, default="train", help="Label for --graph_dir when --compare_dir is set.")
    parser.add_argument("--compare_label", type=str, default="test", help="Label for --compare_dir when set.")
    parser.add_argument("--max_graphs", type=str, default="100%", help="Graph sampling specification: integer count, percentage like 5%, or all/100%.")

    parser.add_argument("--bc_exact_max_nodes", type=int, default=2500, help="Betweenness exact computation threshold.")
    parser.add_argument("--bc_sample_k", type=int, default=300, help="Betweenness sample size for large graphs.")
    parser.add_argument("--comm_max_nodes", type=int, default=6000, help="Community detection node cap (subsample above).")
    parser.add_argument("--seed", type=int, default=123, help="Deterministic seed for sampling.")

    parser.add_argument("--flow_attr", type=str, default="auto", choices=["auto", "ward", "node_type", "state"],
                        help="Node attribute used to define categories for flow Sankey.")
    parser.add_argument("--flow_top_k", type=int, default=10, help="Keep top-K categories by volume (others -> Other).")
    parser.add_argument("--flow_max_links", type=int, default=40, help="Max Sankey links to draw (top by volume).")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes for per-file GraphML summaries; 0=auto, 1=serial.")

    args = parser.parse_args()

    graph_dir = Path(args.graph_dir).expanduser().resolve()
    compare_dir = Path(args.compare_dir).expanduser().resolve() if args.compare_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_matplotlib()
    palette = make_identity_palette(args.identity, n=10)

    df_a, _ = run_folder(graph_dir, args, out_dir, palette, label=args.label)

    if compare_dir is not None:
        df_b, _ = run_folder(compare_dir, args, out_dir, palette, label=args.compare_label)

        train_df = df_a[df_a.get("read_error").isna()] if "read_error" in df_a.columns else df_a
        test_df = df_b[df_b.get("read_error").isna()] if "read_error" in df_b.columns else df_b

        print("GRAPHML compare | plotting train_vs_test_shift", flush=True)
        shift_df = make_train_vs_test_shift_figure(
            train_df=train_df,
            test_df=test_df,
            palette=palette,
            out_png=out_dir / "figure_train_vs_test_shift.png",
            out_pdf=out_dir / "figure_train_vs_test_shift.pdf",
            title=f"{args.title}: {args.label} vs {args.compare_label} (distribution shift)",
            label_train=args.label,
            label_test=args.compare_label,
        )
        shift_df.to_csv(out_dir / "train_vs_test_shift_stats.csv", index=False)

        print("GRAPHML compare | plotting train_vs_test_ecdf", flush=True)
        make_train_vs_test_ecdf_figure(
            train_df=train_df,
            test_df=test_df,
            palette=palette,
            out_png=out_dir / "figure_train_vs_test_ecdf.png",
            out_pdf=out_dir / "figure_train_vs_test_ecdf.pdf",
            title=f"{args.title}: {args.label} vs {args.compare_label} (ECDF)",
            label_train=args.label,
            label_test=args.compare_label,
        )

        print("GRAPHML compare | plotting timeline_diff_test_minus_train", flush=True)
        make_timeline_train_test_diff_figure(
            train_df=train_df,
            test_df=test_df,
            out_png=out_dir / "figure_timeline_diff_test_minus_train.png",
            out_pdf=out_dir / "figure_timeline_diff_test_minus_train.pdf",
            title=f"{args.title}: timeline difference (test - train)",
        )

        combined = pd.concat([train_df, test_df], ignore_index=True)
        combined.to_csv(out_dir / "graph_summary_combined.csv", index=False)

        print("DT_SHIFT_FILES shift=figure_train_vs_test_shift.png ecdf=figure_train_vs_test_ecdf.png stats=train_vs_test_shift_stats.csv")
        print("DT_TIMELINE_FILES "
              "timeline_train=figure_timeline_nodes_edges_train.png "
              "timeline_test=figure_timeline_nodes_edges_test.png "
              "timeline_diff=figure_timeline_diff_test_minus_train.png "
              "state_train=figure_state_percentages_train.png "
              "state_test=figure_state_percentages_test.png")

    latex_path = write_latex_txt(out_dir=out_dir, main_title=args.title)

    print(f"DT_FIG_META out_dir={out_dir} palette_seed={palette['seed_int']}")
    print(f"DT_LATEX_FILE latex={latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
