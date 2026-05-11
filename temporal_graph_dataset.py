#!/usr/bin/env python3
"""
temporal_graph_dataset.py
========================

AMR-only temporal dataset:

- Reads a folder of daily .pt graphs
- Produces sliding windows of length T

Trajectory grouping (UPDATED):
- Prefer Data.sim_id + Data.day (robust to mixing pooled folders).
- Fallback to filename parsing: <sim_prefix>_t<day>[ _L<label> ].pt

Enhancement (stable node identity support):
- Builds (or loads) a per-folder node vocabulary from Data.node_names
- Attaches Data.node_id (LongTensor) aligned to Data.x row order

Returns:
  graphs      : list[T] of PyG Data
  labels_dict : {}  (kept for interface compatibility; tasks read labels from graphs)
"""

import json
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def natural_key(s: str):
    parts = re.split(r"(\d+)", s)
    out: List[object] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def _parse_sim_day_label(fname: str, file_ext: str):
    """
    Parse filenames of the form:
        <sim_prefix>_t<day>[ _L<label> ]<file_ext>

    Returns:
        (sim_prefix: str, day: int, label: Optional[int])
    or None if the filename does not match.
    """
    fe = str(file_ext)
    pat = rf"^(?P<prefix>.+?)_t(?P<t>\d+)(?:_L(?P<label>\d+))?{re.escape(fe)}$"
    m = re.match(pat, str(fname))
    if not m:
        return None
    prefix = str(m.group("prefix"))
    t = int(m.group("t"))
    lab = m.group("label")
    label = int(lab) if lab is not None else None
    return prefix, t, label


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return int(x.item())
        except Exception:
            return None
    return None


def _read_pt_metadata(path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Read (sim_id, day) from a .pt Data object if present.
    Returns (None, None) if not present or not parseable.

    Note: This loads the object.
    """
    try:
        data = torch.load(path, weights_only=False)
    except Exception:
        return None, None

    sim_id = getattr(data, "sim_id", None)
    day = getattr(data, "day", None)

    if sim_id is None or day is None:
        return None, None

    sim_id_s = str(sim_id)
    day_i = _safe_int(day)
    if sim_id_s.strip() == "" or day_i is None:
        return None, None

    return sim_id_s, int(day_i)


class TemporalGraphDataset(Dataset):
    """Temporal windows dataset for AMR daily graphs."""

    def __init__(
        self,
        folder: str,
        T: int,
        sliding_step: int = 1,
        file_ext: str = ".pt",
        cache_all: bool = False,
        build_node_vocab: bool = True,
        node_vocab_filename: str = "node_vocab.json",
        prefer_pt_metadata: bool = True,
        require_pt_metadata: bool = True,
        fail_on_noncontiguous: bool = True,
    ):
        self.folder = os.path.abspath(folder)
        self.T = int(T)
        self.sliding_step = int(sliding_step)
        self.file_ext = str(file_ext)

        self.cache_all = bool(cache_all)
        self._cache_max = max(1, 2 * self.T)
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._disk_paths: Dict[str, str] = {}

        self.groups: List[List[str]] = []

        # New controls
        self.prefer_pt_metadata = bool(prefer_pt_metadata)
        self.require_pt_metadata = bool(require_pt_metadata)
        self.fail_on_noncontiguous = bool(fail_on_noncontiguous)

        # Node vocabulary for stable node identities within this folder
        self.build_node_vocab = bool(build_node_vocab)
        self.node_vocab_filename = str(node_vocab_filename)
        self.node_vocab_path = os.path.join(self.folder, self.node_vocab_filename)
        self.node_vocab: Dict[str, int] = {}
        self.node_vocab_inv: List[str] = []

        self._scan_folder()

        if self.build_node_vocab:
            self._init_node_vocab()

    # ---------------------------------------------------------------------
    # Folder scanning
    # ---------------------------------------------------------------------
    def _scan_folder(self):
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        files = [f for f in os.listdir(self.folder) if f.endswith(self.file_ext)]
        files.sort(key=natural_key)

        for f in files:
            self._disk_paths[f] = os.path.join(self.folder, f)

        # Decide grouping mode:
        # - If prefer_pt_metadata: attempt metadata grouping; if none found, fallback to filename.
        # - If require_pt_metadata: hard fail unless ALL .pt have sim_id and day.
        use_metadata = False

        if self.prefer_pt_metadata or self.require_pt_metadata:
            any_meta = False
            all_meta = True
            meta_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

            for f in files:
                sim_id, day = _read_pt_metadata(self._disk_paths[f])
                meta_cache[f] = (sim_id, day)
                if sim_id is not None and day is not None:
                    any_meta = True
                else:
                    all_meta = False

            if self.require_pt_metadata and not all_meta:
                missing = [f for f, (sid, d) in meta_cache.items() if sid is None or d is None]
                ex = missing[0] if missing else "unknown"
                raise RuntimeError(
                    f"PT metadata required but missing for {len(missing)}/{len(files)} files "
                    f"(example: {ex}). Ensure convert_to_pt.py writes Data.sim_id and Data.day."
                )

            use_metadata = bool(any_meta) if self.prefer_pt_metadata else False
        else:
            meta_cache = {}

        if use_metadata:
            # Group by Data.sim_id and enforce day contiguity via Data.day
            sim_groups: Dict[str, List[Tuple[int, str]]] = {}  # sim_id -> list[(day, filename)]
            skipped_no_meta = 0
            for f in files:
                sim_id, day = meta_cache.get(f, (None, None))
                if sim_id is None or day is None:
                    skipped_no_meta += 1
                    continue
                sim_groups.setdefault(str(sim_id), []).append((int(day), f))

            n_nonempty = 0
            n_windows = 0
            n_skipped_windows = 0

            for sim_id, tf in sim_groups.items():
                if not tf:
                    continue
                n_nonempty += 1
                tf.sort(key=lambda z: int(z[0]))

                ds = [int(d) for d, _ in tf]
                fs = [f for _, f in tf]

                for start in range(0, len(fs) - self.T + 1, self.sliding_step):
                    d0 = ds[start]
                    ok = True
                    for k in range(1, self.T):
                        if ds[start + k] != d0 + k:
                            ok = False
                            break
                    if not ok:
                        n_skipped_windows += 1
                        if self.fail_on_noncontiguous:
                            raise RuntimeError(
                                f"Non-contiguous window detected (metadata mode) in '{self.folder}' "
                                f"sim_id='{sim_id}' start_day={d0} expected={list(range(d0, d0 + self.T))} "
                                f"got={ds[start:start + self.T]}"
                            )
                        continue
                    self.groups.append(fs[start: start + self.T])
                    n_windows += 1

            if skipped_no_meta > 0:
                print(
                    f"⚠️ Skipped {skipped_no_meta} files missing Data.sim_id/Data.day in '{self.folder}'",
                    flush=True,
                )
            if n_skipped_windows > 0 and not self.fail_on_noncontiguous:
                print(
                    f"⚠️ Skipped {n_skipped_windows} non-contiguous windows (missing days) in '{self.folder}'",
                    flush=True,
                )

            print(
                f"🧩 AMR dataset (metadata): {len(self.groups)} windows from {n_nonempty} trajectories in '{self.folder}'",
                flush=True,
            )
            return

        # ---------------- Fallback: filename grouping ----------------
        prefix_groups: Dict[str, List[Tuple[int, str]]] = {}  # prefix -> list[(t, filename)]
        skipped = 0
        for f in files:
            parsed = _parse_sim_day_label(f, self.file_ext)
            if parsed is None:
                skipped += 1
                continue
            prefix, t, _ = parsed
            prefix_groups.setdefault(prefix, []).append((int(t), f))

        n_nonempty = 0
        n_windows = 0
        n_skipped_windows = 0
        for prefix, tf in prefix_groups.items():
            if not tf:
                continue
            n_nonempty += 1
            tf.sort(key=lambda z: int(z[0]))

            ts = [int(t) for t, _ in tf]
            fs = [f for _, f in tf]

            for start in range(0, len(fs) - self.T + 1, self.sliding_step):
                t0 = ts[start]
                ok = True
                for k in range(1, self.T):
                    if ts[start + k] != t0 + k:
                        ok = False
                        break
                if not ok:
                    n_skipped_windows += 1
                    if self.fail_on_noncontiguous:
                        raise RuntimeError(
                            f"Non-contiguous window detected (filename mode) in '{self.folder}' "
                            f"prefix='{prefix}' start_t={t0} expected={list(range(t0, t0 + self.T))} "
                            f"got={ts[start:start + self.T]}"
                        )
                    continue
                self.groups.append(fs[start: start + self.T])
                n_windows += 1

        if skipped > 0:
            print(
                f"⚠️ Skipped {skipped} files that do not match '*_t<day>[ _L<label> ]{self.file_ext}'",
                flush=True,
            )
        if n_skipped_windows > 0 and not self.fail_on_noncontiguous:
            print(
                f"⚠️ Skipped {n_skipped_windows} non-contiguous windows (missing days) in '{self.folder}'",
                flush=True,
            )

        print(
            f"🧩 AMR dataset (filename): {len(self.groups)} windows from {n_nonempty} simulations in '{self.folder}'",
            flush=True,
        )

    # ---------------------------------------------------------------------
    # Node vocabulary
    # ---------------------------------------------------------------------
    def _load_vocab_from_disk(self) -> bool:
        if not os.path.isfile(self.node_vocab_path):
            return False
        try:
            with open(self.node_vocab_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            vocab = obj.get("node_vocab", None)
            inv = obj.get("node_vocab_inv", None)
            if not isinstance(vocab, dict) or not isinstance(inv, list):
                return False

            self.node_vocab = {str(k): int(v) for k, v in vocab.items()}
            self.node_vocab_inv = [str(x) for x in inv]

            if len(self.node_vocab_inv) == 0 or len(self.node_vocab) == 0:
                return False

            return True
        except Exception:
            return False

    def _save_vocab_to_disk(self) -> None:
        try:
            obj = {
                "node_vocab": self.node_vocab,
                "node_vocab_inv": self.node_vocab_inv,
            }
            with open(self.node_vocab_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _init_node_vocab(self) -> None:
        """Load existing node vocab if present; else build from all .pt files."""
        if self._load_vocab_from_disk():
            print(
                f"✅ Loaded node vocabulary from '{self.node_vocab_path}' ({len(self.node_vocab)} nodes).",
                flush=True,
            )
            return

        vocab: Dict[str, int] = {}
        inv: List[str] = []

        fnames = sorted(self._disk_paths.keys(), key=natural_key)

        for fname in fnames:
            path = self._disk_paths[fname]
            try:
                data = torch.load(path, weights_only=False)
            except Exception:
                continue

            node_names = getattr(data, "node_names", None)
            if node_names is None:
                try:
                    n = int(getattr(data, "num_nodes", 0))
                except Exception:
                    n = 0
                node_names = [str(i) for i in range(n)]

            for name in node_names:
                key = str(name)
                if key not in vocab:
                    vocab[key] = len(inv)
                    inv.append(key)

        self.node_vocab = vocab
        self.node_vocab_inv = inv

        if len(self.node_vocab) == 0:
            print(
                "⚠️ Node vocabulary build produced 0 entries; node_id will fall back to positional indices.",
                flush=True,
            )
        else:
            print(
                f"✅ Built node vocabulary for folder '{self.folder}' ({len(self.node_vocab)} nodes).",
                flush=True,
            )

        self._save_vocab_to_disk()

    def _attach_node_ids(self, data) -> None:
        """Attach data.node_id aligned to data.x rows, using per-folder vocabulary."""
        node_names = getattr(data, "node_names", None)
        if node_names is None:
            try:
                n = int(getattr(data, "num_nodes", 0))
            except Exception:
                n = 0
            node_names = [str(i) for i in range(n)]
            data.node_names = node_names

        if not self.node_vocab:
            data.node_id = torch.arange(len(node_names), dtype=torch.long)
            return

        ids: List[int] = []
        vocab_changed = False
        for name in node_names:
            key = str(name)
            if key not in self.node_vocab:
                self.node_vocab[key] = len(self.node_vocab_inv)
                vocab_changed = True
                self.node_vocab_inv.append(key)
            ids.append(self.node_vocab[key])

        data.node_id = torch.tensor(ids, dtype=torch.long)

        if vocab_changed:
            self._save_vocab_to_disk()

    # ---------------------------------------------------------------------
    # Dataset protocol
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.groups)

    def _load_graph(self, fname: str):
        if fname in self._cache:
            self._cache.move_to_end(fname)
            return self._cache[fname]

        if fname not in self._disk_paths:
            raise KeyError(f"Missing file in index: {fname}")

        data = torch.load(self._disk_paths[fname], weights_only=False)

        if self.build_node_vocab:
            self._attach_node_ids(data)

        self._cache[fname] = data

        if not self.cache_all and len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return data

    def __getitem__(self, idx: int):
        fnames = self.groups[idx]
        graphs = [self._load_graph(f) for f in fnames]
        labels_dict: Dict[str, torch.Tensor] = {}
        return graphs, labels_dict


def collate_temporal_graph_batch(batch):
    graphs_list, labels_list = zip(*batch)
    T = len(graphs_list[0])

    from torch_geometric.data import Batch as PyGBatch

    graphs_per_t: List[List[object]] = [[] for _ in range(T)]
    for graphs in graphs_list:
        for t, g in enumerate(graphs):
            graphs_per_t[t].append(g)

    batched_graphs = [PyGBatch.from_data_list(graphs_per_t[t]) for t in range(T)]

    merged: Dict[str, torch.Tensor] = {}
    if labels_list and labels_list[0]:
        common_keys = set(labels_list[0].keys())
        for d in labels_list[1:]:
            common_keys &= set(d.keys())
        for k in common_keys:
            merged[k] = torch.stack([d[k] for d in labels_list], dim=0)

    return batched_graphs, merged
