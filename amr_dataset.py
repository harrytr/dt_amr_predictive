#!/usr/bin/env python3
"""
amr_dataset.py
==============

Compatibility wrapper around TemporalGraphDataset for AMR-only training.

Kept minimal: the platform only supports AMR, so this is a convenience import layer.
"""

from temporal_graph_dataset import TemporalGraphDataset, collate_temporal_graph_batch


class AMRDataset(TemporalGraphDataset):
    def __init__(self, folder: str, T: int, sliding_step: int = 1, file_ext: str = ".pt", cache_all: bool = False):
        super().__init__(folder=folder, T=T, sliding_step=sliding_step, file_ext=file_ext, cache_all=cache_all)


def collate_amr_batch(batch):
    return collate_temporal_graph_batch(batch)
