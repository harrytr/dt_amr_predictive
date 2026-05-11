#!/usr/bin/env python3
"""
models_amr.py
=============

AMR-only temporal model: Edge-aware GraphSAGE encoder + Transformer over time.

Input:
  graphs: list[T] of PyG Batch objects

Output:
  [B, n_outputs]

This model expects:
  - g.x exists with fixed feature width (from convert_to_pt.py)
  - g.edge_attr optionally exists (weight + scaled edge type)
"""

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch

try:
    from torch_geometric.nn import MessagePassing
except Exception:  # pragma: no cover
    from torch_geometric.nn.conv import MessagePassing


class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 0, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)

        msg_in = self.in_channels + (self.edge_dim if self.edge_dim > 0 else 0)
        self.lin_neigh = nn.Linear(msg_in, self.out_channels, bias=True)
        self.lin_root = nn.Linear(self.in_channels, self.out_channels, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.lin_root(x)
        return out

    def message(self, x_j, edge_attr):
        if self.edge_dim > 0 and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if edge_attr.size(-1) != self.edge_dim:
                edge_attr = edge_attr[:, : self.edge_dim]
            msg = torch.cat([x_j, edge_attr], dim=-1)
        else:
            msg = x_j
        return self.lin_neigh(msg)


class NodeAttentionPool(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.score = nn.Linear(hidden_channels, 1)

    def forward(self, x, batch, return_attention: bool = False):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        out = torch.zeros_like(x)
        attn = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)

        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            w = F.softmax(s[idx], dim=0)
            attn[idx] = w
            out[idx] = x[idx] * w.unsqueeze(-1)

        pooled = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch, out)
        if return_attention:
            return pooled, attn
        return pooled


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        dropout: float,
        n_layers: int = 2,
    ):
        super().__init__()
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(EdgeSAGEConv(in_channels, hidden_channels, edge_dim=edge_dim))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(n_layers - 1):
            self.convs.append(EdgeSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.pool = NodeAttentionPool(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch, return_attention: bool = False):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch, return_attention=return_attention)


class AMRDyGFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        heads: int,
        T: int,
        dropout: float,
        use_cls_token: bool,
        n_outputs: int,
        n_layers: int = 2,
        sage_layers: int = 2,
        use_softplus: bool = True,
        output_activation: Optional[str] = None,
    ):
        super().__init__()

        self.T = int(T)
        self.hidden_channels = int(hidden_channels)
        self.use_cls_token = bool(use_cls_token)
        self.use_softplus = bool(use_softplus)
        # output_activation overrides use_softplus when provided
        if output_activation is None:
            self.output_activation = "softplus" if self.use_softplus else "identity"
        else:
            self.output_activation = str(output_activation).lower().strip()

        self.gnn = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            dropout=dropout,
            n_layers=sage_layers,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.T + (1 if self.use_cls_token else 0), hidden_channels))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))
            self.head = nn.Linear(hidden_channels, n_outputs)
        else:
            self.head = nn.Linear(hidden_channels, n_outputs)

    def encode_batched_graph(self, g: Batch, return_attention: bool = False):
        x = g.x
        edge_index = g.edge_index
        edge_attr = getattr(g, "edge_attr", None)
        batch = g.batch
        return self.gnn(x, edge_index, edge_attr, batch, return_attention=return_attention)

    def encode_day_graphs(self, graphs: List[Batch], return_attention: bool = False):
        day_embs = []
        day_attn = []
        for g in graphs:
            if return_attention:
                pooled, attn = self.encode_batched_graph(g, return_attention=True)
                day_embs.append(pooled)
                day_attn.append(attn)
            else:
                pooled = self.encode_batched_graph(g, return_attention=False)
                day_embs.append(pooled)

        H = torch.stack(day_embs, dim=1)
        if return_attention:
            return H, day_attn
        return H

    def encode_day_graph(self, g: Batch, return_attention: bool = False):
        x = g.x
        edge_index = g.edge_index
        edge_attr = getattr(g, "edge_attr", None)
        batch = g.batch
        return self.gnn(x, edge_index, edge_attr, batch, return_attention=return_attention)

    def encode_day_graphs(self, graphs: List[Batch], return_attention: bool = False):
        day_embs = []
        day_attn = []
        for g in graphs:
            if return_attention:
                pooled, attn = self.encode_day_graph(g, return_attention=True)
                day_embs.append(pooled)
                day_attn.append(attn)
            else:
                day_embs.append(self.encode_day_graph(g, return_attention=False))
        H = torch.stack(day_embs, dim=1)
        if return_attention:
            return H, day_attn
        return H

    def forward_from_day_embeddings(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, T, hidden_channels]
        """
        if self.use_cls_token:
            B = H.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            H = torch.cat([cls, H], dim=1)

        H = H + self.pos_emb[: H.size(1)].unsqueeze(0)
        H = self.transformer(H)

        if self.use_cls_token:
            out = self.head(H[:, 0])
        else:
            out = self.head(H.mean(dim=1))

        act = getattr(self, "output_activation", "softplus" if self.use_softplus else "identity")
        if act == "softplus":
            out = F.softplus(out)
        elif act == "sigmoid":
            out = torch.sigmoid(out)
        elif act == "identity" or act == "none":
            pass
        else:
            raise ValueError(f"Unknown output_activation: {act}")

        return out

    def forward(self, graphs: List[Batch]) -> torch.Tensor:
        H = self.encode_day_graphs(graphs, return_attention=False)
        return self.forward_from_day_embeddings(H)
