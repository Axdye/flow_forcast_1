"""Sparse road-graph construction and propagation utilities.

v1 uses the published binary, undirected topology.  ``A = D^-1 (W + I)`` and
``A[i, j]`` means node ``i`` aggregates node ``j``.  Graph diffusion is
``S_{r+1} = A S_r`` and is always performed with sparse COO matrix-vector
products on ``[N, B*D]`` reshapes, never by materializing ``A^2`` / ``A^3`` or
``[B, K, T, N, N]`` tensors.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch

from sbona.contracts import GraphContext


class GraphError(ValueError):
    """Raised for malformed topology or node alignment."""


def load_edge_list(path: str | Path, node_ids: np.ndarray | None = None) -> np.ndarray:
    """Load an edge list as ``[E, 2]`` integer pairs.

    Accepted formats:
      * ``.npz`` with ``edge_index`` ``[2, E]`` or ``edges`` ``[E, 2]``;
      * ``.npy`` with ``[2, E]`` or ``[E, 2]``;
      * ``.csv`` / ``.txt`` with an optional ``from,to,cost`` header.

    If ``node_ids`` maps file ids to column indices, the mapping is applied.
    """
    path = Path(path)
    if not path.is_file():
        raise GraphError(f"graph file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "edge_index" in archive:
                edges = np.asarray(archive["edge_index"])
                if edges.ndim == 2 and edges.shape[0] == 2:
                    edges = edges.T
            elif "edges" in archive:
                edges = np.asarray(archive["edges"])
            else:
                raise GraphError(f"{path} has neither 'edge_index' nor 'edges'")
    elif suffix == ".npy":
        edges = np.load(path, allow_pickle=False)
    elif suffix in (".csv", ".txt"):
        rows: list[tuple[int, int]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    i, j = int(row[0]), int(row[1])
                except (TypeError, ValueError):
                    continue  # header or comment
                rows.append((i, j))
        if not rows:
            raise GraphError(f"no edges parsed from {path}")
        edges = np.asarray(rows, dtype=np.int64)
    else:
        raise GraphError(f"unsupported graph format: {path.suffix}")

    edges = np.asarray(edges)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise GraphError(f"edge list must be [E, 2], got {edges.shape}")
    edges = edges.astype(np.int64, copy=False)
    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        mapping = {int(node_id): index for index, node_id in enumerate(node_ids)}
        try:
            edges = np.vectorize(mapping.__getitem__, otypes=[np.int64])(edges)
        except KeyError as exc:
            raise GraphError(f"edge endpoint {exc.args[0]} not present in node_ids") from exc
    return edges


def _unique_undirected(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = edges[edges[:, 0] != edges[:, 1]]
    if pairs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0).astype(np.int64)


def build_graph_context(
    num_nodes: int,
    edges: np.ndarray,
    node_ids: np.ndarray | None = None,
    topology: str = "published_binary_undirected",
    node_order_hash_value: str | None = None,
) -> GraphContext:
    """Build row-normalized sparse diffusion and constraint edges."""
    if num_nodes < 1:
        raise GraphError("num_nodes must be >= 1")
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise GraphError(f"edge endpoints must be in [0, {num_nodes})")

    base_undirected = _unique_undirected(edges)
    if topology == "identity":
        diffusion = torch.eye(num_nodes, dtype=torch.float32).to_sparse_coo()
        constraint_edges = base_undirected
        edge_weight = torch.ones(constraint_edges.shape[0], dtype=torch.float32)
    elif topology in ("published_binary_undirected", "derived_binary_undirected"):
        # ``published_binary_undirected``: edges are the publisher's road
        # connectivity.  ``derived_binary_undirected``: edges were binarized
        # from a published weighted/directed matrix by the data step (recorded
        # in the manifest), so the topology label must not claim the former.
        # Constraint/smoothness edges are the undirected road edges.
        constraint_edges = base_undirected
        edge_weight = torch.ones(constraint_edges.shape[0], dtype=torch.float32)

        # Diffusion includes both directions and self loops.  Build dense only
        # during preparation; the runtime graph stays sparse.
        dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for i, j in constraint_edges.tolist():
            dense[i, j] = 1.0
            dense[j, i] = 1.0
        dense += np.eye(num_nodes, dtype=np.float32)
        degree = dense.sum(axis=1, keepdims=True)
        normalized = dense / np.clip(degree, 1.0, None)
        diffusion = torch.from_numpy(normalized.astype(np.float32)).to_sparse_coo()
    else:
        raise GraphError(f"unknown topology {topology!r}")

    if node_ids is None:
        node_ids = np.arange(num_nodes, dtype=np.int64)
    else:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.shape != (num_nodes,):
            raise GraphError("node_ids shape does not match num_nodes")
    if node_order_hash_value is None:
        node_order_hash_value = hashlib.sha256(node_ids.tobytes()).hexdigest()

    digest = hashlib.sha256()
    digest.update(topology.encode("utf-8"))
    digest.update(str(num_nodes).encode("utf-8"))
    digest.update(constraint_edges.tobytes())
    digest.update(b"row_normalized_with_self_loops")
    graph_hash = digest.hexdigest()

    return GraphContext(
        diffusion=diffusion.coalesce(),
        edge_index=torch.from_numpy(constraint_edges.T.copy()),
        edge_weight=edge_weight,
        graph_hash=graph_hash,
        node_order_hash=node_order_hash_value,
    )


def propagate(diffusion: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """One sparse diffusion step.

    ``diffusion``: sparse COO ``[N, N]``.
    ``features``: ``[N, ...]``; internally reshaped to ``[N, -1]`` so a single
    ``torch.sparse.mm`` handles the whole batch.  Sparse ops run in fp32.
    """
    if not features.is_floating_point() or features.dtype != torch.float32:
        features = features.float()
    if features.device != diffusion.device:
        raise GraphError("diffusion and features must be on the same device")
    trailing = features.shape[1:]
    flat = features.reshape(features.shape[0], -1)
    out = torch.sparse.mm(diffusion, flat)
    return out.reshape((features.shape[0],) + trailing)


def multi_hop_basis(
    diffusion: torch.Tensor,
    features: torch.Tensor,
    hops: int,
) -> torch.Tensor:
    """Return ``[R+1, N, ...]`` stacked ``S_r`` including ``S_0``."""
    if hops < 0:
        raise GraphError("hops must be >= 0")
    basis = [features]
    current = features
    for _ in range(hops):
        current = propagate(diffusion, current)
        basis.append(current)
    return torch.stack(basis, dim=0)


__all__ = [
    "GraphError",
    "load_edge_list",
    "build_graph_context",
    "propagate",
    "multi_hop_basis",
]
