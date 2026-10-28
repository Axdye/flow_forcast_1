"""Shared helpers for the vendored backbone adapters.

Three kinds of glue live here, and nothing else:

1. **Calendar channels.**  The contract stores ``day_of_week`` as an integer
   ``0..6`` while the upstream repositories disagree about whether they want a
   weekday index, ``dow/7`` or a one-hot block.  Each adapter therefore asks
   for the exact variant it needs through :func:`stack_features` instead of
   silently sharing one convention (a mismatch does not raise, it just learns
   the wrong embedding).
2. **Dense graph artefacts.**  Upstream models consume dense ``[N, N]``
   adjacency matrices, hop matrices or Laplacian eigenvectors while
   :class:`GraphContext` stores a sparse row-normalised diffusion operator.
   The conversions are explicit and cached per ``(graph_hash, device)``.
3. **The adapter shell.**  :class:`TorchBackboneAdapter` performs the shape /
   node-alignment checks required by the contract and then delegates to
   ``forward_features`` which returns ``(node_hidden, base_prediction)`` from a
   *single* forward pass.

Orientation of the graph
------------------------
``GraphContext.diffusion`` satisfies ``A[i, j] == `` "node ``i`` aggregates
node ``j``".  Most upstream implementations instead write
``out[v] = sum_w A[v, w] * x[w]`` (``einsum('ncvl,vw->ncwl')`` in Graph
WaveNet) which is the transpose of that convention.  :func:`graph_supports`
returns both orientations and every adapter documents which one it feeds.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import torch
from torch import Tensor, nn

from sbona.backbones.base import BackboneError, BackboneWrapper, check_input_against_graph
from sbona.contracts import BackboneOutput, GraphContext, ModelInput

T = TypeVar("T")

#: Calendar channels understood by :func:`stack_features`.
FEATURE_SPECS = ("flow", "tod", "dow_index", "dow_fraction", "dow_onehot")


def _expand(channel: Tensor, reference: Tensor) -> Tensor:
    """Broadcast ``[B, L]`` or ``[B, L, C]`` to ``[B, L, N, C]``."""
    batch, length, nodes, _ = reference.shape
    if channel.dim() == 2:
        channel = channel.unsqueeze(-1)
    return channel.to(reference.dtype)[:, :, None, :].expand(batch, length, nodes, channel.shape[-1])


def time_of_day_channel(inp: ModelInput) -> Tensor:
    """``[B, L, N, 1]`` in ``[0, 1)``."""
    return _expand(inp.time_of_day, inp.history)


def day_of_week_index(inp: ModelInput) -> Tensor:
    """``[B, L, N, 1]`` integer-valued weekday index ``0..6``."""
    return _expand(inp.day_of_week, inp.history)


def day_of_week_fraction(inp: ModelInput) -> Tensor:
    """``[B, L, N, 1]`` weekday position inside one week, ``0 <= x < 1``."""
    return day_of_week_index(inp) / 7.0


def day_of_week_one_hot(inp: ModelInput) -> Tensor:
    """``[B, L, N, 7]`` one-hot weekday block (PDFormer's layout)."""
    index = inp.day_of_week.long()
    if bool((index < 0).any()) or bool((index > 6).any()):
        raise BackboneError("day_of_week must be in 0..6 for a one-hot calendar block")
    one_hot = torch.nn.functional.one_hot(index, num_classes=7).float()
    return _expand(one_hot, inp.history)


def stack_features(inp: ModelInput, spec: Sequence[str]) -> Tensor:
    """Concatenate the requested channel groups after the flow channel.

    ``spec`` entries must come from :data:`FEATURE_SPECS`; ``"flow"`` is always
    the first group and must appear at most once.  The returned tensor is
    ``[B, L, N, C]`` and is contiguous so upstream ``view``/``reshape`` calls
    work unchanged.
    """
    groups: list[Tensor] = []
    seen: set[str] = set()
    for name in spec:
        if name not in FEATURE_SPECS:
            raise BackboneError(f"unknown feature group {name!r}; choose from {FEATURE_SPECS}")
        if name in seen:
            raise BackboneError(f"feature group {name!r} requested twice")
        seen.add(name)
        if name == "flow":
            groups.append(inp.history)
        elif name == "tod":
            groups.append(time_of_day_channel(inp))
        elif name == "dow_index":
            groups.append(day_of_week_index(inp))
        elif name == "dow_fraction":
            groups.append(day_of_week_fraction(inp))
        else:
            groups.append(day_of_week_one_hot(inp))
    if not groups:
        raise BackboneError("stack_features needs at least one group")
    return torch.cat(groups, dim=-1).contiguous()


def future_calendar(
    inp: ModelInput, steps_per_day: int, horizon: int
) -> tuple[Tensor, Tensor]:
    """Deterministically extrapolate calendar features for the forecast window.

    Only the *calendar* is extrapolated, never the target.  ``time_of_day`` and
    ``day_of_week`` are known in advance, so models whose official code consumes
    future covariates (HimNet, ST-SSDL) keep their published input layout while
    the contract stays history-only.

    Returns ``(tod [B, horizon] float32 in [0, 1), dow [B, horizon] int64)``.
    """
    if steps_per_day < 1:
        raise BackboneError("steps_per_day must be >= 1")
    if horizon < 1:
        raise BackboneError("horizon must be >= 1")
    last_tod = inp.time_of_day[:, -1].float()
    last_dow = inp.day_of_week[:, -1].float()
    offsets = torch.arange(1, horizon + 1, device=last_tod.device, dtype=last_tod.dtype)
    advanced = last_tod[:, None] + offsets[None, :] / float(steps_per_day)
    tod = torch.remainder(advanced, 1.0)
    # ``floor`` overshoots by one exactly when the window crosses midnight;
    # clamping the remainder guards against 0.9999999 -> 1.0000001 noise.
    tod = torch.where(tod >= 1.0, tod - 1.0, tod)
    day_jump = torch.floor(advanced)
    dow = torch.remainder(last_dow[:, None] + day_jump, 7.0).long()
    return tod, dow


# ---------------------------------------------------------------------------
# Graph conversions
# ---------------------------------------------------------------------------


def dense_diffusion(graph: GraphContext) -> Tensor:
    """Dense ``A`` with ``A[i, j]`` = "row ``i`` aggregates column ``j``"."""
    return graph.diffusion.to_dense().float()


def graph_supports(graph: GraphContext, symmetric: bool = True) -> tuple[Tensor, ...]:
    """Adjacency list in upstream orientation ``out[v] = sum_w A[v, w] x[w]``.

    ``supports[0]`` is ``A.T`` (upstream convention), and ``supports[1]`` is
    ``A`` itself, which is the ``doubletransition`` pair used by Graph WaveNet
    and D2STGNN.  With ``symmetric=False`` only the transposed support is
    returned.
    """
    adjacency = dense_diffusion(graph)
    transposed = adjacency.transpose(0, 1).contiguous()
    return (transposed, adjacency) if symmetric else (transposed,)


def hop_distance(graph: GraphContext, max_hops: int = 5) -> Tensor:
    """Integer hop count ``[N, N]``; unreachable pairs get ``max_hops + 1``.

    BFS over the undirected constraint edges.  Used by PDFormer, whose official
    code takes a precomputed shortest-path matrix and masks nodes further away
    than ``far_mask_delta``.
    """
    nodes = graph.num_nodes
    adjacency = torch.zeros((nodes, nodes), dtype=torch.bool)
    edges = graph.edge_index.detach().cpu()
    if edges.numel() > 0:
        adjacency[edges[0], edges[1]] = True
        adjacency[edges[1], edges[0]] = True
    adjacency.fill_diagonal_(True)

    sentinel = max_hops + 1
    distance = torch.full((nodes, nodes), sentinel, dtype=torch.int64)
    reached = torch.eye(nodes, dtype=torch.bool)
    distance[reached] = 0
    adjacency_float = adjacency.to(torch.float32)
    for step in range(1, max_hops + 1):
        expanded = reached.to(torch.float32) @ adjacency_float > 0
        newly = expanded & ~reached
        if not bool(newly.any()):
            break
        distance[newly] = step
        reached = reached | newly
    return distance


def laplacian_pe(adjacency: Tensor, dim: int) -> Tensor:
    """First ``dim`` non-trivial Laplacian eigenvectors ``[N, dim]``.

    Same construction as PDFormer's ``LaplacianPE`` input: symmetric
    normalised Laplacian ``I - D^-1/2 A D^-1/2`` of the (symmetrised) road
    graph.  Signs are canonicalised (sum over nodes non-negative) so repeated
    builds on the same graph agree.
    """
    if dim < 0:
        raise BackboneError("laplacian_pe dim must be >= 0")
    nodes = adjacency.shape[0]
    if dim == 0:
        return adjacency.new_zeros((nodes, 0))
    matrix = adjacency.to(torch.float64)
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    degree = matrix.sum(dim=-1)
    inv_sqrt = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    laplacian = torch.eye(nodes, dtype=matrix.dtype, device=matrix.device)
    laplacian = laplacian - inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    laplacian = 0.5 * (laplacian + laplacian.transpose(0, 1))
    _, vectors = torch.linalg.eigh(laplacian)
    take = vectors[:, 1 : 1 + dim]
    if take.shape[1] < dim:
        take = torch.cat(
            [take, take.new_zeros((nodes, dim - take.shape[1]))], dim=1
        )
    signs = torch.sign(take.sum(dim=0))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (take * signs).float()


class GraphCache:
    """Per-adapter cache for dense graph artefacts.

    Keyed by ``(graph_hash, device)`` so moving the graph or swapping datasets
    cannot silently reuse stale tensors.  Entries are built lazily once per
    process per adapter, which keeps the BFS/eigendecomposition out of the
    training loop.
    """

    def __init__(self) -> None:
        self._key: tuple[str, str] | None = None
        self._items: dict[str, Any] = {}

    def get(self, graph: GraphContext, name: str, builder: Callable[[], T]) -> T:
        key = (graph.graph_hash, str(graph.diffusion.device))
        if key != self._key:
            self._key = key
            self._items.clear()
        if name not in self._items:
            self._items[name] = builder()
        return self._items[name]


# ---------------------------------------------------------------------------
# Adapter shell
# ---------------------------------------------------------------------------


class TorchBackboneAdapter(BackboneWrapper):
    """Common adapter shell: validate input, run one forward, check output.

    Subclasses implement :meth:`forward_features`, which receives the validated
    :class:`ModelInput` and :class:`GraphContext` and returns
    ``(node_hidden [B, N, D], base_pred_z [B, T, N, 1])``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        partial_patterns: tuple[str, ...] = (),
        node_feature_dim: int,
        num_nodes: int,
        in_steps: int,
        out_steps: int,
    ) -> None:
        super().__init__(
            model=model,
            mode=mode,
            partial_patterns=partial_patterns,
            node_feature_dim=node_feature_dim,
        )
        self.num_nodes = int(num_nodes)
        self.in_steps = int(in_steps)
        self.out_steps = int(out_steps)
        self._graph_cache = GraphCache()

    @abstractmethod
    def forward_features(
        self, inp: ModelInput, graph: GraphContext
    ) -> tuple[Tensor, Tensor]:
        """Return ``(node_hidden [B, N, D], base_pred_z [B, T, N, 1])``."""

    def encode_and_predict(self, inp: ModelInput, graph: GraphContext) -> BackboneOutput:
        check_input_against_graph(inp, graph)
        if inp.num_nodes != self.num_nodes:
            raise BackboneError(
                f"{type(self).__name__} built for {self.num_nodes} nodes, "
                f"input has {inp.num_nodes}"
            )
        if inp.input_steps != self.in_steps:
            raise BackboneError(
                f"{type(self).__name__} built for L={self.in_steps}, "
                f"input has L={inp.input_steps}"
            )
        with self._context():
            hidden, prediction = self.forward_features(inp, graph)
        expected = (inp.batch_size, self.out_steps, self.num_nodes, 1)
        if tuple(prediction.shape) != expected:
            raise BackboneError(
                f"{type(self).__name__} produced prediction {tuple(prediction.shape)}, "
                f"expected {expected}"
            )
        if hidden.dim() != 3 or hidden.shape[:2] != (inp.batch_size, self.num_nodes):
            raise BackboneError(
                f"{type(self).__name__} produced hidden {tuple(hidden.shape)}, "
                "expected [B, N, D]"
            )
        return BackboneOutput(
            base_pred_z=prediction.float(), node_hidden=hidden.float()
        )

    # Small helpers shared by several adapters -----------------------------

    @staticmethod
    def _validate_horizon(prediction: Tensor, out_steps: int, name: str) -> Tensor:
        if prediction.shape[1] != out_steps:
            raise BackboneError(
                f"{name} produced T={prediction.shape[1]}, expected {out_steps}"
            )
        return prediction


__all__ = [
    "FEATURE_SPECS",
    "GraphCache",
    "TorchBackboneAdapter",
    "dense_diffusion",
    "day_of_week_fraction",
    "day_of_week_index",
    "day_of_week_one_hot",
    "future_calendar",
    "graph_supports",
    "hop_distance",
    "laplacian_pe",
    "stack_features",
    "time_of_day_channel",
]
