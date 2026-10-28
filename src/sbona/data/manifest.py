"""Data manifest, audit and raw-file IO."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA_VERSION = 1


class DataError(RuntimeError):
    """Raised for missing, malformed or inconsistent data."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass
class FileRecord:
    path: str
    sha256: str
    size_bytes: int
    role: str


@dataclass
class AuditStats:
    num_steps: int
    num_nodes: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    zero_fraction: float
    negative_fraction: float
    nan_fraction: float
    inf_fraction: float
    per_node_valid_count_min: int
    per_node_valid_count_max: int
    per_node_valid_count_mean: float
    fully_invalid_nodes: list[int] = field(default_factory=list)
    constant_nodes: list[int] = field(default_factory=list)


@dataclass
class DatasetManifest:
    dataset: str
    source: str
    license: str
    files: list[FileRecord]
    shape: tuple[int, int]
    flow_channel: int
    unit: str
    interval_minutes: int
    start_index: int
    end_index: int
    timestamps_source: str
    node_ids: list[int]
    node_order_hash: str
    adjacency_source: str
    missing_definition: str
    audit: AuditStats
    calendar_available: bool
    protocol_id: str = "flow_12x12_v1"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["files"] = [FileRecord(**item) for item in raw["files"]]
        raw["audit"] = AuditStats(**raw["audit"])
        raw["shape"] = tuple(raw["shape"])
        return cls(**raw)


def audit_array(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    fully_invalid_threshold: int = 1,
) -> AuditStats:
    """Compute honest audit statistics on raw flow.

    ``data`` is ``[M, N]``.  Invalid entries are those with ``mask == False``
    or non-finite values.  We do not treat zero as missing.
    """
    if data.ndim != 2:
        raise DataError(f"data must be 2-D [M, N], got {data.shape}")
    finite = np.isfinite(data)
    if mask is None:
        valid = finite
    else:
        if mask.shape != data.shape:
            raise DataError(f"mask shape {mask.shape} != data shape {data.shape}")
        valid = mask.astype(bool) & finite
    values = data[valid]
    counts = valid.sum(axis=0)
    fully_invalid = np.where(counts < fully_invalid_threshold)[0].tolist()
    constant = [
        int(i)
        for i in range(data.shape[1])
        if counts[i] > 0 and float(data[valid[:, i], i].max() - data[valid[:, i], i].min()) == 0.0
    ]
    return AuditStats(
        num_steps=int(data.shape[0]),
        num_nodes=int(data.shape[1]),
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None,
        mean=float(values.mean()) if values.size else None,
        std=float(values.std()) if values.size else None,
        zero_fraction=float((values == 0).mean()) if values.size else 0.0,
        negative_fraction=float((values < 0).mean()) if values.size else 0.0,
        nan_fraction=float(np.isnan(data).mean()),
        inf_fraction=float(np.isinf(data).mean()),
        per_node_valid_count_min=int(counts.min()) if counts.size else 0,
        per_node_valid_count_max=int(counts.max()) if counts.size else 0,
        per_node_valid_count_mean=float(counts.mean()) if counts.size else 0.0,
        fully_invalid_nodes=fully_invalid,
        constant_nodes=constant,
    )


def load_raw_array(
    path: str | Path,
    flow_channel: int = 0,
    missing_sentinel: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load a raw dataset.

    Accepted formats:
      * ``.npy`` with ``[M, N]`` or ``[M, N, C]``;
      * ``.npz`` containing ``data`` with one of the above, plus optional
        ``mask``, ``timestamps`` and ``node_ids``.

    Returns ``(data [M,N] float32, source_mask [M,N] bool | None, timestamps)``.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        mask = None
        timestamps = None
        node_ids = None
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "data" not in archive:
                raise DataError(f"{path} must contain a 'data' array")
            data = archive["data"]
            mask = archive["mask"] if "mask" in archive else None
            timestamps = archive["timestamps"] if "timestamps" in archive else None
            node_ids = archive["node_ids"] if "node_ids" in archive else None
    else:
        raise DataError(f"unsupported data format: {path.suffix}")

    data = np.asarray(data)
    if data.ndim == 3:
        if not (0 <= flow_channel < data.shape[-1]):
            raise DataError(f"flow_channel {flow_channel} out of range for channels={data.shape[-1]}")
        data = data[..., flow_channel]
    if data.ndim != 2:
        raise DataError(f"data must be [M, N] after channel selection, got {data.shape}")
    data = data.astype(np.float32, copy=False)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 3:
            mask = mask[..., flow_channel]
        if mask.shape != data.shape:
            raise DataError(f"mask shape {mask.shape} != data shape {data.shape}")
    if missing_sentinel is not None:
        sentinel = np.isclose(data, float(missing_sentinel))
        if mask is None:
            mask = ~sentinel
        else:
            mask = mask & ~sentinel
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.shape[0] != data.shape[0]:
            raise DataError("timestamps length does not match data")
    if node_ids is not None:
        node_ids = np.asarray(node_ids)
        if node_ids.shape[0] != data.shape[1]:
            raise DataError("node_ids length does not match data")
        # ``read_node_ids`` below is the single supported accessor; the tuple
        # return intentionally stays ``(data, mask, timestamps)``.
    return data, mask, timestamps


def load_timestamps_file(path: str | Path, num_steps: int) -> np.ndarray:
    """Load an external calendar file for the raw time axis.

    Accepted layouts:
      * ``[M]`` ``datetime64`` (absolute timestamps);
      * ``[M]`` integer step indices or minutes since epoch (the latter only
        when values exceed ``1e5``);
      * ``[M, 2]`` float table ``(time_of_day in [0, 1), day_of_week in 0..6)``.

    The file must cover exactly the ``M`` raw steps of the dataset.  A calendar
    is only trusted when the array order is verified against the flow array;
    the manifest records the file hash so a swap is detectable.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"timestamps_file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "timestamps" not in archive:
                raise DataError(f"{path} must contain a 'timestamps' array")
            stamps = np.asarray(archive["timestamps"])
    elif suffix == ".npy":
        stamps = np.asarray(np.load(path, allow_pickle=False))
    else:
        raise DataError(f"unsupported timestamps_file format: {path.suffix}")
    if stamps.ndim not in (1, 2) or (stamps.ndim == 2 and stamps.shape[1] != 2):
        raise DataError(f"timestamps must be [M] or [M, 2], got {stamps.shape}")
    if stamps.shape[0] != num_steps:
        raise DataError(
            f"timestamps length {stamps.shape[0]} does not match {num_steps} raw steps"
        )
    return stamps


def read_node_ids(data_path: str | Path, num_nodes: int) -> np.ndarray:
    """Read optional node ids from an npz file; defaults to 0..N-1."""
    path = Path(data_path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "node_ids" in archive:
                ids = np.asarray(archive["node_ids"])
                if ids.shape != (num_nodes,):
                    raise DataError("node_ids must have shape [N]")
                return ids.astype(np.int64)
    return np.arange(num_nodes, dtype=np.int64)


def read_node_ids_file(path: str | Path, num_nodes: int) -> np.ndarray:
    """Read an explicit node-id order from ``.npy``/``.npz``/text/CSV.

    Text files use the first column and may contain a non-numeric header; the
    result must contain exactly ``num_nodes`` integer ids.
    """
    path = Path(path)
    if not path.is_file():
        raise DataError(f"node_ids_file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        ids = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = "node_ids" if "node_ids" in archive else ("ids" if "ids" in archive else None)
            if key is None:
                raise DataError(f"{path} must contain 'node_ids' or 'ids'")
            ids = archive[key]
    else:
        values: list[int] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                token = line.strip().split(",")[0].strip()
                if not token:
                    continue
                try:
                    values.append(int(float(token)))
                except ValueError:
                    continue  # header or comment
        ids = np.asarray(values, dtype=np.int64)
    ids = np.asarray(ids).reshape(-1).astype(np.int64)
    if ids.shape != (num_nodes,):
        raise DataError(
            f"node_ids_file {path} has {ids.size} ids, expected {num_nodes}"
        )
    if np.unique(ids).size != ids.size:
        raise DataError(f"node_ids_file {path} contains duplicate ids")
    return ids


def node_order_hash(node_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(node_ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


def make_file_record(path: str | Path, role: str, display_path: str | None = None) -> FileRecord:
    path = Path(path)
    return FileRecord(
        path=display_path or str(path),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        role=role,
    )


__all__ = [
    "DataError",
    "FileRecord",
    "AuditStats",
    "DatasetManifest",
    "sha256_file",
    "sha256_array",
    "audit_array",
    "load_raw_array",
    "load_timestamps_file",
    "read_node_ids",
    "read_node_ids_file",
    "node_order_hash",
    "make_file_record",
]
