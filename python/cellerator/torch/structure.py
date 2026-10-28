"""Logical sparse structure, identities, and caller-built workset contracts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import torch
from torch import Tensor


@dataclass(frozen=True)
class SupportSpec:
    """Logical directed pair support with stable IDs and canonical ordinals."""

    edge_ids: tuple[int, ...]
    sources: tuple[int, ...]
    destinations: tuple[int, ...]
    structure_epoch: int = 1

    def __post_init__(self):
        if not (len(self.edge_ids) == len(self.sources) == len(self.destinations)):
            raise ValueError("edge IDs, sources, and destinations must have equal lengths")
        if self.structure_epoch < 0:
            raise ValueError("structure_epoch must be nonnegative")
        if any(edge <= 0 for edge in self.edge_ids):
            raise ValueError("logical edge IDs must be positive")
        if any(entity < 0 for entity in self.sources + self.destinations):
            raise ValueError("endpoint ordinals must be nonnegative")
        if tuple(sorted(self.edge_ids)) != self.edge_ids or len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("edge IDs must be unique and sorted in canonical order")


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    offset: int
    shape: tuple[int, ...]
    numel: int


def sparse_parameter_layout(*, entities: int, mechanisms: int,
                            rank_count: int, width: int = 64) -> tuple[list[dict], Tensor]:
    """Describe the one shared FP32 parameter vector used by both slices."""
    if min(entities, mechanisms, rank_count, width) <= 0:
        raise ValueError("parameter dimensions must be positive")
    blocks = []
    values = []
    offset = 0
    shapes = (
        ("entity_embedding", (entities, width)),
        ("rank_embedding", (rank_count, width)),
        ("scorer", (width,)),
        ("transition_A", (mechanisms, width, width)),
        ("transition_B", (mechanisms, width, width)),
        ("transition_bias", (mechanisms, width)),
    )
    for name, shape in shapes:
        count = 1
        for dimension in shape:
            count *= dimension
        blocks.append({"name": name, "offset": offset, "shape": list(shape),
                       "numel": count})
        # Small initial values keep the first tanh transitions in their smooth
        # range; all blocks still share the same canonical flat owner.
        scale = 1.0 / max(width, 1) ** 0.5
        values.append(torch.randn(count, dtype=torch.float32) * scale)
        offset += count
    return blocks, torch.cat(values).contiguous()


def _stable_scalar_ids(axis_words: tuple[int, ...], count: int) -> list[tuple[int, int]]:
    if len(axis_words) != 8:
        raise ValueError("parameter axis identity requires eight uint64 words")
    if any(int(word) < 0 or int(word) >= 1 << 64 for word in axis_words):
        raise ValueError("parameter axis words must fit uint64")
    namespace = b"cellerator.sparse.parameter-block.v1" + b"".join(
        int(word).to_bytes(8, "little", signed=False) for word in axis_words)
    result = []
    for index in range(count):
        digest = hashlib.blake2b(namespace + index.to_bytes(8, "little"), digest_size=16).digest()
        result.append((int.from_bytes(digest[:8], "little"),
                       int.from_bytes(digest[8:], "little")))
    return result


@dataclass(frozen=True)
class PerCellWorkset:
    """Device-resident CSR view; overflow must be reported before construction."""

    cell_offsets: Tensor
    edge_ordinals: Tensor
    logical_edge_ids: Tensor
    gate_value_indices: Tensor
    structure_epoch: int
    state_generation: int
    capacity: int

    def validate(self, *, cells: int, edge_count: int, structure_epoch: int) -> None:
        arrays = (self.cell_offsets, self.edge_ordinals,
                  self.logical_edge_ids, self.gate_value_indices)
        if any(value.device.type != "cuda" for value in arrays):
            raise ValueError("steady-state workset tensors must be CUDA resident")
        if any(value.dtype != torch.int64 or value.ndim != 1 or not value.is_contiguous()
               for value in arrays):
            raise ValueError("workset arrays must be contiguous rank-one int64 tensors")
        if self.cell_offsets.numel() != cells + 1:
            raise ValueError("cell_offsets must have cells + 1 entries")
        if not (self.edge_ordinals.numel() == self.logical_edge_ids.numel()
                == self.gate_value_indices.numel()):
            raise ValueError("workset item arrays must have equal lengths")
        if self.edge_ordinals.numel() > self.capacity:
            raise OverflowError("workset exceeds its declared capacity")
        if self.structure_epoch != structure_epoch:
            raise ValueError("workset structure epoch is stale")
        if edge_count < 0:
            raise ValueError("edge_count must be nonnegative")


@dataclass(frozen=True)
class GateValues:
    probabilities: Tensor
    admitted: Tensor
    train_values: Tensor
    logical_edge_ids: Tensor
    edge_ordinals: Tensor
    structure_epoch: int
    state_generation: int


def _snapshot_edges(snapshot: dict) -> dict[int, tuple[int, int]]:
    """Decode the pair-specialized numerical view from a logical snapshot."""
    result = {}
    for name in ("base", "delta"):
        relation = snapshot[name]
        ids = relation["edge_ids"]
        offsets = relation["participant_offsets"]
        roles = relation["participant_roles"]
        entities = relation["participant_entity_ids"]
        if (len(offsets) != len(ids) + 1 or len(offsets) == 0
                or int(offsets[0]) != 0 or int(offsets[-1]) != len(entities)
                or len(roles) != len(entities)):
            raise ValueError("logical support snapshot has malformed participant arrays")
        previous_id = 0
        for index, edge_id in enumerate(ids.tolist()):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            edge_id = int(edge_id)
            if edge_id <= previous_id or edge_id in result:
                raise ValueError("logical support snapshot IDs are not unique and sorted")
            if (begin < 0 or end < begin or end > len(entities)
                    or end - begin != 2 or roles[begin:end].tolist() != [0, 1]):
                raise ValueError("numeric relation projection requires ordered pair roles 0/1")
            endpoints = tuple(int(value) - 1 for value in entities[begin:end])
            if min(endpoints) < 0:
                raise ValueError("logical support snapshot endpoints must be positive identities")
            result[edge_id] = endpoints
            previous_id = edge_id
    return result
