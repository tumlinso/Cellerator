"""Torch adapters and controlled fixtures for native sparse support programs.

The native owner holds one flat FP32 master vector. This module provides the
Torch-facing parameter views, learned gate math, and the two first application
shapes; relation and latent numerical operators remain native.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import contextvars
import hashlib
import time
from typing import Iterable, Optional
import weakref

import torch
from torch import Tensor, nn



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
        for index, edge_id in enumerate(ids.tolist()):
            begin, end = int(offsets[index]), int(offsets[index + 1])
            if end - begin != 2 or roles[begin:end].tolist() != [0, 1]:
                raise ValueError("numeric relation projection requires ordered pair roles 0/1")
            result[int(edge_id)] = tuple(int(value) - 1 for value in entities[begin:end])
    return result


def learned_gates(scores: Tensor, *, temperature: float, threshold: float,
                  prune: bool, straight_through: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    """Return p, admission, and trainable q with the declared explicit STE."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    p = torch.sigmoid(scores / temperature)
    admitted = p >= threshold
    if not prune:
        return p, torch.ones_like(admitted), p
    q = p * admitted.to(p.dtype)
    if straight_through:
        q = p + (q - p).detach()
    return p, admitted, q


class _ParameterReadLease:
    """Complete a native reader ticket when its autograd graph is abandoned."""

    __slots__ = ("_module", "ticket", "completed", "__weakref__")

    def __init__(self, module, ticket):
        self._module = weakref.ref(module)
        self.ticket = ticket
        self.completed = False

    def complete(self):
        if self.completed:
            return
        module = self._module()
        if module is not None:
            module._complete_read(self)
        else:
            self.completed = True

    def __del__(self):
        try:
            self.complete()
        except Exception:
            # Destructors must not surface errors while Python is collecting an
            # abandoned graph; normal guarded publication reports native errors.
            pass


class _SparseParameterRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameters: Tensor, module, lease):
        ctx.module = module
        ctx.lease = lease
        return parameters.view_as(parameters)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        ctx.module._complete_read(ctx.lease)
        return gradient, None, None


class SparseSupportModule(nn.Module):
    """A trainable Torch view over one native sparse owner and support handle."""

    def __init__(self, handle: object, support: SupportSpec, *,
                 parameter_names: Iterable[str] = (),
                 parameter_blocks: Optional[dict[str, tuple[int, ...]]] = None,
                 support_index: object = None):
        super().__init__()
        import importlib
        torch_native = importlib.import_module("._torch", "cellerator.torch")
        from .mechanism import attach_native_owner

        self._native = handle
        self._support_index = support_index or handle.support
        self.support = support
        self._torch_native = torch_native
        (self._source_endpoint_indices,
         self._destination_endpoint_indices) = torch_native.sparse_endpoint_indices(handle)
        self.parameters_flat = nn.Parameter(torch_native.sparse_parameters(handle), requires_grad=True)
        attach_native_owner(self, handle, self.parameters_flat, torch_native)
        self._active_parameters = contextvars.ContextVar(
            f"cellerator_sparse_parameters_{id(self)}", default=None)
        self._read_tickets = weakref.WeakSet()
        self._pending_promotions: dict[int, tuple[int, int, int, int, float]] = {}
        self._pending_revocations: set[int] = set()
        # Learned entity-to-mechanism support is a discrete, bounded structural
        # view of the shared parameters. It is refreshed at forward-window
        # boundaries after the owner generation changes and held fixed through
        # that forward/backward tape.
        self._mechanism_route_generation = -1
        self._mechanism_route_postings: tuple[tuple[int, ...], ...] = ()
        self._gate_admitted_total: Optional[Tensor] = None
        self._gate_candidate_total: Optional[Tensor] = None
        self.application_state: dict[str, object] = {}
        self.gate_policy = {
            "temperature": 1.0, "threshold": 0.5, "estimator": "soft",
            "derivative_candidate_policy": "admitted_plus_bounded_dormant",
        }
        raw_blocks = torch_native.sparse_parameter_blocks(handle)
        self._blocks = {entry["name"]: ParameterBlock(
            entry["name"], int(entry["offset"]), tuple(map(int, entry["shape"])),
            int(entry["numel"])) for entry in raw_blocks}
        if parameter_blocks is not None:
            for name, shape in parameter_blocks.items():
                block = self._blocks.get(name)
                if block is None or block.shape != tuple(shape):
                    raise ValueError(f"native parameter block {name!r} does not match the declaration")
        if parameter_names and set(parameter_names) != set(self._blocks):
            raise ValueError("native parameter block names do not match the declaration")
        if self.parameters_flat.ndim != 1 or self.parameters_flat.dtype != torch.float32:
            raise RuntimeError("native sparse owner must expose one flat FP32 parameter leaf")
        if self.parameters_flat.numel() != sum(block.numel for block in self._blocks.values()):
            raise RuntimeError("native sparse parameter blocks do not cover the flat owner")
        self.structure_epoch = int(torch_native.sparse_structure_epoch(handle))

    def block(self, name: str) -> Tensor:
        block = self._blocks[name]
        parameters = self._active_parameters.get()
        if parameters is None:
            raise RuntimeError("sparse parameter blocks may only be read inside a pinned forward window")
        return parameters.narrow(0, block.offset, block.numel).view(block.shape)

    def _complete_read(self, lease: _ParameterReadLease) -> None:
        if lease.completed:
            return
        self._torch_native.sparse_complete_read(self._native, lease.ticket)
        lease.completed = True
        self._read_tickets.discard(lease)

    def retire_tapes(self) -> None:
        """Explicitly retire intentionally discarded forward graphs before publication."""
        for lease in tuple(self._read_tickets):
            self._complete_read(lease)

    def stage_promotion(self, edge_id: int, source: int, destination: int, *,
                        discovery_round_id: int = 0, scorer_generation: int = 0,
                        proposal_score: float = 0.0) -> None:
        """Stage a directed support route; publication remains a window boundary."""
        import numpy as np

        participants = np.asarray([0, 1], dtype=np.int32)
        entities = np.asarray([source + 1, destination + 1], dtype=np.int64)
        if edge_id <= 0 or source < 0 or destination < 0:
            raise ValueError("support promotion requires positive edge ID and nonnegative endpoints")
        entity_count = self._blocks["entity_embedding"].shape[0]
        if source >= entity_count or destination >= entity_count:
            raise ValueError("support promotion endpoint is outside the prepared logical domain")
        if edge_id in self._pending_promotions or edge_id in self.support.edge_ids:
            raise ValueError("support promotion edge ID is already active or pending")
        prepared = self.application_state.get("prepared_capacity")
        if isinstance(prepared, dict):
            base_count = int(prepared.get("initial_support_edges", len(self.support.edge_ids)))
            growth_budget = int(prepared.get("support_growth_budget", 0))
            if len(self.support.edge_ids) - base_count + len(self._pending_promotions) + 1 > growth_budget:
                raise OverflowError("support promotion exceeds the prepared projection growth budget")
        self._support_index.stage_promotion(
            edge_id, participants, entities, discovery_round_id,
            scorer_generation, proposal_score)
        self._pending_promotions[edge_id] = (
            source, destination, discovery_round_id, scorer_generation, proposal_score)

    def stage_revocation(self, edge_id: int) -> None:
        if edge_id not in self.support.edge_ids:
            raise ValueError("support revocation edge ID is not active")
        self._support_index.stage_revocation(edge_id)
        self._pending_revocations.add(edge_id)

    def publish_support(self, next_epoch: int) -> None:
        """Publish staged routes after all training tapes have retired.

        All logical pair arrays are validated before mutating the manager. If
        rebinding the lightweight numerical handle fails after publication, the
        module is poisoned and refuses subsequent execution.
        """
        import numpy as np
        started_at = time.perf_counter()

        if self._read_tickets:
            raise RuntimeError("support publication requires all parameter read tapes to retire")
        snapshot = self._support_index.snapshot()
        edges_by_id = {}
        for name in ("base", "delta"):
            relation = snapshot[name]
            ids = relation["edge_ids"]
            offsets = relation["participant_offsets"]
            roles = relation["participant_roles"]
            entities = relation["participant_entity_ids"]
            for index, edge_id in enumerate(ids.tolist()):
                begin, end = int(offsets[index]), int(offsets[index + 1])
                if end - begin != 2 or roles[begin:end].tolist() != [0, 1]:
                    raise ValueError("numeric relation projection requires ordered pair roles 0/1")
                source, destination = (int(value) - 1 for value in entities[begin:end])
                edges_by_id[int(edge_id)] = (source, destination)
        for edge_id in self._pending_revocations:
            edges_by_id.pop(edge_id, None)
        for edge_id, (source, destination, *_evidence) in self._pending_promotions.items():
            if edge_id in edges_by_id:
                raise ValueError("pending support promotion duplicates an active logical edge ID")
            edges_by_id[edge_id] = (source, destination)
        edges = [(edge_id, *endpoints) for edge_id, endpoints in sorted(edges_by_id.items())]
        edge_ids = tuple(edge[0] for edge in edges)
        sources = tuple(edge[1] for edge in edges)
        destinations = tuple(edge[2] for edge in edges)
        if any(min(source, destination) < 0 for _, source, destination in edges):
            raise ValueError("published support endpoint identity is invalid")
        published = self._support_index.publish_pending_snapshot(next_epoch)
        try:
            published_edges = {}
            for name in ("base", "delta"):
                relation = published[name]
                ids = relation["edge_ids"]
                offsets = relation["participant_offsets"]
                roles = relation["participant_roles"]
                entities = relation["participant_entity_ids"]
                for index, edge_id in enumerate(ids.tolist()):
                    begin, end = int(offsets[index]), int(offsets[index + 1])
                    if end - begin != 2 or roles[begin:end].tolist() != [0, 1]:
                        raise ValueError("published support is not an ordered pair relation")
                    source, destination = (int(value) - 1
                                           for value in entities[begin:end])
                    published_edges[int(edge_id)] = (source, destination)
            if published_edges != edges_by_id or int(published["structure_epoch"]) != next_epoch:
                raise RuntimeError("published support differs from the staged relation")
            self._native = self._torch_native.sparse_rebind_handle(
                self._native, list(edge_ids), list(sources), list(destinations),
                next_epoch)
            self.support = SupportSpec(edge_ids, sources, destinations, next_epoch)
            self.structure_epoch = next_epoch
            (self._source_endpoint_indices,
             self._destination_endpoint_indices) = self._torch_native.sparse_endpoint_indices(self._native)
            self._pending_promotions.clear()
            self._pending_revocations.clear()
            self.application_state["support_publish_seconds"] = (
                float(self.application_state.get("support_publish_seconds", 0.0))
                + time.perf_counter() - started_at)
        except Exception:
            self._poisoned = True
            raise

    @contextmanager
    def _pinned_forward(self):
        ticket = self._torch_native.sparse_pin_read(self._native)
        lease = _ParameterReadLease(self, ticket)
        self._read_tickets.add(lease)
        grad_enabled = torch.is_grad_enabled()
        token = None
        completed = False
        try:
            parameters = _SparseParameterRead.apply(self.parameters_flat, self, lease)
            token = self._active_parameters.set(parameters)
            yield
            completed = True
        finally:
            if token is not None:
                self._active_parameters.reset(token)
            if not grad_enabled or not completed:
                self._complete_read(lease)

    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001
        raise RuntimeError("native sparse parameter storage cannot be moved or recast with Module.to/_apply")

    def _load_from_state_dict(self, *args, **kwargs):
        state_dict = args[0] if args else kwargs.get("state_dict", {})
        prefix = args[1] if len(args) > 1 else kwargs.get("prefix", "")
        if prefix + "parameters_flat" in state_dict:
            raise RuntimeError("restore sparse native owners with load_checkpoint(), not state_dict")
        return super()._load_from_state_dict(*args, **kwargs)

    def relation(self, inputs: Tensor, edge_values: Tensor, gates: GateValues,
                 primal: PerCellWorkset, derivative: PerCellWorkset) -> Tensor:
        cells = inputs.shape[0]
        if getattr(self, "_poisoned", False):
            raise RuntimeError("sparse module was poisoned by a failed support publication")
        primal.validate(cells=cells, edge_count=len(self.support.edge_ids),
                        structure_epoch=self.structure_epoch)
        derivative.validate(cells=cells, edge_count=len(self.support.edge_ids),
                            structure_epoch=self.structure_epoch)
        if gates.structure_epoch != self.structure_epoch:
            raise ValueError("gate values have a stale structure epoch")
        if gates.state_generation != derivative.state_generation:
            raise ValueError("gate candidate state generation is stale")
        if inputs.dtype != torch.float16:
            raise ValueError("native relation input must be FP16")
        if edge_values.dtype != torch.float32 or edge_values.ndim != 1:
            raise ValueError("logical edge values must be rank-one FP32")
        return self._torch_native.sparse_relation_apply(
            inputs.contiguous(), edge_values.contiguous(), gates.train_values.contiguous(),
            gates.admitted.to(torch.uint8).contiguous(),
            primal.cell_offsets, primal.edge_ordinals, primal.logical_edge_ids,
            primal.gate_value_indices, derivative.cell_offsets,
            derivative.edge_ordinals, derivative.logical_edge_ids,
            derivative.gate_value_indices, gates.logical_edge_ids,
            gates.edge_ordinals, primal.state_generation, primal.capacity,
            derivative.state_generation, derivative.capacity,
            gates.state_generation, self._native,
        )

    def logical_edge_values(self) -> Tensor:
        """Derive values from shared endpoint/scorer blocks, never per-edge slots."""
        embedding = self.block("entity_embedding")
        scorer = self.block("scorer").reshape(-1)
        source = embedding.index_select(0, self._source_endpoint_indices)
        destination = embedding.index_select(0, self._destination_endpoint_indices)
        return (source * destination * scorer).sum(dim=-1) / (embedding.shape[-1] ** 0.5)

    def _refresh_mechanism_route_postings(self) -> None:
        """Compile a small learned entity→mechanism posting index.

        Compatibility is scored once per parameter generation from the shared
        entity embeddings and transition input matrices. Each entity retains
        its best route and at most one near-best route, allowing variable
        degree while bounding proposal work. The discrete route choice is a
        structural support decision; no gradient is claimed through it.
        """
        started_at = time.perf_counter()
        generation = int(self._torch_native.sparse_generation(self._native))
        if generation == self._mechanism_route_generation:
            return
        with torch.no_grad():
            entities = self.block("entity_embedding").detach().float()
            transition_b = self.block("transition_B").detach().float()
            # This full compatibility evaluation happens once at a window
            # boundary, not once per event. It is bounded by entities × the
            # small prepared mechanism table; event-time retrieval uses only
            # these postings.
            response = torch.einsum("ew,mvw->emv", entities, transition_b)
            compatibility = response.square().mean(dim=-1)
            route_count = min(2, compatibility.shape[1])
            top = torch.topk(compatibility, k=route_count, dim=1, sorted=True)
            rows: list[tuple[int, ...]] = []
            for scores, mechanisms in zip(top.values.cpu().tolist(),
                                          top.indices.cpu().tolist()):
                chosen = [int(mechanisms[0])]
                if (route_count > 1 and scores[0] > 0.0
                        and scores[1] >= 0.75 * scores[0]):
                    chosen.append(int(mechanisms[1]))
                rows.append(tuple(sorted(set(chosen))))
        self._mechanism_route_postings = tuple(rows)
        self._mechanism_route_generation = generation
        self.application_state["mechanism_route_generation"] = generation
        self.application_state["mechanism_route_count"] = sum(map(len, rows))
        self.application_state["route_cache_refresh_seconds"] = (
            float(self.application_state.get("route_cache_refresh_seconds", 0.0))
            + time.perf_counter() - started_at)

    def affected_mechanism_workset(self, entity_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Retrieve the union of prepared mechanism postings for each cell."""
        if entity_ids.ndim != 2 or entity_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("mechanism-route lookup requires [cells, participants] integer IDs")
        self._refresh_mechanism_route_postings()
        rows: list[list[int]] = []
        for entity_row in entity_ids.detach().cpu().tolist():
            mechanisms = set()
            for entity in entity_row:
                if entity >= 0:
                    if entity >= len(self._mechanism_route_postings):
                        raise ValueError("mechanism-route entity is outside the prepared domain")
                    mechanisms.update(self._mechanism_route_postings[entity])
            rows.append(sorted(mechanisms))
        offsets = [0]
        flat: list[int] = []
        for row in rows:
            flat.extend(row)
            offsets.append(len(flat))
        device = self.parameters_flat.device
        return (torch.tensor(offsets, dtype=torch.int64, device=device),
                torch.tensor(flat, dtype=torch.int32, device=device))

    def reset_gate_diagnostics(self) -> None:
        self._gate_admitted_total = None
        self._gate_candidate_total = None

    def gate_admission_rate(self) -> float:
        if self._gate_candidate_total is None:
            return 0.0
        return float((self._gate_admitted_total /
                      self._gate_candidate_total.clamp_min(1)).item())

    def _ensure_mechanism_route_index(self) -> None:
        self._refresh_mechanism_route_postings()

    @torch.no_grad()
    def score_pair_proposals(self, sources: Iterable[int],
                             destinations: Iterable[int]) -> Tensor:
        """Score a bounded proposed-pair set with shared learned embeddings."""
        source_ids = tuple(map(int, sources))
        destination_ids = tuple(map(int, destinations))
        if len(source_ids) != len(destination_ids):
            raise ValueError("proposal source and destination arrays must have equal lengths")
        count = self._blocks["entity_embedding"].shape[0]
        if any(value < 0 or value >= count for value in source_ids + destination_ids):
            raise ValueError("proposal endpoint is outside the entity embedding table")
        with self._pinned_forward():
            embedding = self.block("entity_embedding")
            scorer = self.block("scorer").reshape(-1)
            sources_tensor = torch.as_tensor(source_ids, device=embedding.device,
                                              dtype=torch.int64)
            destinations_tensor = torch.as_tensor(destination_ids,
                                                  device=embedding.device,
                                                  dtype=torch.int64)
            left = embedding.index_select(0, sources_tensor)
            right = embedding.index_select(0, destinations_tensor)
            return (left * right * scorer).sum(dim=-1) / (embedding.shape[-1] ** 0.5)

    def score_candidates(self, context: Tensor, candidates: PerCellWorkset, *,
                         temperature: float, threshold: float,
                         prune: bool = False) -> GateValues:
        """Score only the supplied condition/window-eligible derivative pool."""
        if context.ndim != 3 or context.shape[0] != candidates.cell_offsets.numel() - 1:
            raise ValueError("context must have shape [cells, source entities, width]")
        candidates.validate(cells=context.shape[0], edge_count=len(self.support.edge_ids),
                            structure_epoch=self.structure_epoch)
        counts = candidates.cell_offsets[1:] - candidates.cell_offsets[:-1]
        cell_ids = torch.repeat_interleave(
            torch.arange(context.shape[0], device=context.device), counts,
            output_size=candidates.edge_ordinals.numel())
        source_ids = self._source_endpoint_indices.index_select(0, candidates.edge_ordinals)
        destination_ids = self._destination_endpoint_indices.index_select(0, candidates.edge_ordinals)
        entity = self.block("entity_embedding")
        scorer = self.block("scorer").reshape(-1)
        source = context[cell_ids, source_ids]
        destination = entity[destination_ids]
        scores = (source * destination * scorer).sum(dim=-1) / (context.shape[-1] ** 0.5)
        p, admitted, q_train = learned_gates(
            scores, temperature=temperature, threshold=threshold, prune=prune,
            straight_through=True)
        admitted_count = admitted.detach().to(torch.float32).sum()
        candidate_count = admitted_count.new_tensor(float(admitted.numel()))
        self._gate_admitted_total = (admitted_count if self._gate_admitted_total is None
                                     else self._gate_admitted_total + admitted_count)
        self._gate_candidate_total = (candidate_count if self._gate_candidate_total is None
                                      else self._gate_candidate_total + candidate_count)
        self.gate_policy = {
            "temperature": float(temperature), "threshold": float(threshold),
            "estimator": "straight_through" if prune else "soft",
            "derivative_candidate_policy": "admitted_plus_bounded_dormant",
        }
        return GateValues(p, admitted, q_train, candidates.logical_edge_ids,
                          candidates.edge_ordinals, self.structure_epoch,
                          candidates.state_generation)

    def select_gate_worksets(self, candidates: PerCellWorkset, gates: GateValues, *,
                             seed: int, window_counter: Optional[int] = None,
                             dormant_budget: int = 4) -> tuple[PerCellWorkset, PerCellWorkset]:
        """Build forward-admitted and bounded derivative-only CSR worksets.

        Gate decisions are made over the condition/window-eligible candidate
        CSR. The forward CSR retains only actual hard-admitted candidates. The
        derivative CSR retains those same candidates plus a rotating sample
        of closed candidates. Gate values remain indexed in full candidate
        order, so the native VJP scatters only declared derivative items.
        """
        import numpy as np

        candidates.validate(cells=candidates.cell_offsets.numel() - 1,
                            edge_count=len(self.support.edge_ids),
                            structure_epoch=self.structure_epoch)
        if gates.structure_epoch != self.structure_epoch or (
                gates.state_generation != candidates.state_generation):
            raise ValueError("gate decisions do not match candidate structure/state generation")
        offsets = candidates.cell_offsets.detach().cpu().numpy().astype(np.int64, copy=False)
        edge_ids = candidates.logical_edge_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        ordinals = candidates.edge_ordinals.detach().cpu().numpy().astype(np.int64, copy=False)
        admitted_mask = gates.admitted.detach().cpu().numpy().astype(np.bool_, copy=False)
        cell_count = len(offsets) - 1
        if len(edge_ids) != len(ordinals) or len(edge_ids) != len(admitted_mask):
            raise ValueError("candidate IDs, ordinals and gate decisions have different lengths")
        admitted_offsets = [0]
        admitted_ids: list[int] = []
        admitted_ordinals: list[int] = []
        dormant_offsets = [0]
        dormant_ids: list[int] = []
        dormant_ordinals: list[int] = []
        candidate_gate_lookup: list[dict[int, int]] = []
        for cell in range(cell_count):
            begin, end = int(offsets[cell]), int(offsets[cell + 1])
            gate_lookup = {}
            for candidate_index in range(begin, end):
                edge_id = int(edge_ids[candidate_index])
                gate_lookup[edge_id] = candidate_index
                if bool(admitted_mask[candidate_index]):
                    admitted_ids.append(edge_id)
                    admitted_ordinals.append(int(ordinals[candidate_index]))
                else:
                    dormant_ids.append(edge_id)
                    dormant_ordinals.append(int(ordinals[candidate_index]))
            candidate_gate_lookup.append(gate_lookup)
            admitted_offsets.append(len(admitted_ids))
            dormant_offsets.append(len(dormant_ids))
        counter = candidates.state_generation if window_counter is None else int(window_counter)
        native = self._torch_native.build_native_derivative_sample(
            np.ascontiguousarray(admitted_offsets, dtype=np.int64),
            np.ascontiguousarray(admitted_ids, dtype=np.int64),
            np.ascontiguousarray(admitted_ordinals, dtype=np.int64),
            np.ascontiguousarray(dormant_offsets, dtype=np.int64),
            np.ascontiguousarray(dormant_ids, dtype=np.int64),
            np.ascontiguousarray(dormant_ordinals, dtype=np.int64),
            np.full(cell_count, dormant_budget, dtype=np.int64),
            counter, int(seed), self.structure_epoch, candidates.state_generation)
        if (int(native["structure_epoch"]) != self.structure_epoch
                or int(native["state_generation"]) != candidates.state_generation):
            raise RuntimeError("native derivative sampler returned stale workset stamps")
        derivative_offsets = np.asarray(native["derivative_offsets"], dtype=np.int64)
        derivative_ids = np.asarray(native["derivative_edge_ids"], dtype=np.int64)
        derivative_ordinals = np.asarray(native["derivative_edge_ordinals"], dtype=np.int64)
        primal_gate_indices: list[int] = []
        derivative_gate_indices: list[int] = []
        for cell in range(cell_count):
            p_begin, p_end = admitted_offsets[cell], admitted_offsets[cell + 1]
            primal_gate_indices.extend(candidate_gate_lookup[cell][edge_id]
                                       for edge_id in admitted_ids[p_begin:p_end])
            d_begin, d_end = int(derivative_offsets[cell]), int(derivative_offsets[cell + 1])
            derivative_gate_indices.extend(
                candidate_gate_lookup[cell][int(edge_id)]
                for edge_id in derivative_ids[d_begin:d_end])
        device = candidates.cell_offsets.device
        capacity = candidates.capacity
        primal = PerCellWorkset(
            torch.as_tensor(admitted_offsets, device=device, dtype=torch.int64),
            torch.as_tensor(admitted_ordinals, device=device, dtype=torch.int64),
            torch.as_tensor(admitted_ids, device=device, dtype=torch.int64),
            torch.as_tensor(primal_gate_indices, device=device, dtype=torch.int64),
            self.structure_epoch, candidates.state_generation, capacity)
        derivative = PerCellWorkset(
            torch.as_tensor(derivative_offsets, device=device, dtype=torch.int64),
            torch.as_tensor(derivative_ordinals, device=device, dtype=torch.int64),
            torch.as_tensor(derivative_ids, device=device, dtype=torch.int64),
            torch.as_tensor(derivative_gate_indices, device=device, dtype=torch.int64),
            self.structure_epoch, candidates.state_generation, capacity)
        self.gate_policy["derivative_candidate_policy"] = str(native["sampling_policy"])
        self.application_state["derivative_sampler"] = {
            "policy": str(native["sampling_policy"]), "window_counter": counter,
            "seed": int(seed), "eligible_dormant_counts": np.asarray(
                native["eligible_dormant_counts"], dtype=np.int64).tolist(),
            "selected_dormant_counts": np.asarray(
                native["selected_dormant_counts"], dtype=np.int64).tolist(),
            "remaining_dormant_counts": np.asarray(
                native["remaining_dormant_counts"], dtype=np.int64).tolist(),
        }
        return primal, derivative

    def _sparse_checkpoint_state(self):
        # Keep the published base/delta partition and discovery evidence as
        # logical checkpoint data; rebuilding a projection must not erase why
        # an edge entered the support or how it was classified at publication.
        support_snapshot = self._support_index.snapshot()
        return {
            "structure_epoch": self.structure_epoch,
            "support": {
                "edge_ids": self.support.edge_ids,
                "sources": self.support.sources,
                "destinations": self.support.destinations,
            },
            "support_snapshot": support_snapshot,
            "gate_policy": dict(self.gate_policy),
            "application_state": self.application_state,
        }

    def _prepare_sparse_checkpoint_state(self, state):
        if state is None or int(state.get("structure_epoch", -1)) < 0:
            raise ValueError("checkpoint sparse support state is missing or malformed")
        import numpy as np

        saved_snapshot = state.get("support_snapshot")
        if not isinstance(saved_snapshot, dict) or not {
                "structure_epoch", "base", "delta", "revoked_edge_ids",
                "promotion_evidence_edge_ids", "promotion_evidence_round_ids",
                "promotion_evidence_scorer_generations", "promotion_evidence_scores"
        }.issubset(saved_snapshot):
            raise ValueError("checkpoint support base/delta snapshot is missing or malformed")
        if int(saved_snapshot["structure_epoch"]) != int(state["structure_epoch"]):
            raise ValueError("checkpoint support snapshot has a stale structure epoch")
        desired_edges = _snapshot_edges(saved_snapshot)
        saved = state.get("support", {})
        try:
            saved_ids = tuple(map(int, saved["edge_ids"]))
            saved_sources = tuple(map(int, saved["sources"]))
            saved_destinations = tuple(map(int, saved["destinations"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint logical support is malformed") from exc
        if not (len(saved_ids) == len(saved_sources) == len(saved_destinations)
                and tuple(sorted(saved_ids)) == saved_ids
                and len(set(saved_ids)) == len(saved_ids)):
            raise ValueError("checkpoint logical support arrays are inconsistent")
        if desired_edges != dict(zip(saved_ids, zip(saved_sources, saved_destinations))):
            raise ValueError("checkpoint logical support disagrees with its base/delta snapshot")
        current_snapshot = self._support_index.snapshot()
        current_edges = _snapshot_edges(current_snapshot)
        if int(current_snapshot["structure_epoch"]) > int(state["structure_epoch"]):
            raise ValueError("target support epoch is newer than the checkpoint")
        for edge_id in current_edges.keys() & desired_edges.keys():
            if current_edges[edge_id] != desired_edges[edge_id]:
                raise ValueError("checkpoint reuses a logical edge ID for different endpoints")
        current_revoked = set(map(int, current_snapshot["revoked_edge_ids"].tolist()))
        desired_revoked = set(map(int, saved_snapshot["revoked_edge_ids"].tolist()))
        if not desired_revoked.issubset(current_revoked | set(current_edges)):
            raise ValueError("checkpoint contains revoked IDs absent from the target support history")
        if desired_edges.keys() & current_revoked:
            raise ValueError("checkpoint reuses a logical edge ID revoked by the target support history")
        evidence_fields = ("promotion_evidence_edge_ids", "promotion_evidence_round_ids",
                           "promotion_evidence_scorer_generations",
                           "promotion_evidence_scores")
        evidence_sizes = {len(saved_snapshot[key]) for key in evidence_fields}
        if len(evidence_sizes) != 1:
            raise ValueError("checkpoint support promotion evidence arrays disagree")
        evidence_ids = set(map(int, saved_snapshot[evidence_fields[0]].tolist()))
        if not evidence_ids.issubset(desired_edges):
            raise ValueError("checkpoint promotion evidence references inactive support")
        if (current_edges == desired_edges
                and int(current_snapshot["structure_epoch"]) != int(state["structure_epoch"])):
            raise ValueError("checkpoint epoch differs but has no support revision to publish")
        policy = state.get("gate_policy", {})
        if (policy.get("estimator") not in ("soft", "straight_through")
                or float(policy.get("temperature", 0.0)) <= 0
                or not 0 <= float(policy.get("threshold", -1.0)) <= 1):
            raise ValueError("checkpoint gate policy is invalid")
        return state

    def _restore_sparse_checkpoint_state(self, state):
        self._prepare_sparse_checkpoint_state(state)
        desired_edges = _snapshot_edges(state["support_snapshot"])
        current_edges = _snapshot_edges(self._support_index.snapshot())
        evidence = state["support_snapshot"]
        evidence_by_id = {}
        evidence_ids = evidence["promotion_evidence_edge_ids"].tolist()
        evidence_rounds = evidence["promotion_evidence_round_ids"].tolist()
        evidence_generations = evidence["promotion_evidence_scorer_generations"].tolist()
        evidence_scores = evidence["promotion_evidence_scores"].tolist()
        evidence_by_id.update({int(edge_id): (int(round_id), int(generation), float(score))
                               for edge_id, round_id, generation, score in zip(
                                   evidence_ids, evidence_rounds,
                                   evidence_generations, evidence_scores)})
        for edge_id in sorted(current_edges.keys() - desired_edges.keys()):
            self.stage_revocation(edge_id)
        for edge_id in sorted(desired_edges.keys() - current_edges.keys()):
            source, destination = desired_edges[edge_id]
            round_id, generation, score = evidence_by_id.get(edge_id, (0, 0, 0.0))
            self.stage_promotion(edge_id, source, destination,
                                 discovery_round_id=round_id,
                                 scorer_generation=generation,
                                 proposal_score=score)
        if desired_edges != current_edges:
            current_epoch = int(self._support_index.structure_epoch)
            self.publish_support(max(int(state["structure_epoch"]), current_epoch + 1))
            if _snapshot_edges(self._support_index.snapshot()) != desired_edges:
                raise RuntimeError("checkpoint support restoration did not reproduce logical routes")
        self.application_state = dict(state.get("application_state", {}))
        self.gate_policy = dict(state["gate_policy"])


def prepare_sparse_slice(module_type, support: SupportSpec, *, entities: int,
                         mechanisms: int, rank_count: int, device: int,
                         parameter_axis_words: tuple[int, ...], seed: int = 0,
                         reader_capacity: int = 32, ce_geo_image: Optional[bytes] = None,
                         source_group_members: Iterable[int] = (),
                         destination_group_members: Iterable[int] = (),
                         geometry_edge_ids: Optional[Iterable[int]] = None,
                         projection_id: int = 1, max_cells: int = 1,
                         max_primal_items: int = 1):
    """Create a trainable slice from a logical pair support and one native owner.

    This is an explicit preparation boundary: topology and owner allocation
    happen here. Forward calls use the returned module and caller-built worksets.
    """
    import importlib
    import numpy as np

    if module_type not in (RankedGeneSlice, KaryoplastEventSlice):
        raise TypeError("module_type must be RankedGeneSlice or KaryoplastEventSlice")
    if device < 0 or not torch.cuda.is_available():
        raise RuntimeError("sparse trainable slices require the optional CUDA Torch extension")
    if len(support.sources) and max(support.sources + support.destinations) >= entities:
        raise ValueError("support endpoint ordinal exceeds the entity parameter block")
    if len(parameter_axis_words) != 8:
        raise ValueError("parameter_axis_words must contain domain/order/geometry/partition pairs")
    if geometry_edge_ids is not None:
        geometry_edge_ids = tuple(int(value) for value in geometry_edge_ids)
        if len(geometry_edge_ids) != len(support.edge_ids) or any(value < 0 for value in geometry_edge_ids):
            raise ValueError("geometry_edge_ids must align with support order and be nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    blocks, initial = sparse_parameter_layout(
        entities=entities, mechanisms=mechanisms, rank_count=rank_count, width=64)
    # Re-seed the canonical initializer explicitly for reproducible fixtures.
    offset = 0
    for block in blocks:
        count = int(block["numel"])
        initial[offset:offset + count] = torch.randn(count, generator=generator) / (64 ** 0.5)
        offset += count

    native = importlib.import_module("cellerator.torch._torch")
    edge_ids = np.asarray(support.edge_ids, dtype=np.int64)
    participant_offsets = np.arange(len(edge_ids) + 1, dtype=np.int64) * 2
    participant_roles = np.tile(np.asarray([0, 1], dtype=np.int32), len(edge_ids))
    participant_entities = np.asarray(
        [entity + 1 for pair in zip(support.sources, support.destinations)
         for entity in pair], dtype=np.int64)
    relation_id = (int(parameter_axis_words[0]) & 0x7FFF_FFFF_FFFF_FFFF) or 1
    structure_id = (int(parameter_axis_words[4]) & 0x7FFF_FFFF_FFFF_FFFF) or 1
    support_handle = native.NativeSupportHandle(
        edge_ids, participant_offsets, participant_roles, participant_entities,
        relation_id, structure_id, support.structure_epoch, relation_id + 1)
    handle = native.sparse_create_handle(
        edge_ids.tolist(), list(support.sources), list(support.destinations),
        entities, entities, support.structure_epoch, list(parameter_axis_words),
        _stable_scalar_ids(parameter_axis_words, initial.numel()), initial, blocks,
        device, reader_capacity, support_handle)
    module = module_type(handle, support, parameter_blocks={
        entry["name"]: tuple(entry["shape"]) for entry in blocks},
        support_index=support_handle)
    if ce_geo_image is not None:
        native.sparse_prepare_ce_geo_n64(
            handle, ce_geo_image, list(source_group_members),
            list(destination_group_members), projection_id,
            max_cells, max_primal_items,
            [] if geometry_edge_ids is None else list(geometry_edge_ids))
    return module


def compile_sparse_support_ce_geo(support: SupportSpec, *, source_count: int,
                                  destination_count: int,
                                  projection_id: int = 1) -> dict:
    """Compile logical pair support with Cellerator's canonical CE-GEO builder.

    This is explicit preparation work. The returned image and group maps can
    be passed unchanged to :func:`prepare_sparse_slice`; training and inference
    consume the resulting immutable projection.
    """
    import importlib

    native = importlib.import_module("cellerator.torch._torch")
    return native.compile_sparse_support_ce_geo(
        list(support.edge_ids), list(support.sources), list(support.destinations),
        source_count, destination_count, support.structure_epoch, projection_id)


class RankedGeneSlice(SparseSupportModule):
    """Rank-aware embedding consumer; rank is a feature, never an event time."""

    def forward(self, genes: Tensor, ranks: Tensor, primal: PerCellWorkset,
                derivative: PerCellWorkset, gates: Optional[GateValues] = None, *,
                temperature: float = 1.0, threshold: float = 0.5,
                prune: bool = False, derivative_seed: int = 0,
                window_counter: Optional[int] = None) -> Tensor:
        self.reset_gate_diagnostics()
        with self._pinned_forward():
            return self._forward_pinned(genes, ranks, primal, derivative, gates,
                                        temperature=temperature, threshold=threshold,
                                        prune=prune, derivative_seed=derivative_seed,
                                        window_counter=window_counter)

    def _forward_pinned(self, genes: Tensor, ranks: Tensor, primal: PerCellWorkset,
                        derivative: PerCellWorkset, gates: Optional[GateValues] = None, *,
                        temperature: float = 1.0, threshold: float = 0.5,
                        prune: bool = False, derivative_seed: int = 0,
                        window_counter: Optional[int] = None) -> Tensor:
        if genes.shape != ranks.shape or genes.ndim != 2:
            raise ValueError("ranked token input requires matching [cells, selected genes] arrays")
        if genes.dtype not in (torch.int32, torch.int64) or ranks.dtype not in (torch.int32, torch.int64):
            raise ValueError("gene IDs and rank IDs must be integer tensors")
        parameter_embeddings = self.block("entity_embedding")
        genes = genes.to(device=parameter_embeddings.device, dtype=torch.int64)
        ranks = ranks.to(device=parameter_embeddings.device, dtype=torch.int64)
        is_visible = genes >= 0
        gene_embedding = parameter_embeddings[genes.clamp_min(0)]
        rank_embedding = self.block("rank_embedding")[ranks]
        gene_embedding = gene_embedding * is_visible.unsqueeze(-1)
        rank_embedding = rank_embedding * is_visible.unsqueeze(-1)
        batch, selected, width = gene_embedding.shape
        self._ensure_mechanism_route_index()
        affected_offsets, affected_mechanisms = self.affected_mechanism_workset(genes)
        mechanisms = self.block("transition_A").shape[0]
        # Ranked order is a feature, not elapsed time: initialize a shared
        # latent prior from the rank-independent selected set, then make one
        # learned transition with the rank-aware aggregate as input.
        visible_count = is_visible.sum(dim=1).clamp_min(1).to(torch.float32)
        prior = gene_embedding.float().sum(dim=1) / visible_count[:, None]
        state = prior[:, None, :].expand(-1, mechanisms, -1).contiguous()
        transition = self._torch_native.sparse_latent_transition(
            state.to(torch.float16),
            (gene_embedding + rank_embedding).float().sum(dim=1).to(torch.float16),
            affected_offsets, affected_mechanisms,
            self.block("transition_A"), self.block("transition_B"),
            self.block("transition_bias"), self._native)
        latent_context = transition.float().mean(dim=1)
        source_count = self.block("entity_embedding").shape[0]
        source = gene_embedding.new_zeros((batch, source_count, width), dtype=torch.float32)
        source.scatter_add_(1, genes.clamp_min(0).to(torch.int64).unsqueeze(-1).expand(-1, -1, width),
                            (gene_embedding + rank_embedding).float())
        source = source + latent_context[:, None, :]
        if gates is None:
            gates = self.score_candidates(source, derivative, temperature=temperature,
                                          threshold=threshold, prune=prune)
        primal, derivative = self.select_gate_worksets(
            derivative, gates, seed=derivative_seed, window_counter=window_counter)
        edge_values = self.logical_edge_values()
        return self.relation(source.to(torch.float16), edge_values, gates, primal, derivative)

    def masked_gene_logits(self, encoded: Tensor) -> Tensor:
        with self._pinned_forward():
            if encoded.ndim != 3 or encoded.shape[-1] != self.block("entity_embedding").shape[-1]:
                raise ValueError("encoded ranked-gene representation has an invalid shape")
            entity = self.block("entity_embedding").float()
            if encoded.shape[1] != entity.shape[0]:
                raise ValueError("ranked readout requires logical destination rows")
            return (encoded.float() * entity.unsqueeze(0)).sum(dim=-1)

    def training_loss(self, encoded: Tensor, targets: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(self.masked_gene_logits(encoded), targets.long())


class KaryoplastEventSlice(SparseSupportModule):
    """Ordered positive-event consumer preserving repeated occurrences."""

    def forward(self, event_ids: Tensor, event_worksets, *, timestamps: Optional[Tensor] = None,
                windows: Optional[Tensor] = None, temperature: float = 1.0,
                threshold: float = 0.5, prune: bool = False,
                derivative_seed: int = 0) -> Tensor:
        self.reset_gate_diagnostics()
        with self._pinned_forward():
            return self._forward_pinned(event_ids, event_worksets, timestamps=timestamps,
                                        windows=windows, temperature=temperature,
                                        threshold=threshold, prune=prune,
                                        derivative_seed=derivative_seed)

    def _forward_pinned(self, event_ids: Tensor, event_worksets, *,
                        timestamps: Optional[Tensor] = None,
                        windows: Optional[Tensor] = None, temperature: float = 1.0,
                        threshold: float = 0.5, prune: bool = False,
                        derivative_seed: int = 0) -> Tensor:
        if event_ids.ndim != 2 or event_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("event IDs must be integer [cells, occurrences]")
        if timestamps is not None and timestamps.shape != event_ids.shape:
            raise ValueError("timestamps must preserve the event occurrence shape")
        if windows is not None and (windows.ndim != 2 or windows.shape[0] != event_ids.shape[0]):
            raise ValueError("windows must be a per-cell table")
        provider = event_worksets if callable(event_worksets) else None
        if (timestamps is not None or windows is not None) and provider is None:
            raise ValueError("temporal inputs require a provider that window-filters occurrences before gate scoring")
        event_worksets = None if provider is not None else iter(event_worksets)
        embeddings = self.block("entity_embedding")
        transition_a = self.block("transition_A")
        transition_b = self.block("transition_B")
        transition_bias = self.block("transition_bias")
        cells = event_ids.shape[0]
        mechanisms, state_width, _ = transition_a.shape
        state = embeddings.new_zeros((cells, mechanisms, state_width))
        self._ensure_mechanism_route_index()
        previous_state = self.application_state.get("latent_state")
        if previous_state is not None:
            previous_state = torch.as_tensor(previous_state, dtype=torch.float32,
                                             device=embeddings.device)
            if tuple(previous_state.shape) != tuple(state.shape):
                raise ValueError("restored Karyoplast latent state does not match this batch shape")
            state = previous_state
        output = None
        position_offset = int(self.application_state.get("event_position", 0))
        # Each position is a distinct event occurrence, even when gene IDs repeat.
        for position in range(event_ids.shape[1]):
            if provider is not None:
                work = provider(position_offset + position, event_ids[:, :position + 1],
                                None if timestamps is None else timestamps[:, :position + 1],
                                windows)
            else:
                work = next(event_worksets)
            if len(work) not in (4, 5):
                raise ValueError("event work provider must return worksets and per-cell affected mechanisms")
            primal, derivative, affected_offsets, affected_mechanisms = work[:4]
            event = event_ids[:, position].to(device=embeddings.device, dtype=torch.int64)
            event_input = embeddings[event]
            state = self._torch_native.sparse_latent_transition(
                state.to(torch.float16), event_input.to(torch.float16),
                affected_offsets, affected_mechanisms.to(torch.int32).contiguous(),
                transition_a, transition_b, transition_bias, self._native,
            )
            inputs = embeddings.new_zeros((cells, embeddings.shape[0], embeddings.shape[1]))
            inputs.scatter_add_(1, event[:, None, None].expand(-1, 1, embeddings.shape[1]),
                                event_input.unsqueeze(1))
            # Current latent mechanism state conditions candidate scoring while
            # the ordered event occurrence remains the relation input.
            if len(work) == 5:
                history_sources = work[4]
                if len(history_sources) != cells:
                    raise ValueError("event context rows must match the cell batch")
                history_input = torch.zeros_like(inputs)
                for cell, source_row in enumerate(history_sources):
                    if source_row:
                        source_index = torch.as_tensor(
                            source_row, device=embeddings.device, dtype=torch.int64)
                        history_input[cell].index_add_(
                            0, source_index, embeddings.index_select(0, source_index))
                context = history_input + state.float().mean(dim=1, keepdim=True)
            else:
                context = inputs + state.float().mean(dim=1, keepdim=True)
            context = context.expand(-1, embeddings.shape[0], -1).contiguous()
            gates = self.score_candidates(context, derivative, temperature=temperature,
                                          threshold=threshold, prune=prune)
            primal, derivative = self.select_gate_worksets(
                derivative, gates, seed=derivative_seed,
                window_counter=primal.state_generation)
            output = self.relation(inputs.to(torch.float16), self.logical_edge_values(),
                                   gates, primal, derivative)
        self.application_state["latent_state"] = state.detach()
        self.application_state["event_position"] = position_offset + int(event_ids.shape[1])
        if event_ids.shape[1]:
            self.application_state["last_event_ids"] = event_ids[:, -1].detach()
        if timestamps is not None and timestamps.shape[1]:
            self.application_state["last_timestamps"] = timestamps[:, -1].detach()
        if windows is not None:
            self.application_state["windows"] = windows.detach()
        if output is None:
            return self.parameters_flat.new_empty((cells, 0, state_width))
        return output

    def next_event_logits(self, encoded: Tensor) -> Tensor:
        with self._pinned_forward():
            entity = self.block("entity_embedding").float()
            if encoded.shape[1] != entity.shape[0]:
                raise ValueError("event readout requires logical destination rows")
            return (encoded.float() * entity.unsqueeze(0)).sum(dim=-1)

    def training_loss(self, encoded: Tensor, targets: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(self.next_event_logits(encoded), targets.long())

def make_next_event_fixture(*, seed: int = 0, cells: int = 32,
                            steps: int = 12, entities: int = 24):
    """Plant overlapping multi-event mechanisms; hidden rules are validator-only."""
    if min(cells, steps, entities) <= 0 or entities < 20 or steps < 6:
        raise ValueError("fixture requires positive dimensions, entities >= 20, and steps >= 6")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]], dtype=torch.int64)
    target_ids = torch.arange(16, 20, dtype=torch.int64)
    events = torch.empty((cells, steps), dtype=torch.int64)
    mechanism = torch.randint(0, 4, (cells,), generator=generator)
    targets = target_ids[mechanism]
    timestamps = torch.zeros((cells, steps), dtype=torch.int64)
    windows = torch.empty((cells, 2), dtype=torch.int64)
    time_windows = torch.empty((cells, 2), dtype=torch.int64)
    for cell in range(cells):
        chosen = groups[mechanism[cell]].clone()
        chosen = chosen[torch.randperm(3, generator=generator)]
        # Interleave distractors and repeat a participant; the target depends
        # on the complete participant set within the declared recent window.
        prefix = [8 + int(torch.randint(0, 8, (), generator=generator))
                  for _ in range(steps - 6)]
        tail = [int(chosen[0]), 8 + cell % 8, int(chosen[1]),
                int(chosen[2]), int(chosen[0]), 8 + (cell + 3) % 8]
        events[cell] = torch.tensor(prefix + tail)
        intervals = torch.randint(1, 4, (steps,), generator=generator)
        timestamps[cell] = torch.cumsum(intervals, dim=0)
        windows[cell] = torch.tensor([max(0, steps - 6), steps - 1])
        # The independent time prerequisite excludes the first counted
        # occurrence; its repeated participant later in the count window
        # keeps the planted multi-participant rule observable.
        time_windows[cell] = torch.tensor([timestamps[cell, steps - 5], timestamps[cell, steps - 1]])
    hidden_dependencies = torch.cat((groups, target_ids[:, None]), dim=1)
    return {"events": events, "targets": targets, "timestamps": timestamps,
            "count_windows": windows, "time_windows": time_windows,
            "validator_support": hidden_dependencies,
            "validator_mechanism": mechanism, "seed": int(seed)}


def make_masked_gene_fixture(*, seed: int = 0, cells: int = 32,
                             selected: int = 8, entities: int = 32):
    """Plant rank-invariant and rank-dependent modules; rules remain validator-only."""
    if min(cells, selected, entities) <= 0 or selected < 5 or entities < 24:
        raise ValueError("fixture requires selected >= 5 and entities >= 24")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]], dtype=torch.int64)
    genes = torch.empty((cells, selected), dtype=torch.int64)
    ranks = torch.empty_like(genes)
    masked_position = torch.empty((cells,), dtype=torch.int64)
    targets = torch.empty((cells,), dtype=torch.int64)
    rank_dependent = torch.zeros((cells,), dtype=torch.bool)
    for cell in range(cells):
        module = int(torch.randint(0, 4, (), generator=generator))
        context = groups[module].tolist()
        distractor_count = selected - 4
        distractors = torch.randperm(8, generator=generator)[:distractor_count] + 8
        is_rank_dependent = bool(cell % 2)
        rank_dependent[cell] = is_rank_dependent
        context_order = torch.randperm(3, generator=generator).tolist()
        ranked_context = [context[index] for index in context_order]
        target = 16 + module * 2
        visible_context = ranked_context + distractors.tolist()
        target_position = int(torch.randint(0, selected, (), generator=generator))
        first_rank = context_order.index(0)
        second_rank = context_order.index(1)
        if first_rank >= target_position:
            first_rank += 1
        if second_rank >= target_position:
            second_rank += 1
        row = visible_context[:target_position] + [target] + visible_context[target_position:]
        genes[cell] = torch.tensor(row, dtype=torch.int64)
        ranks[cell] = torch.arange(selected)
        masked_position[cell] = target_position
        if is_rank_dependent and first_rank < second_rank:
            target += 1
        genes[cell, target_position] = target
        targets[cell] = target
    visible = genes.clone()
    visible.scatter_(1, masked_position[:, None], -1)
    target_pairs = torch.stack((16 + 2 * torch.arange(4),
                                17 + 2 * torch.arange(4)), dim=1)
    hidden_dependencies = torch.cat((groups, target_pairs), dim=1)
    return {"genes": visible, "ranks": ranks, "masked_position": masked_position,
            "targets": targets, "validator_support": hidden_dependencies,
            "validator_rank_dependent": rank_dependent, "seed": int(seed)}
