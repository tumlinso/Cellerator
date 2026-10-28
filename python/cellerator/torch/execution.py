"""Torch execution view over the canonical native sparse parameter owner."""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import time
from typing import Iterable, Optional
import weakref

import torch
from torch import Tensor, nn

from .structure import (GateValues, ParameterBlock, PerCellWorkset, SupportSpec,
                        _snapshot_edges)
from .gating import learned_gates
from .layout import _logical_values_for_ordinals


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

    def logical_edge_values(self, *, policy: str = "all",
                            primal: Optional[PerCellWorkset] = None,
                            derivative: Optional[PerCellWorkset] = None,
                            cells: Optional[int] = None) -> Tensor:
        """Derive shared-parameter edge values, optionally for P∪D only.

        ``workset_union`` still returns the canonical logical E-vector expected
        by the native relation op. Edges outside the supplied primal and
        derivative worksets are zero and receive no value-path gradient. No
        result is cached across parameter publication.
        """
        if policy not in ("all", "workset_union"):
            raise ValueError("edge value policy must be 'all' or 'workset_union'")
        embedding = self.block("entity_embedding")
        scorer = self.block("scorer").reshape(-1)
        if policy == "all":
            # Preserve the original default route's direct gathers and arithmetic.
            source = embedding.index_select(0, self._source_endpoint_indices)
            destination = embedding.index_select(0, self._destination_endpoint_indices)
            return (source * destination * scorer).sum(dim=-1) / (embedding.shape[-1] ** 0.5)
        if primal is None or derivative is None or cells is None:
            raise ValueError("workset_union requires primal, derivative, and cells")
        edge_count = len(self.support.edge_ids)
        primal.validate(cells=cells, edge_count=edge_count,
                        structure_epoch=self.structure_epoch)
        derivative.validate(cells=cells, edge_count=edge_count,
                            structure_epoch=self.structure_epoch)
        ordinals = torch.unique(torch.cat((primal.edge_ordinals,
                                           derivative.edge_ordinals)), sorted=True)
        # Native relation application remains authoritative for ordinal/ID
        # consistency. Avoid a host identity read or device synchronization
        # on the steady forward path.
        return _logical_values_for_ordinals(
            embedding, scorer, self._source_endpoint_indices,
            self._destination_endpoint_indices, ordinals, len(self.support.edge_ids),
            normalize=False)

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
        if state is None or int(state.get("structure_epoch", -1)) <= 0:
            raise ValueError("checkpoint sparse support state is missing or malformed")
        import numpy as np

        if self._read_tickets:
            raise RuntimeError("sparse checkpoint restore requires all parameter read tapes to retire")
        if self._pending_promotions or self._pending_revocations:
            raise RuntimeError("sparse checkpoint restore requires no staged support edits")

        saved_snapshot = state.get("support_snapshot")
        if not isinstance(saved_snapshot, dict) or not {
                "structure_epoch", "base", "delta", "revoked_edge_ids",
                "promotion_evidence_edge_ids", "promotion_evidence_round_ids",
                "promotion_evidence_scorer_generations", "promotion_evidence_scores"
        }.issubset(saved_snapshot):
            raise ValueError("checkpoint support base/delta snapshot is missing or malformed")
        if int(saved_snapshot["structure_epoch"]) != int(state["structure_epoch"]):
            raise ValueError("checkpoint support snapshot has a stale structure epoch")
        current_snapshot = self._support_index.snapshot()
        for name in ("base", "delta"):
            relation = saved_snapshot[name]
            current_relation = current_snapshot[name]
            if (int(relation["structure_epoch"]) != int(state["structure_epoch"])
                    or int(relation["structure_id"]) != int(current_relation["structure_id"])
                    or int(relation["relation_id"]) != int(current_relation["relation_id"])):
                raise ValueError("checkpoint support relation identity differs from the target")
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
        current_edges = _snapshot_edges(current_snapshot)
        if int(current_snapshot["structure_epoch"]) > int(state["structure_epoch"]):
            raise ValueError("target support epoch is newer than the checkpoint")
        for edge_id in current_edges.keys() & desired_edges.keys():
            if current_edges[edge_id] != desired_edges[edge_id]:
                raise ValueError("checkpoint reuses a logical edge ID for different endpoints")
        current_revoked = set(map(int, current_snapshot["revoked_edge_ids"].tolist()))
        revoked_values = tuple(map(int, saved_snapshot["revoked_edge_ids"].tolist()))
        desired_revoked = set(revoked_values)
        if (tuple(sorted(desired_revoked)) != revoked_values
                or any(edge_id <= 0 for edge_id in revoked_values)
                or desired_revoked & desired_edges.keys()):
            raise ValueError("checkpoint tombstones are malformed or overlap active support")
        if desired_edges.keys() & current_revoked:
            raise ValueError("checkpoint reuses a logical edge ID revoked by the target support history")
        current_history = current_edges.keys() | current_revoked
        desired_history = desired_edges.keys() | desired_revoked
        if not current_history.issubset(desired_history):
            raise ValueError("checkpoint would discard target support identity history")
        evidence_fields = ("promotion_evidence_edge_ids", "promotion_evidence_round_ids",
                           "promotion_evidence_scorer_generations",
                           "promotion_evidence_scores")
        evidence_sizes = {len(saved_snapshot[key]) for key in evidence_fields}
        if len(evidence_sizes) != 1:
            raise ValueError("checkpoint support promotion evidence arrays disagree")
        evidence_ids_list = tuple(map(int, saved_snapshot[evidence_fields[0]].tolist()))
        evidence_ids = set(evidence_ids_list)
        evidence_scores = saved_snapshot["promotion_evidence_scores"].tolist()
        if (len(evidence_ids) != len(evidence_ids_list)
                or not evidence_ids.issubset(desired_history)
                or not np.isfinite(np.asarray(evidence_scores, dtype=np.float32)).all()):
            raise ValueError("checkpoint promotion evidence is malformed or references unknown support")
        policy = state.get("gate_policy", {})
        if (policy.get("estimator") not in ("soft", "straight_through")
                or float(policy.get("temperature", 0.0)) <= 0
                or not 0 <= float(policy.get("threshold", -1.0)) <= 1):
            raise ValueError("checkpoint gate policy is invalid")
        return state

    def _restore_sparse_checkpoint_state(self, state):
        self._prepare_sparse_checkpoint_state(state)
        desired_edges = _snapshot_edges(state["support_snapshot"])
        epoch = int(state["structure_epoch"])
        try:
            # Restore the exact logical partition and identity history before
            # rebuilding the numerical view. The existing manager object is
            # retained by both handles, and sparse_rebind_handle retains the
            # one canonical parameter owner and Torch leaf.
            self._support_index.restore_snapshot(state["support_snapshot"])
            self._native = self._torch_native.sparse_rebind_handle(
                self._native, list(desired_edges),
                [desired_edges[edge_id][0] for edge_id in desired_edges],
                [desired_edges[edge_id][1] for edge_id in desired_edges], epoch)
            self.support = SupportSpec(
                tuple(desired_edges),
                tuple(desired_edges[edge_id][0] for edge_id in desired_edges),
                tuple(desired_edges[edge_id][1] for edge_id in desired_edges), epoch)
            self.structure_epoch = epoch
            (self._source_endpoint_indices,
             self._destination_endpoint_indices) = (
                self._torch_native.sparse_endpoint_indices(self._native))
            self._mechanism_route_generation = -1
            self._pending_promotions.clear()
            self._pending_revocations.clear()
            self.application_state = dict(state.get("application_state", {}))
            self.gate_policy = dict(state["gate_policy"])
        except Exception:
            # Restore may have replaced the manager before an allocation or
            # projection rebind fails. Refuse future execution in that case.
            self._poisoned = True
            raise
