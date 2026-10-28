"""Model-to-mechanism consumers and explicit native preparation routes."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor

from .structure import (GateValues, PerCellWorkset, SupportSpec,
                        _stable_scalar_ids)
from .execution import SparseSupportModule
from .geometry import compile_sparse_support_ce_geo


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


class RankedGeneSlice(SparseSupportModule):
    """Rank-aware embedding consumer; rank is a feature, never an event time."""

    def forward(self, genes: Tensor, ranks: Tensor, primal: PerCellWorkset,
                derivative: PerCellWorkset, gates: Optional[GateValues] = None, *,
                temperature: float = 1.0, threshold: float = 0.5,
                prune: bool = False, derivative_seed: int = 0,
                window_counter: Optional[int] = None,
                edge_value_policy: str = "all") -> Tensor:
        self.reset_gate_diagnostics()
        with self._pinned_forward():
            return self._forward_pinned(genes, ranks, primal, derivative, gates,
                                        temperature=temperature, threshold=threshold,
                                        prune=prune, derivative_seed=derivative_seed,
                                        window_counter=window_counter,
                                        edge_value_policy=edge_value_policy)

    def _forward_pinned(self, genes: Tensor, ranks: Tensor, primal: PerCellWorkset,
                        derivative: PerCellWorkset, gates: Optional[GateValues] = None, *,
                        temperature: float = 1.0, threshold: float = 0.5,
                        prune: bool = False, derivative_seed: int = 0,
                        window_counter: Optional[int] = None,
                        edge_value_policy: str = "all") -> Tensor:
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
        edge_values = self.logical_edge_values(
            policy=edge_value_policy, primal=primal, derivative=derivative,
            cells=batch)
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


def _sum_history_sources(embeddings: Tensor, history_sources, *, cells: int) -> Tensor:
    """Sum eligible source occurrences per cell, retaining repeated IDs."""
    entities, width = embeddings.shape
    if len(history_sources) != cells:
        raise ValueError("event context rows must match the cell batch")
    history_input = embeddings.new_zeros((cells, entities, width))
    for cell, source_row in enumerate(history_sources):
        source_indices = torch.as_tensor(source_row, device=embeddings.device,
                                         dtype=torch.int64).reshape(-1)
        if source_indices.numel() == 0:
            continue
        history_input[cell].index_add_(
            0, source_indices, embeddings.index_select(0, source_indices))
    return history_input


def _validate_relation_input_policy(application_state: dict[str, object],
                                    requested: str) -> None:
    if requested not in ("current_event", "eligible_history"):
        raise ValueError("relation_input must be 'current_event' or 'eligible_history'")
    position = int(application_state.get("event_position", 0))
    previous = application_state.get("relation_input_policy")
    if position and previous is None:
        # Pre-policy checkpoints used current-event relation inputs.
        previous = "current_event"
    if position and previous != requested:
        raise ValueError("cannot change relation_input policy during a live event stream")


def _relation_source_inputs(current_inputs: Tensor, history_inputs: Optional[Tensor],
                           policy: str) -> Tensor:
    if policy == "current_event":
        return current_inputs
    if policy == "eligible_history":
        if history_inputs is None:
            raise ValueError(
                "eligible_history relation input requires provider work[4] history rows")
        return history_inputs
    raise ValueError("relation_input must be 'current_event' or 'eligible_history'")


class KaryoplastEventSlice(SparseSupportModule):
    """Ordered positive-event consumer preserving repeated occurrences."""

    def forward(self, event_ids: Tensor, event_worksets, *, timestamps: Optional[Tensor] = None,
                windows: Optional[Tensor] = None, temperature: float = 1.0,
                threshold: float = 0.5, prune: bool = False,
                derivative_seed: int = 0,
                relation_input: str = "current_event",
                edge_value_policy: str = "all") -> Tensor:
        self.reset_gate_diagnostics()
        with self._pinned_forward():
            return self._forward_pinned(event_ids, event_worksets, timestamps=timestamps,
                                        windows=windows, temperature=temperature,
                                        threshold=threshold, prune=prune,
                                        derivative_seed=derivative_seed,
                                        relation_input=relation_input,
                                        edge_value_policy=edge_value_policy)

    def _forward_pinned(self, event_ids: Tensor, event_worksets, *,
                        timestamps: Optional[Tensor] = None,
                        windows: Optional[Tensor] = None, temperature: float = 1.0,
                        threshold: float = 0.5, prune: bool = False,
                        derivative_seed: int = 0,
                        relation_input: str = "current_event",
                        edge_value_policy: str = "all") -> Tensor:
        if event_ids.ndim != 2 or event_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("event IDs must be integer [cells, occurrences]")
        _validate_relation_input_policy(self.application_state, relation_input)
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
        position_offset = int(self.application_state.get("event_position", 0))
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
            # Current latent mechanism state conditions candidate scoring.
            history_input = None
            if len(work) == 5:
                history_sources = work[4]
                history_input = _sum_history_sources(embeddings, history_sources,
                                                     cells=cells)
                context = history_input + state.float().mean(dim=1, keepdim=True)
            else:
                context = inputs + state.float().mean(dim=1, keepdim=True)
            context = context.expand(-1, embeddings.shape[0], -1).contiguous()
            gates = self.score_candidates(context, derivative, temperature=temperature,
                                          threshold=threshold, prune=prune)
            primal, derivative = self.select_gate_worksets(
                derivative, gates, seed=derivative_seed,
                window_counter=primal.state_generation)
            relation_sources = _relation_source_inputs(inputs, history_input, relation_input)
            output = self.relation(
                relation_sources.to(torch.float16),
                self.logical_edge_values(policy=edge_value_policy, primal=primal,
                                         derivative=derivative, cells=cells),
                gates, primal, derivative)
        self.application_state["latent_state"] = state.detach()
        self.application_state["event_position"] = position_offset + int(event_ids.shape[1])
        if event_ids.shape[1]:
            self.application_state["relation_input_policy"] = relation_input
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
