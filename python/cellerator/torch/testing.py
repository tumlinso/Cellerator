"""Deterministic synthetic fixtures used by validation and examples."""
from __future__ import annotations

import torch


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
