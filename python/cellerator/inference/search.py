"""Bounded search over caller-proposed event histories.

This module searches a supplied proposal space. It does not discover event
semantics, fit a dynamics law, or interpret normalized compatibility weights as
a Bayesian posterior. The caller owns proposals, event application, likelihoods,
and the event/state copy policy.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import copy as _copy
import itertools
import math
from types import MappingProxyType
from typing import Any

from cellerator.modeling import Model, StepContext, StepResult

from .core import Observation, Particle, _normalize_log_weights


@dataclass(frozen=True, slots=True)
class TimedEvent:
    """A proposed event at a time in model time units."""

    event: Any
    time: float
    provenance: Any = None

    def __post_init__(self) -> None:
        value = float(self.time)
        if not math.isfinite(value):
            raise ValueError("proposed event time must be finite")
        object.__setattr__(self, "time", value)


@dataclass(frozen=True, slots=True)
class EventPath:
    """One alternative sequence of events proposed for an observation interval."""

    events: tuple[TimedEvent, ...] = ()
    log_prior_score: float = 0.0
    provenance: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.log_prior_score)
        if math.isnan(score) or score == math.inf:
            raise ValueError("event path score must be finite or negative infinity")
        values = tuple(self.events)
        if not all(isinstance(event, TimedEvent) for event in values):
            raise TypeError("EventPath.events must contain TimedEvent values")
        if any(right.time < left.time for left, right in zip(values, values[1:])):
            raise ValueError("events in an EventPath must be ordered by time")
        object.__setattr__(self, "events", values)
        object.__setattr__(self, "log_prior_score", score)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SearchHypothesis:
    """One retained history and its score and within-beam compatibility weight."""

    state: Any
    time: float
    events: tuple[TimedEvent, ...]
    log_score: float
    compatibility_weight: float
    provenance: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class EventSearchResult:
    """Retained candidate histories and explicit search-bound diagnostics.

    ``compatibility_weight`` is normalized among the returned hypotheses only.
    Beam or resource pruning can omit candidates, so it is not a posterior over
    the complete model space and must not be interpreted as one.
    """

    hypotheses: tuple[SearchHypothesis, ...]
    expanded: int
    pruned: int
    complete: bool
    truncation_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expanded < 0 or self.pruned < 0:
            raise ValueError("search counts must be nonnegative")
        if self.hypotheses:
            weights = tuple(item.compatibility_weight for item in self.hypotheses)
            if any(not math.isfinite(weight) or weight < 0 for weight in weights):
                raise ValueError("compatibility weights must be finite and nonnegative")
            if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("compatibility weights must sum to one")
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "truncation_reasons", tuple(self.truncation_reasons))


@dataclass(slots=True)
class _Node:
    state: Any
    time: float
    events: tuple[TimedEvent, ...]
    log_score: float
    provenance: tuple[Any, ...]
    metadata: Mapping[str, Any]
    order: int


Proposal = Callable[[Any, StepContext, Observation], Iterable[EventPath]]
EventApply = Callable[[Any, Any, StepContext], Any]


def _checked_score(value: Any, name: str) -> float:
    score = float(value)
    if math.isnan(score) or score == math.inf:
        raise ValueError(f"{name} must return a finite score or -inf")
    return score


def _advance(model: Model, state: Any, start: float, end: float,
             observation: Observation, observation_index: int) -> Any:
    """Advance exactly to ``end`` or reject an explicitly early/late step."""
    dt = end - start
    if dt < 0:
        raise ValueError("proposed events must not precede the current state time")
    if dt == 0:
        return state
    context = StepContext(
        time=start, dt=dt, inputs=observation.inputs, parameters=model.parameters,
        metadata={**observation.metadata, "observation_index": observation_index},
    )
    result = model.step(state, context)
    if result.time is not None and not math.isclose(
        result.time, end, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(
            f"model step ended at {result.time}, expected {end}; "
            "advance explicitly to the requested event or observation time"
        )
    return result.state


def _apply_event(apply_event: EventApply, state: Any, timed: TimedEvent,
                 *, start: float, observation: Observation,
                 observation_index: int, event_index: int, model: Model) -> Any:
    context = StepContext(
        time=timed.time, dt=timed.time - start, inputs=observation.inputs,
        parameters=model.parameters,
        metadata={**observation.metadata, "observation_index": observation_index,
                  "event_index": event_index, "event_provenance": timed.provenance},
    )
    result = apply_event(state, timed.event, context)
    if isinstance(result, StepResult):
        if result.time is not None and not math.isclose(
            result.time, timed.time, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"event update ended at {result.time}, expected event time {timed.time}"
            )
        return result.state
    return result


def search_event_histories(
    model: Model,
    initial: Iterable[Particle | Any],
    observations: Iterable[Observation],
    propose: Proposal,
    apply_event: EventApply,
    log_likelihood: Callable[[Any, Any], float],
    *,
    beam_width: int = 128,
    max_expansions: int = 10_000,
    max_proposals_per_hypothesis: int = 128,
    max_events_per_path: int = 64,
    copy_state: Callable[[Any], Any] = _copy.deepcopy,
) -> EventSearchResult:
    """Search event histories proposed for partially observed trajectories.

    At each observation interval ``propose(state, context, observation)`` yields
    ``EventPath`` alternatives; yield an empty path to represent no event. Each
    path is applied in timestamp order, with supplied model dynamics advancing
    state between its events and to the observation time. Observation scores
    and path prior scores accumulate additively in log space. Distinct histories
    are retained even when their states or scores agree.

    Stable tie order follows initial particle order, proposal order, then prior
    path order. Beam pruning ranks by descending score and never merges distinct
    histories. Reaching any declared bound sets ``complete=False`` and records
    the reason; this makes returned weights visibly conditional on the retained
    beam. Invalid proposals or model-time mismatches raise ``ValueError``.
    """
    if not isinstance(model, Model):
        raise TypeError("model must be a cellerator.modeling.Model")
    if not callable(propose) or not callable(apply_event) or not callable(log_likelihood):
        raise TypeError("propose, apply_event and log_likelihood must be callable")
    for name, value in (("beam_width", beam_width), ("max_expansions", max_expansions),
                        ("max_proposals_per_hypothesis", max_proposals_per_hypothesis),
                        ("max_events_per_path", max_events_per_path)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    seeds = tuple(value if isinstance(value, Particle) else Particle(value) for value in initial)
    if not seeds:
        raise ValueError("at least one initial state is required")
    seed_weights, _ = _normalize_log_weights(tuple(seed.log_weight for seed in seeds))
    nodes = [
        _Node(copy_state(seed.state), seed.time, (), math.log(weight) if weight else -math.inf,
              seed.provenance, seed.metadata, index)
        for index, (seed, weight) in enumerate(zip(seeds, seed_weights))
    ]
    expanded = 0
    pruned = 0
    next_order = len(nodes)
    reasons: list[str] = []
    beam_pruned = max(0, len(nodes) - beam_width)
    if beam_pruned:
        pruned += beam_pruned
        reasons.append("beam_width")
        nodes = sorted(nodes, key=lambda node: (-node.log_score, node.order))[:beam_width]

    for observation_index, observation in enumerate(observations):
        if not isinstance(observation, Observation):
            raise TypeError("observations must contain Observation values")
        candidates: list[_Node] = []
        limit_hit = False
        proposal_limit_hit = False
        for node_index, node in enumerate(nodes):
            if observation.time < node.time:
                raise ValueError("observations must be ordered after particle time")
            context = StepContext(
                time=node.time, dt=observation.time - node.time,
                inputs=observation.inputs, parameters=model.parameters,
                metadata={**observation.metadata, "observation_index": observation_index},
            )
            bounded = itertools.islice(
                iter(propose(node.state, context, observation)),
                max_proposals_per_hypothesis + 1,
            )
            paths = tuple(bounded)
            if len(paths) > max_proposals_per_hypothesis:
                paths = paths[:max_proposals_per_hypothesis]
                proposal_limit_hit = True
                pruned += 1
            for path_index, path in enumerate(paths):
                if not isinstance(path, EventPath):
                    raise TypeError("proposal function must yield EventPath values")
                if len(path.events) > max_events_per_path:
                    proposal_limit_hit = True
                    pruned += 1
                    continue
                if any(event.time < node.time or event.time > observation.time for event in path.events):
                    raise ValueError("proposed event times must fall within their observation interval")
                if expanded >= max_expansions:
                    limit_hit = True
                    break
                state = copy_state(node.state)
                current_time = node.time
                for event_index, timed in enumerate(path.events):
                    state = _advance(model, state, current_time, timed.time,
                                     observation, observation_index)
                    state = _apply_event(
                        apply_event, state, timed, start=current_time,
                        observation=observation, observation_index=observation_index,
                        event_index=event_index, model=model,
                    )
                    current_time = timed.time
                state = _advance(model, state, current_time, observation.time,
                                 observation, observation_index)
                predicted = model.observe(
                    state,
                    StepContext(
                        time=observation.time, dt=0.0, inputs=observation.inputs,
                        parameters=model.parameters, metadata=context.metadata,
                    ),
                )
                observation_score = _checked_score(
                    log_likelihood(predicted, observation.value), "log_likelihood"
                )
                score = node.log_score + path.log_prior_score + observation_score
                if score != -math.inf:
                    candidates.append(_Node(
                        state, observation.time, node.events + path.events, score,
                        node.provenance + ((
                            "event_path", path.provenance,
                            tuple(event.provenance for event in path.events),
                        ), ("observation", observation.provenance
                            if observation.provenance is not None else observation_index)),
                        {**node.metadata, **path.metadata,
                         "last_observation": observation.provenance if observation.provenance is not None
                         else observation_index},
                        next_order,
                    ))
                    next_order += 1
                expanded += 1
            if limit_hit:
                # A remaining path or beam item means the expansion budget hid work.
                if path_index + 1 < len(paths) or node_index + 1 < len(nodes):
                    break
                # A proposal overflow also signals omitted work.
                break
        if proposal_limit_hit:
            reasons.append("proposal_limit")
        if limit_hit:
            reasons.append("max_expansions")
        if not candidates:
            nodes = []
            break
        candidates.sort(key=lambda node: (-node.log_score, node.order))
        if len(candidates) > beam_width:
            pruned += len(candidates) - beam_width
            if "beam_width" not in reasons:
                reasons.append("beam_width")
            candidates = candidates[:beam_width]
        nodes = candidates

    if nodes:
        weights, _ = _normalize_log_weights(tuple(node.log_score for node in nodes))
        hypotheses = tuple(
            SearchHypothesis(
                state=node.state, time=node.time, events=node.events,
                log_score=node.log_score, compatibility_weight=weight,
                provenance=node.provenance, metadata=node.metadata,
            )
            for node, weight in zip(nodes, weights)
        )
    else:
        hypotheses = ()
    # Keep reasons deterministic and unique while preserving their first occurrence.
    reasons = list(dict.fromkeys(reasons))
    return EventSearchResult(hypotheses, expanded, pruned, not reasons, tuple(reasons))
