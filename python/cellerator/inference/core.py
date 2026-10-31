"""Inference over supplied biological models and arbitrary model-native state.

The routines here define how candidate states are propagated and weighted. They
do not choose a biological observation model, event vocabulary, or lineage rule;
callers supply likelihoods and copy policies explicitly.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import copy as _copy
import math
import operator
import random
from types import MappingProxyType
from typing import Any

from cellerator.modeling import DynamicalModel, StepContext

StateCopy = Callable[[Any], Any]
LogLikelihood = Callable[[Any, Any], float]


@dataclass(frozen=True, slots=True)
class Observation:
    """A measurement at a model-time coordinate with optional model inputs."""

    time: float
    value: Any
    inputs: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Any = None

    def __post_init__(self) -> None:
        time = float(self.time)
        if not math.isfinite(time):
            raise ValueError("observation time must be finite")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Particle:
    """One candidate realization and its unnormalized posterior log weight."""

    state: Any
    time: float = 0.0
    log_weight: float = 0.0
    lineage: tuple[Any, ...] = ()
    provenance: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time)):
            raise ValueError("particle time must be finite")
        if math.isnan(float(self.log_weight)) or float(self.log_weight) == math.inf:
            raise ValueError("particle log_weight must be finite or negative infinity")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "log_weight", float(self.log_weight))
        object.__setattr__(self, "lineage", tuple(self.lineage))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class Posterior:
    """Normalized particles, effective sample size, and evidence bookkeeping."""

    particles: tuple[Particle, ...]
    weights: tuple[float, ...]
    effective_sample_size: float
    log_evidence: float
    resampled_at: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.particles) != len(self.weights) or not self.particles:
            raise ValueError("posterior needs one weight per nonempty particle")
        if any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("posterior weights must be finite and nonnegative")
        total = math.fsum(self.weights)
        if not math.isclose(total, 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("posterior weights must sum to one")
        if not math.isfinite(self.effective_sample_size) or not math.isfinite(self.log_evidence):
            raise ValueError("posterior diagnostics must be finite")
        object.__setattr__(self, "resampled_at", tuple(self.resampled_at))


def _normalize_log_weights(log_weights: Sequence[float]) -> tuple[tuple[float, ...], float]:
    if not log_weights:
        raise ValueError("at least one particle is required")
    peak = max(log_weights)
    if peak == -math.inf:
        raise ValueError("all candidate particles have zero observation likelihood")
    total = math.fsum(math.exp(weight - peak) for weight in log_weights)
    log_norm = peak + math.log(total)
    return tuple(math.exp(weight - log_norm) for weight in log_weights), log_norm


def systematic_resample(weights: Sequence[float], *, rng: Any = None) -> tuple[int, ...]:
    """Return systematic-resampling indices, preserving the requested count.

    ``rng`` can be :class:`random.Random`, NumPy's Generator, or another object
    exposing ``random()``. The caller owns and can retain the random stream.
    """
    if not weights:
        raise ValueError("weights cannot be empty")
    if any(not math.isfinite(float(w)) or w < 0 for w in weights):
        raise ValueError("weights must be finite and nonnegative")
    total = math.fsum(float(w) for w in weights)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    normalized = [float(w) / total for w in weights]
    cdf = []
    running = 0.0
    for weight in normalized:
        running += weight
        cdf.append(running)
    cdf[-1] = 1.0
    random_source = rng if rng is not None else random
    start = float(random_source.random()) / len(weights)
    indices = []
    cursor = 0
    for k in range(len(weights)):
        point = start + k / len(weights)
        while cursor < len(cdf) - 1 and point > cdf[cursor]:
            cursor += 1
        indices.append(cursor)
    return tuple(indices)


def particle_filter(
    model: DynamicalModel,
    initial: Iterable[Particle | Any],
    observations: Iterable[Observation],
    log_likelihood: LogLikelihood,
    *,
    copy_state: StateCopy = _copy.deepcopy,
    resample: Callable[..., Sequence[int]] | None = systematic_resample,
    resample_below: float | None = 0.5,
    rng: Any = None,
) -> Posterior:
    """Propagate candidate states through observations and update their weights.

    The observation time is the end of the propagation interval. A particle is
    stepped only when the interval is positive, then ``model.observe`` produces
    the predicted measurement. ``log_likelihood(predicted, observed)`` returns
    a scalar natural-log score. State duplication always uses the explicit
    ``copy_state`` policy so external handles, arrays, or immutable values can
    choose their own branching semantics.

    Resampling is systematic by default and occurs when normalized ESS falls
    below ``resample_below * particle_count``. Set ``resample_below=None`` to
    retain weighted particles without resampling. Log evidence accumulates the
    incremental normalizers before each resampling step.
    """
    if not isinstance(model, DynamicalModel):
        raise TypeError("model must be a cellerator.modeling.DynamicalModel")
    if not callable(log_likelihood):
        raise TypeError("log_likelihood must be callable")
    if not callable(copy_state):
        raise TypeError("copy_state must be callable")
    if resample_below is not None and not 0.0 <= resample_below <= 1.0:
        raise ValueError("resample_below must be in [0, 1] or None")

    particles = tuple(
        particle if isinstance(particle, Particle) else Particle(state=particle)
        for particle in initial
    )
    if not particles:
        raise ValueError("at least one initial particle is required")
    particle_weights, _ = _normalize_log_weights(tuple(p.log_weight for p in particles))
    # Start from independent branches according to the declared ownership rule.
    particles = tuple(
        Particle(
            state=copy_state(p.state), time=p.time,
            log_weight=math.log(w) if w > 0 else -math.inf,
            lineage=p.lineage, provenance=p.provenance, metadata=p.metadata,
        )
        for p, w in zip(particles, particle_weights)
    )
    log_evidence = 0.0
    resampled_at: list[float] = []

    for observation_index, observation in enumerate(observations):
        if not isinstance(observation, Observation):
            raise TypeError("observations must contain Observation values")
        advanced: list[Particle] = []
        for particle in particles:
            dt = observation.time - particle.time
            if dt < 0:
                raise ValueError("observation times must not precede particle time")
            context = StepContext(
                time=particle.time, dt=dt, inputs=observation.inputs,
                parameters=model.parameters, rng=rng,
                metadata={**observation.metadata, "observation_index": observation_index},
            )
            state = particle.state
            event_trace = ()
            if dt > 0:
                result = model.step(state, context)
                state = result.state
                event_trace = tuple(result.events)
                if result.time is not None and not math.isclose(
                    result.time, observation.time, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        "model step ended at time "
                        f"{result.time}, before/after observation horizon {observation.time}; "
                        "advance explicitly to the observation time"
                    )
            predicted = model.observe(
                state,
                StepContext(
                    time=observation.time, dt=0.0, inputs=observation.inputs,
                    parameters=model.parameters, rng=rng, metadata=context.metadata,
                ),
            )
            score = float(log_likelihood(predicted, observation.value))
            if math.isnan(score) or score == math.inf:
                raise ValueError("log_likelihood must return a finite value or -inf")
            advanced.append(Particle(
                state=state, time=observation.time,
                log_weight=particle.log_weight + score,
                lineage=particle.lineage,
                provenance=particle.provenance + (("observation", observation.provenance
                    if observation.provenance is not None else observation_index),),
                metadata={**particle.metadata, "last_events": event_trace},
            ))

        weights, increment = _normalize_log_weights(tuple(p.log_weight for p in advanced))
        log_evidence += increment
        ess = 1.0 / math.fsum(weight * weight for weight in weights)
        particles = tuple(
            Particle(
                state=p.state, time=p.time,
                log_weight=math.log(weight) if weight > 0 else -math.inf,
                lineage=p.lineage, provenance=p.provenance, metadata=p.metadata,
            )
            for p, weight in zip(advanced, weights)
        )
        if resample is not None and resample_below is not None and ess < resample_below * len(particles):
            indices = tuple(resample(weights, rng=rng))
            try:
                checked_indices = tuple(operator.index(index) for index in indices)
            except TypeError as exc:
                raise ValueError("resampler indices must be integers") from exc
            if len(indices) != len(particles) or any(
                isinstance(index, bool) or not 0 <= index < len(particles)
                for index in checked_indices
            ):
                raise ValueError("resampler must return one valid particle index per input particle")
            count = len(indices)
            particles = tuple(
                Particle(
                    state=copy_state(particles[index].state), time=particles[index].time,
                    log_weight=-math.log(count),
                    lineage=particles[index].lineage + (index,),
                    provenance=particles[index].provenance + (("resampled_from", index),),
                    metadata=particles[index].metadata,
                )
                for index in checked_indices
            )
            ess = float(count)
            resampled_at.append(observation.time)

    weights, _ = _normalize_log_weights(tuple(p.log_weight for p in particles))
    ess = 1.0 / math.fsum(weight * weight for weight in weights)
    normalized_particles = tuple(
        Particle(state=p.state, time=p.time,
                 log_weight=math.log(weight) if weight > 0 else -math.inf,
                 lineage=p.lineage, provenance=p.provenance, metadata=p.metadata)
        for p, weight in zip(particles, weights)
    )
    return Posterior(normalized_particles, weights, ess, log_evidence, tuple(resampled_at))


@dataclass(frozen=True, slots=True)
class EventHypothesis:
    """Caller-proposed event history and its observation compatibility score."""

    events: tuple[Any, ...]
    log_score: float
    probability: float
    provenance: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.log_score) and self.log_score != -math.inf:
            raise ValueError("log_score must be finite or -inf")
        if not math.isfinite(self.probability) or self.probability < 0:
            raise ValueError("probability must be finite and nonnegative")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def rank_event_hypotheses(
    proposals: Iterable[Iterable[Any]],
    score: Callable[[tuple[Any, ...]], float],
    *,
    provenance: Iterable[Any] | None = None,
    metadata: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[EventHypothesis, ...]:
    """Rank explicitly supplied event histories under a caller's observation score.

    This is a bounded reconstruction primitive: proposal generation and the
    scientific meaning of scores remain with the caller. Returned probabilities
    are normalized compatibility weights, not automatically Bayesian posteriors.
    """
    histories = tuple(tuple(proposal) for proposal in proposals)
    if not histories:
        return ()
    sources = tuple(provenance) if provenance is not None else (None,) * len(histories)
    records = tuple(metadata) if metadata is not None else ({},) * len(histories)
    if len(sources) != len(histories) or len(records) != len(histories):
        raise ValueError("provenance and metadata must align with event proposals")
    scores = tuple(float(score(history)) for history in histories)
    weights, _ = _normalize_log_weights(scores)
    ranked = [EventHypothesis(history, value, weight, source, dict(extra))
              for history, value, weight, source, extra in zip(histories, scores, weights, sources, records)]
    ranked.sort(key=lambda candidate: candidate.log_score, reverse=True)
    return tuple(ranked)
