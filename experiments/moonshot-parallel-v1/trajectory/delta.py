"""CPU float64 last-transmitted linear delta reference.

The ledger maintains y = W @ x_sent in real arithmetic. Floating point
recurrence can accumulate rounding error. The reported bound covers only the
instantaneous omitted linear contribution, excluding that rounding error;
it is not a trajectory error bound. Publication is one staged state swap,
not a thread synchronization guarantee.
"""

from dataclasses import dataclass
import numpy as np


def _array(value, ndim, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim or 0 in raw.shape:
        raise ValueError(f"{name} must be a nonempty real {ndim}D array")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _generation(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("generation must be a nonnegative integer")
    return int(value)


def _tau(value):
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError("tau must be a finite nonnegative scalar")
    result = float(raw)
    if not np.isfinite(result) or result < 0:
        raise ValueError("tau must be a finite nonnegative scalar")
    return result


@dataclass(frozen=True)
class _State:
    W: np.ndarray
    x_sent: np.ndarray
    y: np.ndarray
    parameter_generation: int
    structure_generation: int


class DeltaLedger:
    """Owned snapshots with thresholds measured against last transmission.

    Pass W to update after changing weights; comparison of actual snapshots
    invalidates the cache even when the caller forgets to change a generation.
    Generation changes also reset the cache when weights compare equal.
    """

    def __init__(self, W, x, tau=0.0, *, parameter_generation=0, structure_generation=0):
        weights = _array(W, 2, "W")
        sent = self._input(x, weights)
        self._tau = _tau(tau)
        pg, sg = _generation(parameter_generation), _generation(structure_generation)
        with np.errstate(over="raise", invalid="raise"):
            y = weights @ sent
        if not np.isfinite(y).all():
            raise ValueError("initial output must be finite")
        self._state = _State(weights, sent, y, pg, sg)

    @staticmethod
    def _input(x, W):
        result = _array(x, 1, "x")
        if result.shape != (W.shape[1],):
            raise ValueError("x extent must match W columns")
        return result

    @property
    def W(self):
        return self._state.W.copy()

    @property
    def x_sent(self):
        return self._state.x_sent.copy()

    @property
    def y(self):
        return self._state.y.copy()

    @property
    def tau(self):
        return self._tau

    @property
    def parameter_generation(self):
        return self._state.parameter_generation

    @property
    def structure_generation(self):
        return self._state.structure_generation

    def update(self, x, W=None, *, parameter_generation=None, structure_generation=None):
        """Return (output, instantaneous omission bound) after atomic staging.

        A changed matrix shape is admitted only with a new structure generation.
        Generations may advance but never roll back. Any validation/arithmetic
        failure leaves all ledger state unchanged.
        """
        old = self._state
        weights = old.W if W is None else _array(W, 2, "W")
        pg = old.parameter_generation if parameter_generation is None else _generation(parameter_generation)
        sg = old.structure_generation if structure_generation is None else _generation(structure_generation)
        if pg < old.parameter_generation or sg < old.structure_generation:
            raise ValueError("generation cannot roll back")
        if weights.shape != old.W.shape and sg == old.structure_generation:
            raise ValueError("shape change requires a new structure generation")
        current = self._input(x, weights)
        reset = pg != old.parameter_generation or sg != old.structure_generation or not np.array_equal(weights, old.W)
        with np.errstate(over="raise", invalid="raise"):
            if reset:
                sent = current.copy()
                output = weights @ sent
            else:
                delta = current - old.x_sent
                changed = np.abs(delta) > self._tau
                sent = old.x_sent.copy()
                sent[changed] = current[changed]
                output = old.y + weights[:, changed] @ delta[changed]
            bound = np.abs(weights) @ np.abs(current - sent)
        if not np.isfinite(output).all() or not np.isfinite(bound).all():
            raise ValueError("output and bound must be finite")
        # Allocate return values too before committing, so failed staging cannot
        # leave a partially advanced transmitted vector or generation.
        result = output.copy(), bound.copy()
        self._state = _State(weights, sent, output, pg, sg)
        return result

    def discrepancy_bound(self, x):
        state = self._state
        current = self._input(x, state.W)
        with np.errstate(over="raise", invalid="raise"):
            bound = np.abs(state.W) @ np.abs(current - state.x_sent)
        if not np.isfinite(bound).all():
            raise ValueError("bound must be finite")
        return bound

    def checkpoint(self):
        """Return an independent JSON-compatible float64 recurrence snapshot."""
        state = self._state
        return dict(schema_version=1, dtype="float64", tau=self._tau,
                    W=state.W.tolist(), x_sent=state.x_sent.tolist(), y=state.y.tolist(),
                    parameter_generation=state.parameter_generation,
                    structure_generation=state.structure_generation)

    @classmethod
    def from_checkpoint(cls, payload):
        expected = {"schema_version", "dtype", "tau", "W", "x_sent", "y",
                    "parameter_generation", "structure_generation"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("invalid checkpoint fields")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1 or payload["dtype"] != "float64":
            raise ValueError("unsupported checkpoint schema or dtype")
        weights = _array(payload["W"], 2, "W")
        sent = cls._input(payload["x_sent"], weights)
        y = _array(payload["y"], 1, "y")
        if y.shape != (weights.shape[0],):
            raise ValueError("checkpoint output extent must match W rows")
        # Preserve recurrence output exactly; recomputation would erase drift
        # and change continuation. Integrity/authentication belongs to storage.
        result = cls.__new__(cls)
        result._tau = _tau(payload["tau"])
        result._state = _State(weights, sent, y, _generation(payload["parameter_generation"]),
                               _generation(payload["structure_generation"]))
        return result
