"""CPU Torch witness for supplied linear rewrites; no native concurrency contract."""
from dataclasses import dataclass
from numbers import Integral
import torch


@dataclass(frozen=True)
class Coordinate:
    identity: str
    incarnation: int


@dataclass(frozen=True)
class Snapshot:
    coordinates: tuple
    state: torch.Tensor
    law: torch.Tensor
    readout: torch.Tensor
    structure_epoch: int = 0
    value_generation: int = 0
    replica_maps: tuple = ()


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _clone(snapshot):
    return Snapshot(tuple(snapshot.coordinates), snapshot.state.detach().clone(),
                    snapshot.law.detach().clone(), snapshot.readout.detach().clone(),
                    snapshot.structure_epoch, snapshot.value_generation,
                    tuple(tuple(m) for m in snapshot.replica_maps))


def _validate(snapshot):
    if not isinstance(snapshot, Snapshot):
        raise ValueError('expected Snapshot')
    if not _integer(snapshot.structure_epoch) or not _integer(snapshot.value_generation):
        raise ValueError('epochs must be nonnegative integers')
    coords = snapshot.coordinates
    if not coords or any(not isinstance(c, Coordinate) or not isinstance(c.identity, str)
                         or not c.identity or not _integer(c.incarnation) for c in coords):
        raise ValueError('invalid coordinate identity/incarnation')
    if len({c.identity for c in coords}) != len(coords):
        raise ValueError('coordinate IDs must be unique in declared order')
    n = len(coords)
    tensors = (snapshot.state, snapshot.law, snapshot.readout)
    if any(not isinstance(t, torch.Tensor) or t.device.type != 'cpu' or
           t.dtype not in (torch.float32, torch.float64) or not torch.isfinite(t).all()
           for t in tensors):
        raise ValueError('finite CPU float32/float64 tensors required')
    if len({t.dtype for t in tensors}) != 1:
        raise ValueError('tensor dtypes must agree')
    if snapshot.state.shape != (n,) or snapshot.law.shape != (n, n):
        raise ValueError('state/law shape disagrees with coordinates')
    if snapshot.readout.ndim != 2 or snapshot.readout.shape[1] != n or not snapshot.readout.shape[0]:
        raise ValueError('readout must have shape (outputs, coordinates)')
    for mapping in snapshot.replica_maps:
        if not mapping or any(not _integer(i) or i >= n for i in mapping):
            raise ValueError('invalid physical replica gather map')
    return _clone(snapshot)


_TAPE_ADMISSION = object()


class SavedTape:
    """An actual autograd graph pinned until backward/release, with explicit lifetime."""
    def __init__(self, owner, steps, *, _admission=None):
        if _admission is not _TAPE_ADMISSION:
            raise TypeError('saved tapes must be acquired through SnapshotController.acquire_tape')
        if not isinstance(owner, SnapshotController) or not _integer(steps):
            raise ValueError('invalid saved tape acquisition')
        self._owner = owner
        snap = owner._snapshot
        self.structure_epoch = snap.structure_epoch
        self.coordinates = snap.coordinates
        self._leaves = {name: getattr(snap, name).clone().requires_grad_()
                        for name in ('state', 'law', 'readout')}
        state = self._leaves['state']
        for _ in range(steps):
            state = self._leaves['law'] @ state
        self._output = self._leaves['readout'] @ state
        self._released = False
        self.gradients = None
        # Register only after every graph-building operation has succeeded.
        owner._tapes.add(self)

    @property
    def output(self):
        self._require_live()
        return self._output.clone()

    def _require_live(self):
        if self._released:
            raise RuntimeError('saved tape already released')
        snap = self._owner._snapshot
        if snap.structure_epoch != self.structure_epoch or snap.coordinates != self.coordinates:
            raise RuntimeError('stale saved tape identity')

    def backward(self, adjoint=None):
        self._require_live()
        if adjoint is None:
            adjoint = torch.ones_like(self._output)
        if not isinstance(adjoint, torch.Tensor) or adjoint.shape != self._output.shape or not torch.isfinite(adjoint).all():
            raise ValueError('invalid readout adjoint')
        grads = torch.autograd.grad(self._output, tuple(self._leaves.values()),
                                    grad_outputs=adjoint, allow_unused=True)
        self.gradients = {name: (torch.zeros_like(leaf) if grad is None else grad.detach().clone())
                          for (name, leaf), grad in zip(self._leaves.items(), grads)}
        self.release()
        return {name: grad.clone() for name, grad in self.gradients.items()}

    def release(self):
        if not self._released:
            self._released = True
            self._owner._tapes.remove(self)
            self._output = None
            self._leaves = None

    def __enter__(self):
        self._require_live()
        return self

    def __exit__(self, *_):
        self.release()


class SnapshotController:
    """Same-thread controller; callers must externally synchronize any concurrent access."""
    SCHEMA_VERSION = 1
    CHECKPOINT_KEYS = frozenset(('schema_version', 'coordinates', 'state', 'law', 'readout',
                                 'structure_epoch', 'value_generation', 'replica_maps'))

    def __init__(self, snapshot):
        try:
            self._snapshot = _validate(snapshot)
        except (TypeError, AttributeError, RuntimeError) as exc:
            raise ValueError('malformed snapshot') from exc
        self._tapes = set()

    @property
    def snapshot(self):
        return _clone(self._snapshot)

    def assert_drained(self):
        if self._tapes:
            raise RuntimeError('drain all saved tapes before publication or restore')

    def acquire_tape(self, steps=1):
        if not _integer(steps):
            raise ValueError('steps must be a nonnegative integer')
        return SavedTape(self, steps, _admission=_TAPE_ADMISSION)

    def _guard(self, expected_epoch, expected_coordinates):
        self.assert_drained()
        if not _integer(expected_epoch) or expected_epoch != self._snapshot.structure_epoch or tuple(expected_coordinates) != self._snapshot.coordinates:
            raise RuntimeError('stale structure epoch or ordered incarnation identity')

    def _matrix(self, matrix, shape):
        if not isinstance(matrix, torch.Tensor) or matrix.shape != shape or matrix.device.type != 'cpu' or matrix.dtype != self._snapshot.state.dtype or not torch.isfinite(matrix).all():
            raise ValueError('invalid supplied coordinate map')
        return matrix.detach().clone()

    def _publish(self, candidate):
        try:
            candidate = _validate(candidate)
        except (TypeError, AttributeError, RuntimeError) as exc:
            raise ValueError('malformed candidate snapshot') from exc
        # Validation is complete before the sole publication assignment.
        self._snapshot = candidate
        return self.snapshot

    def publish_transform(self, transform, new_coordinates, *, expected_epoch,
                          expected_coordinates, replica_maps=()):
        self._guard(expected_epoch, expected_coordinates)
        old = self._snapshot
        n = len(old.coordinates)
        transform = self._matrix(transform, (n, n))
        try:
            inverse = torch.linalg.inv(transform)
        except RuntimeError as exc:
            raise ValueError('coordinate transform must be invertible') from exc
        return self._publish(Snapshot(tuple(new_coordinates), transform @ old.state,
            transform @ old.law @ inverse, old.readout @ inverse,
            old.structure_epoch + 1, old.value_generation + 1, replica_maps))

    def publish_quotient(self, projection, embedding, new_coordinates, *, expected_epoch,
                         expected_coordinates, replica_maps=()):
        self._guard(expected_epoch, expected_coordinates)
        old = self._snapshot
        n, k = len(old.coordinates), len(new_coordinates)
        p, e = self._matrix(projection, (k, n)), self._matrix(embedding, (n, k))
        law = p @ old.law @ e
        def close(a, b):
            return torch.allclose(a, b, rtol=1e-6 if a.dtype == torch.float32 else 1e-10,
                                  atol=1e-6 if a.dtype == torch.float32 else 1e-12)
        if not close(p @ e, torch.eye(k, dtype=e.dtype)):
            raise ValueError('quotient requires P E = I')
        if not close(old.law @ e, e @ law):
            raise ValueError('quotient manifold is not invariant under the declared law')
        if not close(old.state, e @ (p @ old.state)):
            raise ValueError('state is off the quotient manifold')
        return self._publish(Snapshot(tuple(new_coordinates), p @ old.state, law,
            old.readout @ e, old.structure_epoch + 1, old.value_generation + 1, replica_maps))

    def replica_values(self):
        return tuple(self._snapshot.state[list(m)].clone() for m in self._snapshot.replica_maps)

    def replica_adjoint(self, adjoints):
        if len(adjoints) != len(self._snapshot.replica_maps):
            raise ValueError('one adjoint per physical replica map required')
        result = torch.zeros_like(self._snapshot.state)
        for mapping, adjoint in zip(self._snapshot.replica_maps, adjoints):
            if not isinstance(adjoint, torch.Tensor) or adjoint.shape != (len(mapping),) or adjoint.dtype != result.dtype or adjoint.device.type != 'cpu' or not torch.isfinite(adjoint).all():
                raise ValueError('invalid physical replica adjoint')
            result.index_add_(0, torch.tensor(mapping, dtype=torch.long), adjoint)
        return result

    def export_checkpoint(self):
        snap = self.snapshot
        return dict(schema_version=self.SCHEMA_VERSION,
                    coordinates=tuple((c.identity, c.incarnation) for c in snap.coordinates),
                    state=snap.state, law=snap.law, readout=snap.readout,
                    structure_epoch=snap.structure_epoch, value_generation=snap.value_generation,
                    replica_maps=snap.replica_maps)

    @classmethod
    def from_checkpoint(cls, payload):
        if not isinstance(payload, dict) or set(payload) != cls.CHECKPOINT_KEYS or type(payload['schema_version']) is not int or payload['schema_version'] != cls.SCHEMA_VERSION:
            raise ValueError('unsupported or incomplete snapshot checkpoint schema')
        try:
            coordinates = tuple(Coordinate(*pair) for pair in payload['coordinates'])
            snapshot = Snapshot(coordinates, payload['state'], payload['law'], payload['readout'],
                payload['structure_epoch'], payload['value_generation'], payload['replica_maps'])
            return cls(snapshot)
        except (TypeError, KeyError) as exc:
            raise ValueError('malformed snapshot checkpoint') from exc

    def restore_checkpoint(self, payload, *, expected_epoch, expected_coordinates):
        self._guard(expected_epoch, expected_coordinates)
        candidate = self.from_checkpoint(payload)._snapshot
        self._snapshot = candidate
        return self.snapshot
