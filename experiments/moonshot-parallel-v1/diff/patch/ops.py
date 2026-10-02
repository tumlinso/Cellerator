"""Native FP32 patch and an explicitly declared stored-half STE surrogate.

Stored-half rounds operands and tanh intermediate to binary16. Its derivatives
use saved rounded operands, unrounded tanh(T), and rounded V for dR. They are
surrogate derivatives, not derivatives of the rounding operation.
"""
from dataclasses import dataclass
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
import os
import numpy as np

_LIB = None

def _lib():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_name('native.cc')
        compiler = os.environ.get('CXX', 'c++')
        version = subprocess.check_output([compiler, '--version'])
        key = hashlib.sha256(source.read_bytes() + version + b'-std=c++17 -O2 -ffp-contract=off').hexdigest()
        cache = Path(tempfile.gettempdir()) / 'cellerator-moonshot-patch' / key
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / 'native.so'
        if not binary.exists():
            fd, temporary = tempfile.mkstemp(suffix='.so', dir=cache)
            os.close(fd)
            try:
                subprocess.run([compiler, '-std=c++17', '-O2', '-ffp-contract=off', '-shared', '-fPIC', str(source), '-o', temporary], check=True)
                os.replace(temporary, binary)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        _LIB = ctypes.CDLL(str(binary))
        ptr = ctypes.POINTER(ctypes.c_float)
        for name, count in [('patch_forward', 6), ('patch_output', 3), ('patch_vjp', 9), ('patch_jvp', 9)]:
            fn = getattr(_LIB, name)
            fn.argtypes = [ctypes.c_int] + [ptr] * count
            fn.restype = ctypes.c_int
    return _LIB

def _call(name, *arrays):
    status = getattr(_lib(), name)(arrays[0].shape[0], *(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrays))
    if status:
        raise ValueError(f'native patch admission or allocation failed ({status})')

def _array(value, n=None):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise TypeError('expected a NumPy float32 array')
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] == 0:
        raise ValueError('expected a nonempty square matrix')
    if not value.flags.aligned:
        raise ValueError('expected aligned float32 storage')
    if not value.flags.c_contiguous:
        raise ValueError('expected C-contiguous storage')
    if n is not None and value.shape != (n, n):
        raise ValueError('matrix shapes must match')
    if value.shape[0] > 46340:
        raise ValueError('matrix exceeds native index capacity')
    if not np.isfinite(value).all():
        raise ValueError('nonfinite operand')
    return value

def _generations(values):
    result = tuple(values)
    if len(result) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 for v in result):
        raise ValueError('expected four nonnegative integer generations')
    return tuple(int(v) for v in result)

def _freeze(array):
    # Backing bytes are immutable: callers cannot re-enable ndarray writes.
    return np.frombuffer(array.tobytes(), dtype=np.float32).reshape(array.shape)

@dataclass(frozen=True)
class Tape:
    x: np.ndarray
    l: np.ndarray
    r: np.ndarray
    t: np.ndarray
    v: np.ndarray
    policy: str
    generations: tuple

    def __post_init__(self):
        n = _array(self.x).shape[0]
        if self.policy not in ('fp32', 'stored_half_ste'):
            raise ValueError('unknown precision policy')
        for name in ('x', 'l', 'r', 't', 'v'):
            object.__setattr__(self, name, _freeze(_array(getattr(self, name), n)))
        object.__setattr__(self, 'generations', _generations(self.generations))


def forward(x, l, r, policy='fp32', generations=(0, 0, 0, 0)):
    _array(x)
    n = x.shape[0]
    _array(l, n); _array(r, n)
    if policy not in ('fp32', 'stored_half_ste'):
        raise ValueError('unknown precision policy')
    generation = _generations(generations)
    operands = [a.copy() for a in (x, l, r)]
    if policy == 'stored_half_ste':
        with np.errstate(over='ignore'):
            operands = [a.astype(np.float16).astype(np.float32) for a in operands]
        if any(not np.isfinite(a).all() for a in operands):
            raise ValueError('operand exceeds finite stored-half range')
    sx, sl, sr = operands
    t, v, y = [np.empty_like(x) for _ in range(3)]
    _call('patch_forward', sx, sl, sr, t, v, y)
    if policy == 'stored_half_ste':
        v = v.astype(np.float16).astype(np.float32)
        _call('patch_output', v, sr, y)
    return y, Tape(sx, sl, sr, t, v, policy, generation)


def _check(tape, current_generations):
    if not isinstance(tape, Tape):
        raise TypeError('expected patch Tape')
    if current_generations is not None and _generations(current_generations) != tape.generations:
        raise ValueError('saved-primal generation mismatch')
    n = _array(tape.x).shape[0]
    for name in ('l', 'r', 't', 'v'):
        _array(getattr(tape, name), n)
    if tape.policy not in ('fp32', 'stored_half_ste'):
        raise ValueError('unknown precision policy')
    _generations(tape.generations)
    return n


def vjp(tape, g, current_generations=None):
    n = _check(tape, current_generations)
    _array(g, n)
    dx, dl, dr = [np.empty_like(tape.x) for _ in range(3)]
    _call('patch_vjp', tape.x, tape.l, tape.r, tape.t, tape.v, g, dx, dl, dr)
    return dx, dl, dr


def jvp(tape, dx, dl, dr, current_generations=None):
    n = _check(tape, current_generations)
    for a in (dx, dl, dr):
        _array(a, n)
    dy = np.empty_like(tape.x)
    _call('patch_jvp', tape.x, tape.l, tape.r, tape.t, tape.v, dx, dl, dr, dy)
    return dy
