"""CPU float32 process packets, differentiated using saved float32 operands."""
from dataclasses import dataclass
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

_LIB = None


def _library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_name("native.cpp")
        flags = ["-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off"]
        compiler = os.environ.get("CXX", "c++")
        identity = subprocess.check_output([compiler, "--version"])
        source_bytes = source.read_bytes()
        digest = hashlib.sha256(source_bytes + identity + repr(flags).encode()).hexdigest()
        cache = Path(tempfile.gettempdir()) / ("ce-moon-product-" + str(os.getuid()))
        cache.mkdir(mode=0o700, exist_ok=True)
        target = cache / (digest + ".so")
        with (cache / (digest + ".lock")).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=cache) as stage:
                    candidate = Path(stage) / "product.so"
                    captured_source = Path(stage) / "native.cpp"
                    captured_source.write_bytes(source_bytes)
                    subprocess.run([compiler, *flags, str(captured_source), "-o", str(candidate)], check=True)
                    candidate.replace(target)
        lib = ctypes.CDLL(str(target))
        fp = ctypes.POINTER(ctypes.c_float)
        ip = ctypes.POINTER(ctypes.c_int64)
        common = [ctypes.c_int64, ctypes.c_int64, fp, fp, ip, ip]
        for name, extra in [("forward", [fp]), ("vjp", [fp, fp, fp]), ("jvp", [fp, fp, fp])]:
            fn = getattr(lib, "product_" + name)
            fn.argtypes = common + extra
            fn.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _vector(value, dtype, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must be a rank-1 {np.dtype(dtype)} array")
    return np.ascontiguousarray(array)


def _generations(value):
    values = tuple(value)
    if len(values) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
        raise ValueError("generations requires four nonnegative integer epochs")
    return tuple(int(v) for v in values)


def _immutable(array):
    # A bytes backing store cannot be made writable by changing ndarray flags.
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


@dataclass(frozen=True)
class Tape:
    x: np.ndarray
    k: np.ndarray
    a: np.ndarray
    b: np.ndarray
    generations: tuple

    def __post_init__(self):
        for name, dtype in [("x", np.float32), ("k", np.float32), ("a", np.int64), ("b", np.int64)]:
            object.__setattr__(self, name, _immutable(_vector(getattr(self, name), dtype, name)))
        if self.a.size != self.k.size or self.b.size != self.k.size:
            raise ValueError("tape packet extents differ")
        if any(np.any((ids < 0) | (ids >= self.x.size)) for ids in (self.a, self.b)):
            raise ValueError("process input index outside logical input")
        object.__setattr__(self, "generations", _generations(self.generations))


def _call(name, tape, *outputs):
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int64)
    args = [tape.x.size, tape.k.size, tape.x.ctypes.data_as(fp), tape.k.ctypes.data_as(fp),
            tape.a.ctypes.data_as(ip), tape.b.ctypes.data_as(ip)]
    error = getattr(_library(), "product_" + name)(*args, *(a.ctypes.data_as(fp) for a in outputs))
    if error:
        raise ValueError(f"native product {name} admission failed ({error})")


def _check(tape, current_generations):
    if not isinstance(tape, Tape):
        raise TypeError("expected product Tape")
    if current_generations is not None and _generations(current_generations) != tape.generations:
        raise ValueError("stale product tape generations")


def forward(x, k, a, b, generations=(0, 0, 0, 0)):
    x, k = _vector(x, np.float32, "x"), _vector(k, np.float32, "k")
    a, b = _vector(a, np.int64, "a"), _vector(b, np.int64, "b")
    if a.size != k.size or b.size != k.size:
        raise ValueError("packet coefficients and both index arrays must have equal length")
    tape = Tape(x, k, a, b, generations)
    y = np.empty(k.size, np.float32)
    _call("forward", tape, y)
    return y, tape


def vjp(tape, g, current_generations=None):
    _check(tape, current_generations)
    g = _vector(g, np.float32, "g")
    if g.size != tape.k.size:
        raise ValueError("cotangent must have one entry per process")
    dx, dk = np.empty_like(tape.x), np.empty_like(tape.k)
    _call("vjp", tape, g, dx, dk)
    return dx, dk


def jvp(tape, dx, dk, current_generations=None):
    _check(tape, current_generations)
    dx, dk = _vector(dx, np.float32, "dx"), _vector(dk, np.float32, "dk")
    if dx.size != tape.x.size or dk.size != tape.k.size:
        raise ValueError("directions must match logical input and packet coefficient extents")
    dy = np.empty_like(tape.k)
    _call("jvp", tape, dx, dk, dy)
    return dy
