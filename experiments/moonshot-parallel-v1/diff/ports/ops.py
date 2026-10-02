"""Native CPU float32 private-port transport; local laws are caller-owned."""
from dataclasses import dataclass
from pathlib import Path
import ctypes
import fcntl
import hashlib
import os
import subprocess
import tempfile
import numpy as np

@dataclass(frozen=True)
class Tape:
    h: np.ndarray
    e: np.ndarray
    d: np.ndarray
    weights: np.ndarray
    widths: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    generations: tuple
    p: int

    def __post_init__(self):
        values = {}
        for name in ('h', 'e', 'd', 'weights'):
            values[name] = _float(getattr(self, name), name)
        for name in ('widths', 'src', 'dst'):
            values[name] = _integer(getattr(self, name), name)
        widths = values['widths']
        if not widths.size or (widths <= 0).any():
            raise ValueError('positive private widths required')
        if isinstance(self.p, (bool, np.bool_)) or not isinstance(self.p, (int, np.integer)) or self.p <= 0:
            raise ValueError('positive integer port width required')
        total = sum(int(v) for v in widths)
        p = int(self.p)
        # Bounds also cover native signed index arithmetic and byte extents.
        limit = np.iinfo(np.intp).max // np.dtype(np.float32).itemsize
        if total > limit or total*p > limit or widths.size*p > limit:
            raise ValueError('private port dimensions overflow native extents')
        if values['h'].size != total or values['e'].size != total*p or values['d'].size != total*p:
            raise ValueError('saved port primal extents disagree')
        k = values['weights'].size
        if values['src'].size != k or values['dst'].size != k:
            raise ValueError('saved edge extents disagree')
        for name in ('src', 'dst'):
            ids = values[name]
            if (ids < 0).any() or (ids >= widths.size).any():
                raise ValueError('saved edge actor index out of range')
        generations = _generations(self.generations)
        for name, array in values.items():
            object.__setattr__(self, name, _freeze(array))
        object.__setattr__(self, 'generations', generations)
        object.__setattr__(self, 'p', p)

_LIB = None

def _library():
    global _LIB
    if _LIB is None:
        source = Path(__file__).with_name('transport.cc')
        source_bytes = source.read_bytes()
        flags = ['-std=c++17', '-O2', '-shared', '-fPIC', '-ffp-contract=off']
        compiler = os.environ.get('CXX', 'c++')
        identity = subprocess.check_output([compiler, '--version'])
        digest = hashlib.sha256(source_bytes + identity + repr(flags).encode()).hexdigest()
        cache = Path(tempfile.gettempdir()) / ('moonshot-ports-' + str(os.getuid()))
        cache.mkdir(mode=0o700, exist_ok=True)
        target = cache / (digest + '.so')
        with (cache / (digest + '.lock')).open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=cache) as stage:
                    captured = Path(stage) / 'transport.cc'
                    captured.write_bytes(source_bytes)
                    temporary = Path(stage) / 'transport.so'
                    subprocess.run([compiler, *flags, str(captured), '-o', str(temporary)], check=True)
                    os.replace(temporary, target)
        _LIB = ctypes.CDLL(str(target)).ports_transport
        _LIB.restype = ctypes.c_int
        _LIB.argtypes = [ctypes.c_int] + [ctypes.c_int64]*3 + [ctypes.c_void_p]*15
    return _LIB

def _float(x, name, size=None):
    a = np.asarray(x)
    if a.dtype != np.float32 or a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError(name + ' must be a finite flattened float32 array')
    if size is not None and a.size != size:
        raise ValueError(name + ' has invalid size')
    return np.ascontiguousarray(a)

def _integer(x, name):
    a = np.asarray(x)
    if a.dtype != np.int64 or a.ndim != 1:
        raise ValueError(name + ' must be a flattened int64 array')
    return np.ascontiguousarray(a)

def _freeze(a):
    # A bytes owner prevents making the saved array writable again.
    return np.frombuffer(a.tobytes(), dtype=a.dtype)

def _generations(g):
    g = tuple(g)
    if len(g) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in g):
        raise ValueError('four nonnegative integer generations required')
    return g

def _check(tape, current):
    if not isinstance(tape, Tape):
        raise TypeError('ports Tape required')
    if current is not None and _generations(current) != tape.generations:
        raise ValueError('stale saved primal generations')

def _run(mode, t, inputs, outputs):
    args = [t.widths,t.src,t.dst,t.h,t.e,t.d,t.weights] + list(inputs) + list(outputs)
    pointers = [None if a is None else a.ctypes.data for a in args]
    if _library()(mode,t.widths.size,t.p,t.weights.size,*pointers):
        raise RuntimeError('native port execution failed')
    if any(not np.isfinite(a).all() for a in outputs if a is not None):
        raise FloatingPointError('native port result overflowed')

def forward(h, e, d, weights, widths, src, dst, generations=(0,0,0,0)):
    widths = _integer(widths,'widths')
    if not widths.size or (widths <= 0).any():
        raise ValueError('positive private widths required')
    total = sum(int(v) for v in widths)
    h = _float(h,'h',total)
    e = _float(e,'e')
    if not e.size or e.size % total:
        raise ValueError('encoder size must be P times total private width')
    p = e.size // total
    d = _float(d,'d',e.size)
    weights = _float(weights,'weights')
    src, dst = _integer(src,'src'), _integer(dst,'dst')
    if src.size != weights.size or dst.size != weights.size:
        raise ValueError('edge array sizes disagree')
    if ((src < 0).any() or (dst < 0).any() or
        (src >= widths.size).any() or (dst >= widths.size).any()):
        raise ValueError('edge actor index out of range')
    t = Tape(h,e,d,weights,widths,src,dst,generations,p)
    y = np.empty_like(h)
    _run(0,t,(None,)*4,(y,None,None,None))
    return y,t

def vjp(tape, g, current_generations=None):
    _check(tape,current_generations)
    g = _float(g,'cotangent',tape.h.size)
    outputs = tuple(np.empty_like(a) for a in (tape.h,tape.e,tape.d,tape.weights))
    _run(1,tape,(g,None,None,None),outputs)
    return outputs

def jvp(tape, dh, de, dd, dweights, current_generations=None):
    _check(tape,current_generations)
    inputs = tuple(_float(v,'tangent',a.size) for v,a in zip((dh,de,dd,dweights),(tape.h,tape.e,tape.d,tape.weights)))
    y = np.empty_like(tape.h)
    _run(2,tape,inputs,(y,None,None,None))
    return y
