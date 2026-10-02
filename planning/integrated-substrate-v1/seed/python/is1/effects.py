"""Double-precision references for closed effect families.

These copies/validations are cold reference behavior. Native integration should
reuse Cellerator's actual identity, storage and prepared-execution owners.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

def finite(value: ArrayLike, ndim: int, name: str) -> Array:
    a = np.array(value, dtype=np.float64, copy=True)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name}: expected finite rank-{ndim} array")
    a.setflags(write=False)
    return a

@dataclass(frozen=True)
class AffineWitness:
    """h'=A h+b; q'=q+C h+d. `left.then(right)` visits left first."""
    A: Array
    b: Array
    C: Array
    d: Array

    def __post_init__(self) -> None:
        for name, rank in (("A",2),("b",1),("C",2),("d",1)):
            object.__setattr__(self, name, finite(getattr(self,name), rank,name))
        n = self.b.size
        if n < 1 or self.A.shape != (n,n) or self.C.shape != (self.d.size,n):
            raise ValueError("inconsistent state/readout extents")

    @staticmethod
    def identity(state: int, readouts: int) -> AffineWitness:
        if state < 1 or readouts < 0:
            raise ValueError("invalid identity extents")
        return AffineWitness(np.eye(state), np.zeros(state),
                             np.zeros((readouts,state)), np.zeros(readouts))

    def then(self, right: AffineWitness) -> AffineWitness:
        if self.A.shape != right.A.shape or self.C.shape != right.C.shape:
            raise ValueError("effect vocabulary/extent mismatch")
        return AffineWitness(right.A @ self.A,
                             right.A @ self.b + right.b,
                             self.C + right.C @ self.A,
                             self.d + right.C @ self.b + right.d)

    def apply(self, h: ArrayLike, q: ArrayLike) -> tuple[Array,Array]:
        h, q = finite(h,1,"h"), finite(q,1,"q")
        if h.shape != self.b.shape or q.shape != self.d.shape:
            raise ValueError("bound state/readout shape mismatch")
        return self.A @ h + self.b, q + self.C @ h + self.d

    def vjp(self, h: ArrayLike, state_bar: ArrayLike,
            readout_bar: ArrayLike) -> dict[str,Array]:
        h = finite(h,1,"h")
        u, v = finite(state_bar,1,"state_bar"), finite(readout_bar,1,"readout_bar")
        if h.shape != self.b.shape or u.shape != h.shape or v.shape != self.d.shape:
            raise ValueError("cotangent extent mismatch")
        return {"h": self.A.T @ u + self.C.T @ v, "q": v.copy(),
                "A": np.outer(u,h), "b": u.copy(),
                "C": np.outer(v,h), "d": v.copy()}

@dataclass(frozen=True)
class HybridAffine:
    """F(s,h)=(transition[s], A[s] h+b[s]); a fixed finite vocabulary."""
    transition: NDArray[np.int64]
    A: Array
    b: Array

    def __post_init__(self) -> None:
        t = np.array(self.transition, copy=True)
        if t.ndim != 1 or t.size < 1 or t.dtype.kind not in "iu":
            raise ValueError("transition must be a nonempty integer vector")
        if np.any(t < 0) or np.any(t >= t.size):
            raise ValueError("transition outside vocabulary")
        t = t.astype(np.int64); t.setflags(write=False)
        object.__setattr__(self,"transition",t)
        object.__setattr__(self,"A",finite(self.A,3,"A"))
        object.__setattr__(self,"b",finite(self.b,2,"b"))
        q, n = self.b.shape
        if n < 1 or q != t.size or self.A.shape != (q,n,n):
            raise ValueError("hybrid effect shape mismatch")

    @staticmethod
    def identity(vocabulary: int, width: int) -> HybridAffine:
        if vocabulary < 1 or width < 1:
            raise ValueError("invalid identity extents")
        return HybridAffine(np.arange(vocabulary),
                            np.repeat(np.eye(width)[None], vocabulary,axis=0),
                            np.zeros((vocabulary,width)))

    def then(self, right: HybridAffine) -> HybridAffine:
        if self.A.shape != right.A.shape:
            raise ValueError("hybrid vocabulary/width mismatch")
        selected = right.A[self.transition]
        return HybridAffine(right.transition[self.transition],
                            selected @ self.A,
                            np.einsum("sij,sj->si",selected,self.b)+right.b[self.transition])

    def apply(self, state: int, h: ArrayLike) -> tuple[int,Array]:
        if isinstance(state,bool) or not isinstance(state,(int,np.integer)):
            raise ValueError("control state must be an integer")
        if not 0 <= state < self.transition.size:
            raise ValueError("control state outside vocabulary")
        h = finite(h,1,"h")
        if h.shape != self.b.shape[1:]:
            raise ValueError("continuous state width mismatch")
        return int(self.transition[state]), self.A[state] @ h + self.b[state]
