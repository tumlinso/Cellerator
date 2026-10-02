"""Reference mathematics for integrated matrix, delta and port operators."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
from .effects import Array, finite

def _square(*xs: ArrayLike) -> tuple[Array,...]:
    a = tuple(finite(x,2,"matrix") for x in xs)
    if not a or a[0].shape[0] == 0 or a[0].shape[0] != a[0].shape[1]:
        raise ValueError("expected a nonempty square matrix")
    if any(x.shape != a[0].shape for x in a):
        raise ValueError("matrix extents differ")
    return a

def quadratic(X: ArrayLike,L: ArrayLike,R: ArrayLike,M: ArrayLike) -> Array:
    X,L,R,M = _square(X,L,R,M)
    return L@X + X@R + X@M@X

def quadratic_jvp(X: ArrayLike,L: ArrayLike,R: ArrayLike,M: ArrayLike,
                  P: ArrayLike) -> Array:
    X,L,R,M,P = _square(X,L,R,M,P)
    return L@P + P@R + P@M@X + X@M@P

def quadratic_vjp(X: ArrayLike,L: ArrayLike,R: ArrayLike,M: ArrayLike,
                  G: ArrayLike) -> dict[str,Array]:
    X,L,R,M,G = _square(X,L,R,M,G)
    return {"X": L.T@G + G@R.T + G@(M@X).T + (X@M).T@G,
            "L": G@X.T, "R": X.T@G, "M": X.T@G@X.T}

def quadratic_delta(X: ArrayLike,D: ArrayLike,L: ArrayLike,R: ArrayLike,
                    M: ArrayLike) -> Array:
    X,D,L,R,M = _square(X,D,L,R,M)
    return L@D + D@R + D@M@X + X@M@D + D@M@D

@dataclass
class QuadraticLedger:
    """Small remembered-transmission witness; not a native asynchronous owner.

    Coefficients are copied and immutable. A parameter epoch change rebuilds the
    cache; reusing an epoch with different coefficients is an error, not a hit.
    Thresholding is approximate execution relative to the current input.
    """
    sent: Array
    L: Array
    R: Array
    M: Array
    parameter_epoch: int = 0

    def __post_init__(self) -> None:
        self.sent,self.L,self.R,self.M = _square(self.sent,self.L,self.R,self.M)
        if isinstance(self.parameter_epoch,bool) or not isinstance(self.parameter_epoch,int) or self.parameter_epoch < 0:
            raise ValueError("negative parameter epoch")
        self.value = quadratic(self.sent,self.L,self.R,self.M)

    def update(self, current: ArrayLike, threshold: float = 0.0) -> Array:
        current, = _square(current)
        if current.shape != self.sent.shape or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid update shape/threshold")
        D = np.where(np.abs(current-self.sent)>threshold,current-self.sent,0.0)
        self.value = self.value + quadratic_delta(self.sent,D,self.L,self.R,self.M)
        self.sent = finite(self.sent+D,2,"sent")
        return self.value.copy()

    def refresh(self, current: ArrayLike) -> Array:
        current, = _square(current)
        if current.shape != self.sent.shape:
            raise ValueError("refresh shape mismatch")
        self.sent = current
        self.value = quadratic(current,self.L,self.R,self.M)
        return self.value.copy()

    def rebind_parameters(self,L: ArrayLike,R: ArrayLike,M: ArrayLike,
                          epoch: int, current: ArrayLike) -> Array:
        L,R,M = _square(L,R,M)
        if isinstance(epoch,bool) or not isinstance(epoch,int) or L.shape != self.L.shape or epoch < self.parameter_epoch:
            raise ValueError("invalid shape or stale parameter epoch")
        equal = all(np.array_equal(a,b) for a,b in zip((L,R,M),(self.L,self.R,self.M)))
        if epoch == self.parameter_epoch and not equal:
            raise ValueError("coefficient mutation without generation publication")
        current, = _square(current)
        if current.shape != self.sent.shape:
            raise ValueError("rebind state shape mismatch")
        new_value=quadratic(current,L,R,M)
        self.L,self.R,self.M,self.parameter_epoch=L,R,M,epoch
        self.sent,self.value=current,new_value
        return self.value.copy()

def multilevel_apply(x: ArrayLike,D: ArrayLike,P: ArrayLike,K: ArrayLike,
                     R: ArrayLike) -> Array:
    x=finite(x,1,"x"); D,P,K,R=(finite(v,2,n) for v,n in zip((D,P,K,R),"DPKR"))
    n=x.size
    if D.shape != (n,n) or R.shape[1] != n or P.shape[0] != n:
        raise ValueError("fine-axis mismatch")
    k=R.shape[0]
    if K.shape != (k,k) or P.shape[1] != k:
        raise ValueError("coarse-axis mismatch")
    return D@x + P@(K@(R@x))

def solve_jvp(A: ArrayLike,x: ArrayLike,dA: ArrayLike,db: ArrayLike) -> Array:
    A,dA = _square(A,dA); x,db=finite(x,1,"x"),finite(db,1,"db")
    if x.shape != (A.shape[0],) or db.shape != x.shape:
        raise ValueError("solve derivative extent mismatch")
    return np.linalg.solve(A,db-dA@x)

def schur_solve(A: ArrayLike,rhs: ArrayLike,boundary: list[int],
                maximum_condition: float=1e12) -> dict[str,Array|float]:
    A, = _square(A); b=finite(rhs,1,"rhs"); n=A.shape[0]
    if b.shape != (n,) or not boundary or len(boundary)>=n:
        raise ValueError("nonempty proper boundary and matching RHS required")
    if any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) for i in boundary):
        raise ValueError("boundary indices must be integers")
    if len(set(boundary))!=len(boundary) or min(boundary)<0 or max(boundary)>=n:
        raise ValueError("invalid/repeated boundary index")
    if not np.isfinite(maximum_condition) or maximum_condition < 1:
        raise ValueError("invalid conditioning policy")
    interior=[i for i in range(n) if i not in set(boundary)]
    Abb=A[np.ix_(boundary,boundary)]; Abi=A[np.ix_(boundary,interior)]
    Aib=A[np.ix_(interior,boundary)]; Aii=A[np.ix_(interior,interior)]
    cond_i=float(np.linalg.cond(Aii))
    if not np.isfinite(cond_i) or cond_i>maximum_condition:
        raise ValueError("interior solve fails conditioning policy")
    recovery=-np.linalg.solve(Aii,Aib)
    shift=np.linalg.solve(Aii,b[interior])
    S=Abb+Abi@recovery; g=b[boundary]-Abi@shift
    cond_s=float(np.linalg.cond(S))
    if not np.isfinite(cond_s) or cond_s>maximum_condition:
        raise ValueError("boundary solve fails conditioning policy")
    xb=np.linalg.solve(S,g); xi=recovery@xb+shift
    x=np.empty(n);x[boundary]=xb;x[interior]=xi
    return {"x":x,"S":S,"g":g,"recovery":recovery,"shift":shift,
            "residual_norm":float(np.linalg.norm(A@x-b)),
            "interior_condition":cond_i,"boundary_condition":cond_s}
