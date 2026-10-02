"""Caller-authorized refactorings, not latent-identity discovery.

Exact algebra in these witnesses does not establish biological equivalence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def conjugate_linear_model(a,readout,change_of_coordinates):
    a=np.asarray(a,dtype=np.float64); o=np.asarray(readout,dtype=np.float64)
    t=np.asarray(change_of_coordinates,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or t.shape!=a.shape or o.shape[1]!=a.shape[0]:
        raise ValueError('invalid model dimensions')
    inv=np.linalg.inv(t)
    return t @ a @ inv,o @ inv

def quotient_linear_model(a,readout,embedding,projection,tolerance=1e-11):
    """Old state s=E q. Exact ONLY on the supplied invariant manifold.

    P E=I and A E=E(P A E) are checked. Equality of two scalar values at one
    instant is insufficient. Tangent/adjoint domains change with the model.
    """
    a=np.asarray(a,dtype=np.float64); o=np.asarray(readout,dtype=np.float64)
    e=np.asarray(embedding,dtype=np.float64); p=np.asarray(projection,dtype=np.float64)
    if e.ndim!=2 or p.shape!=(e.shape[1],e.shape[0]) or a.shape!=(e.shape[0],)*2:
        raise ValueError('invalid quotient shapes')
    if not np.allclose(p@e,np.eye(e.shape[1]),atol=tolerance,rtol=0):
        raise ValueError('projection is not a left inverse')
    new_a=p @ a @ e
    if not np.allclose(a@e,e@new_a,atol=tolerance,rtol=0):
        raise ValueError('declared equality is not dynamically closed')
    return new_a,o@e

def pullback_replicated_adjoint(embedding,old_adjoint):
    """Copies contribute a SUM, not an average, to the shared coordinate."""
    return np.asarray(embedding).T @ np.asarray(old_adjoint)

@dataclass
class EpochGuard:
    structure_epoch: int=1
    live_tapes: int=0
    def acquire(self):
        self.live_tapes+=1
        return self.structure_epoch
    def release(self,epoch):
        if epoch!=self.structure_epoch or self.live_tapes<=0:
            raise RuntimeError('invalid or stale tape')
        self.live_tapes-=1
    def publish(self):
        if self.live_tapes: raise RuntimeError('finish old backward before promotion')
        self.structure_epoch+=1
        return self.structure_epoch

class RecyclableResidual:
    """Reference regrowth branch r=V tanh(U x).

    Reset outgoing V to zero, incoming U to nonzero seeded values, and clear
    optimizer moments. Forward contribution is zero at birth; V can learn.
    This is an experiment controller helper, NOT a native optimizer.
    """
    def __init__(self,input_width,slots,output_width,seed=0):
        if min(input_width,slots,output_width)<=0: raise ValueError('positive sizes required')
        self.rng=np.random.default_rng(seed)
        self.u=self.rng.normal(0,0.1,(slots,input_width))
        self.v=np.zeros((output_width,slots))
        self.m_u=np.zeros_like(self.u); self.v_u=np.zeros_like(self.u)
        self.m_v=np.zeros_like(self.v); self.v_v=np.zeros_like(self.v)
    def recycle(self,slots):
        ix=np.unique(np.asarray(slots,dtype=np.int64))
        if np.any(ix<0) or np.any(ix>=self.u.shape[0]): raise ValueError('invalid slot')
        self.u[ix]=self.rng.normal(0,0.1,(ix.size,self.u.shape[1]))
        self.v[:,ix]=0
        self.m_u[ix]=0; self.v_u[ix]=0; self.m_v[:,ix]=0; self.v_v[:,ix]=0
    def forward(self,x):
        return self.v @ np.tanh(self.u @ x)
    def vjp(self,x,adjoint):
        z=np.tanh(self.u @ x)
        dv=np.outer(adjoint,z)
        du=np.outer((self.v.T @ adjoint)*(1-z*z),x)
        return du,dv
