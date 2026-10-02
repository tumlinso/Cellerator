"""Small, explicit references for proposed single-cell operators.

These use NumPy float64 unless specified. They establish algebra, not CUDA speed.
The hidden coordinates of different actors need not have aligned meanings.
"""
from __future__ import annotations
import numpy as np

def _matrix(x, name):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite matrix")
    return a

def matrix_patch(x, left, right, nonlinear=False):
    """L X R; a nonlinearity, when requested, sits between the two products."""
    x, left, right = (_matrix(a,n) for a,n in ((x,'x'),(left,'left'),(right,'right')))
    if left.shape[1] != x.shape[0] or x.shape[1] != right.shape[0]:
        raise ValueError('incompatible patch shapes')
    t = left @ x
    return (np.tanh(t) if nonlinear else t) @ right

def matrix_patch_vjp(x, left, right, cotangent, nonlinear=False):
    """Returns gradients for X,L,R, in that order. No optimizer mutation."""
    t = left @ x
    v = np.tanh(t) if nonlinear else t
    dt = cotangent @ right.T
    if nonlinear:
        dt *= 1-v*v
    return left.T @ dt, dt @ x.T, v.T @ cotangent

def matrix_patch_jvp(x,left,right,dx,dleft=None,dright=None,nonlinear=False):
    t=left @ x
    dt=left @ dx
    if dleft is not None: dt=dt+dleft @ x
    v=np.tanh(t) if nonlinear else t
    dv=(1-v*v)*dt if nonlinear else dt
    out=dv @ right
    if dright is not None: out=out+v @ dright
    return out

def kronecker_matrix(left,right):
    """vec_F(L X R) = (R.T kron L) vec_F(X). Not an arbitrary dense map."""
    return np.kron(right.T,left)

def local_port_transport(h, encoders, decoders, adjacency):
    """h[i] is LOCAL coordinates; a port is a shared *message interface*.

    encoders[i]: ports x local_width; decoders[i]: local_width x ports.
    adjacency[dst,src]. Edges are supplied, never inferred by this function.
    """
    h=np.asarray(h,dtype=np.float64)
    e=np.asarray(encoders,dtype=np.float64)
    d=np.asarray(decoders,dtype=np.float64)
    a=_matrix(adjacency,'adjacency')
    if h.ndim!=2 or e.ndim!=3 or d.ndim!=3:
        raise ValueError('expected rank-2 states and rank-3 local maps')
    n,width=h.shape
    if e.shape[0]!=n or e.shape[2]!=width or d.shape!=(n,width,e.shape[1]) or a.shape!=(n,n):
        raise ValueError('invalid port dimensions')
    msg=np.einsum('iph,ih->ip',e,h)
    incoming=a @ msg
    return np.einsum('ihp,ip->ih',d,incoming)

def reframe_ports(h,encoders,decoders,changes):
    """Exact change of each actor's local coordinates for the linear interface.

    Does NOT claim an arbitrary parameterized local nonlinear law is closed
    under the transformation. That law must be conjugated separately.
    """
    qs=np.asarray(changes,dtype=np.float64)
    hp=np.einsum('ihk,ik->ih',qs,h)
    ep=np.stack([e @ np.linalg.inv(q) for e,q in zip(encoders,qs)])
    dp=np.einsum('ihk,ikp->ihp',qs,decoders)
    return hp,ep,dp

def product_value_jvp(arguments,directions,coefficient=1.0):
    """Zero-safe product/JVP; repeated slots retain multiplicity."""
    x=np.asarray(arguments,dtype=np.float64); v=np.asarray(directions,dtype=np.float64)
    if x.ndim!=1 or v.shape!=x.shape: raise ValueError('slot shape mismatch')
    n=x.size
    pre=np.ones(n+1); suf=np.ones(n+1)
    for j in range(n): pre[j+1]=pre[j]*x[j]
    for j in range(n-1,-1,-1): suf[j]=suf[j+1]*x[j]
    partials=coefficient*pre[:-1]*suf[1:]
    return coefficient*pre[-1],float(partials @ v),partials

def low_rank_implicit_solve(diagonal,u,v,rhs,dt):
    """Solve (I-dt*(diag(diagonal)+u@v.T)) y=rhs by Woodbury.

    A linear algebra primitive, NOT an adaptive ODE integrator. May be unstable
    if the diagonal or reduced solve is ill-conditioned; caller selects policy.
    """
    a=1-dt*np.asarray(diagonal,dtype=np.float64)
    u=np.asarray(u,dtype=np.float64); v=np.asarray(v,dtype=np.float64)
    b=np.asarray(rhs,dtype=np.float64)
    if u.shape!=v.shape or u.shape[0]!=a.size or b.shape[0]!=a.size:
        raise ValueError('invalid Woodbury shapes')
    if np.any(np.abs(a)<1e-14): raise ValueError('singular diagonal component')
    au=dt*u/a[:,None]
    ab=b/a if b.ndim==1 else b/a[:,None]
    small=np.eye(u.shape[1])-v.T @ au
    return ab+au @ np.linalg.solve(small,v.T @ ab)

def quad_fragment_coordinates(lane,element):
    """PTX sm70 m8n8k4 FP16 -> FP32 accumulator mapping, not WMMA ABI.

    Independent MMA index, row, column. PTX 8.7 section 9.7.14.5.1.
    """
    if not 0<=lane<32 or not 0<=element<8: raise ValueError('out of range')
    group=(lane>>2)&3
    row=(lane&1)+(element&2)+(4 if lane>=16 else 0)
    col=(element&4)+(lane&2)+(element&1)
    return group,row,col


def quadratic_matrix_field(x,left,right,middle):
    """A structured field on existing scalar coordinates: L X + X R + X M X."""
    return left @ x + x @ right + x @ middle @ x

def quadratic_matrix_jvp(x,left,right,middle,direction):
    return (left @ direction + direction @ right
            + direction @ middle @ x + x @ middle @ direction)

def quadratic_matrix_vjp(x,left,right,middle,cotangent):
    return (left.T @ cotangent + cotangent @ right.T
            + cotangent @ (middle @ x).T + (x @ middle).T @ cotangent)

def sylvester_flow(x,exp_left,exp_right):
    """Exact-in-real-arithmetic flow given exp(dt L), exp(dt R).

    Computing/caching exponentials is the caller's job. Valid for autonomous
    dX=LX+XR with fixed L,R. This is NOT the flow of dX=LXR.
    """
    return exp_left @ x @ exp_right
