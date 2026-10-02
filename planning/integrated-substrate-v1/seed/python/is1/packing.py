"""Strategy-independent executable packing/contribution witness.

The CPU implementation validates/allocates liberally. It is an oracle and a
cold-contract example, not Cellerator's native runtime or a timing baseline.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from collections import Counter
from typing import Callable
import numpy as np
from numpy.typing import ArrayLike
from .effects import finite, Array

@dataclass(frozen=True)
class Contribution:
    destination: int
    scale: float = 1.0

@dataclass(frozen=True)
class Process:
    id: int
    opcode: str
    arguments: tuple[int,...]
    parameter: int
    outputs: tuple[Contribution,...]

@dataclass(frozen=True)
class Problem:
    state_extent: int
    output_extent: int
    parameter_extent: int
    processes: tuple[Process,...]

    def validate(self) -> None:
        ext=(self.state_extent,self.output_extent,self.parameter_extent)
        if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in ext):
            raise ValueError("invalid axis extent")
        ids=[p.id for p in self.processes]
        if any(isinstance(i,bool) or not isinstance(i,int) or i<0 for i in ids) or len(set(ids))!=len(ids):
            raise ValueError("process IDs must be unique nonnegative integers")
        for p in self.processes:
            if p.opcode not in ('sum','product'):
                raise ValueError("unsupported reference opcode")
            if not isinstance(p.parameter,int) or not 0<=p.parameter<self.parameter_extent:
                raise ValueError("parameter out of bounds")
            if any(not isinstance(i,int) or not 0<=i<self.state_extent for i in p.arguments):
                raise ValueError("argument outside state axis")
            if any(not isinstance(c.destination,int) or not 0<=c.destination<self.output_extent or not np.isfinite(c.scale) for c in p.outputs):
                raise ValueError("invalid output contribution")

    def signature(self) -> str:
        self.validate()
        return sha256(json.dumps(asdict(self),sort_keys=True,separators=(',',':')).encode()).hexdigest()

@dataclass(frozen=True)
class Plan:
    problem_signature: str
    family: str
    state_order: tuple[int,...]
    groups: tuple[tuple[int,...],...]

    @property
    def work_order(self) -> tuple[int,...]:
        return tuple(i for g in self.groups for i in g)

Strategy=Callable[[Problem,int],Plan]

def validate_plan(problem:Problem,plan:Plan) -> None:
    if plan.problem_signature != problem.signature():
        raise ValueError("stale or wrong semantic problem")
    if not plan.family or any(isinstance(i,bool) or not isinstance(i,int) for i in plan.state_order) or sorted(plan.state_order)!=list(range(problem.state_extent)):
        raise ValueError("state placement is not a complete permutation")
    if any(not g for g in plan.groups):
        raise ValueError("empty cohort")
    if any(isinstance(i,bool) or not isinstance(i,int) for i in plan.work_order) or Counter(plan.work_order)!=Counter(p.id for p in problem.processes):
        raise ValueError("incomplete or duplicated process ownership")

def _chunks(xs:list[int],width:int) -> tuple[tuple[int,...],...]:
    if isinstance(width,bool) or not isinstance(width,int) or width<1:
        raise ValueError("invalid cohort width")
    return tuple(tuple(xs[i:i+width]) for i in range(0,len(xs),width))

def identity(problem:Problem,width:int=32) -> Plan:
    return Plan(problem.signature(),'identity',tuple(range(problem.state_extent)),
                _chunks([p.id for p in problem.processes],width))

def opcode_cohorts(problem:Problem,width:int=32) -> Plan:
    problem.validate();_chunks([],width)
    groups=[]
    for op in sorted(set(p.opcode for p in problem.processes)):
        group=sorted((p for p in problem.processes if p.opcode==op),
                     key=lambda p:(len(p.arguments),p.arguments,p.id))
        groups.extend(_chunks([p.id for p in group],width))
    # An independent state placement favors frequent loads; it does not rename variables.
    freq=Counter(i for p in problem.processes for i in p.arguments)
    state=tuple(sorted(range(problem.state_extent),key=lambda i:(-freq[i],i)))
    return Plan(problem.signature(),'opcode-cohorts',state,tuple(groups))

def support_cohorts(problem:Problem,width:int=32) -> Plan:
    problem.validate();_chunks([],width)
    remaining={p.id:p for p in problem.processes};groups=[]
    while remaining:
        first=min(remaining);p=remaining.pop(first);group=[first];support=set(p.arguments)
        while len(group)<width:
            compatible=[q for q in remaining.values() if q.opcode==p.opcode]
            if not compatible:break
            def key(q:Process) -> tuple[float,int]:
                s=set(q.arguments);u=s|support
                similarity=len(s&support)/len(u) if u else 1.0
                return (-similarity,q.id)
            q=min(compatible,key=key);group.append(q.id);support.update(q.arguments);del remaining[q.id]
        groups.append(tuple(group))
    state=[];seen=set()
    by_id={p.id:p for p in problem.processes}
    for group in groups:
        for pid in group:
            for i in by_id[pid].arguments:
                if i not in seen:state.append(i);seen.add(i)
    state.extend(i for i in range(problem.state_extent) if i not in seen)
    return Plan(problem.signature(),'support-cohorts',tuple(state),tuple(groups))

def propose(problem:Problem,strategy:Strategy,width:int=32) -> Plan:
    """A caller-supplied callable; no central switch or mandatory image format."""
    result=strategy(problem,width);validate_plan(problem,result);return result

def proxy(problem:Problem,plan:Plan) -> dict[str,int]:
    validate_plan(problem,plan);by_id={p.id:p for p in problem.processes}
    return {'groups':len(plan.groups),'gathered_unique_inputs':sum(len({i for pid in g for i in by_id[pid].arguments}) for g in plan.groups),
            'ordered_argument_uses':sum(len(p.arguments) for p in problem.processes),
            'output_contributions':sum(len(p.outputs) for p in problem.processes)}

def _bindings(problem:Problem,plan:Plan,x:ArrayLike,theta:ArrayLike):
    validate_plan(problem,plan)
    x,theta=finite(x,1,'x'),finite(theta,1,'parameters')
    if x.shape!=(problem.state_extent,) or theta.shape!=(problem.parameter_extent,):
        raise ValueError('bound axis mismatch')
    physical=x[list(plan.state_order)];inverse=np.empty(problem.state_extent,dtype=int)
    inverse[list(plan.state_order)]=np.arange(problem.state_extent)
    return physical,inverse,theta,{p.id:p for p in problem.processes}

def execute(problem:Problem,plan:Plan,x:ArrayLike,theta:ArrayLike) -> Array:
    physical,inverse,theta,by_id=_bindings(problem,plan,x,theta)
    out=np.zeros(problem.output_extent)
    for pid in plan.work_order:
        p=by_id[pid];args=physical[inverse[list(p.arguments)]]
        base=float(np.prod(args)) if p.opcode=='product' else float(np.sum(args))
        value=theta[p.parameter]*base
        for c in p.outputs:out[c.destination]+=c.scale*value
    return out

def vjp(problem:Problem,plan:Plan,x:ArrayLike,theta:ArrayLike,
        output_bar:ArrayLike) -> tuple[Array,Array]:
    physical,inverse,theta,by_id=_bindings(problem,plan,x,theta)
    G=finite(output_bar,1,'output_bar')
    if G.shape!=(problem.output_extent,):raise ValueError('cotangent axis mismatch')
    dx=np.zeros(problem.state_extent);dp=np.zeros(problem.parameter_extent)
    for pid in plan.work_order:
        p=by_id[pid];args=physical[inverse[list(p.arguments)]]
        bar=sum(c.scale*G[c.destination] for c in p.outputs)
        base=float(np.prod(args)) if p.opcode=='product' else float(np.sum(args))
        dp[p.parameter]+=bar*base
        for slot,i in enumerate(p.arguments):
            # Exclusion product is deliberately zero-safe and retains repeated slots.
            local=float(np.prod(np.delete(args,slot))) if p.opcode=='product' else 1.0
            dx[i]+=bar*theta[p.parameter]*local
    return dx,dp
