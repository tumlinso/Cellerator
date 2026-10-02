"""Support-driven placement witness. No equivalence/causality inference.

A bit index names an EXECUTION SITE (input use, output use, or active context),
not an embedding column. Similarity nominates co-placement, never shared IDs.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import random

@dataclass(frozen=True)
class Footprint:
    name: str
    reads: frozenset[int]
    writes: frozenset[int]
    contexts: frozenset[int]=frozenset()
    program: str='generic'

def bits(s):
    out=0
    for i in s:
        if i<0: raise ValueError('negative bit index')
        out|=1<<i
    return out

def jaccard(a,b):
    a,b=bits(a),bits(b)
    union=(a|b).bit_count()
    return (a&b).bit_count()/union if union else 0.0

def minhash_candidates(footprints,bands=8,rows=4,seed=0,cap=64):
    """Small CPU witness; use native CE MinHash/LSH for production.

    Features are tagged by direction to prevent mixing incoming and outgoing
    universes. Empty supports nominate no peers. Caps intentionally lose recall.
    """
    if min(bands,rows)<=0 or cap<2: raise ValueError('invalid LSH parameters')
    rng=random.Random(seed); prime=(1<<61)-1
    hashes=[(rng.randrange(1,prime),rng.randrange(prime)) for _ in range(bands*rows)]
    vocab={}
    feature_sets=[]
    for f in footprints:
        tokens=[('r',x) for x in f.reads]+[('w',x) for x in f.writes]+[('c',x) for x in f.contexts]
        for t in sorted(tokens):
            if t not in vocab: vocab[t]=len(vocab)+1
        feature_sets.append([vocab[t] for t in tokens])
    buckets=defaultdict(list)
    for idx,items in enumerate(feature_sets):
        if not items: continue
        sig=[min((a*x+b)%prime for x in items) for a,b in hashes]
        for band in range(bands):
            buckets[(band,tuple(sig[band*rows:(band+1)*rows]))].append(idx)
    pairs=set()
    for key,members in sorted(buckets.items()):
        # Deterministic capped rotating window, NOT a native CE hash match.
        start=sum(key[1])%len(members)
        take=(members[start:]+members[:start])[:cap]
        for pos,a in enumerate(take):
            for b in take[pos+1:]: pairs.add(tuple(sorted((a,b))))
    return sorted(pairs)

def placement_cost(group,width=16):
    """Illustrative traffic/padding proxy, NOT measured execution time."""
    if not group: return 0.0
    if len(group)>width: return float('inf')
    reads=set().union(*(f.reads for f in group))
    writes=set().union(*(f.writes for f in group))
    contexts=set().union(*(f.contexts for f in group))
    # Distinct programs increase dispatch complexity; never pretend unequal
    # parameter maps are one shared matrix just because their opcodes match.
    programs=len({f.program for f in group})
    return 16.0+4*len(reads)+4*len(writes)+0.25*(width-len(group))+3*programs+0.1*len(contexts)

def pack(footprints,width=16):
    candidates=minhash_candidates(footprints)
    groups={i:[i] for i in range(len(footprints))}; owner=list(range(len(footprints)))
    for i,j in candidates:
        a,b=owner[i],owner[j]
        if a==b: continue
        merged=groups[a]+groups[b]
        if len(merged)>width: continue
        get=lambda ids:[footprints[k] for k in ids]
        if placement_cost(get(merged),width)>=placement_cost(get(groups[a]),width)+placement_cost(get(groups[b]),width): continue
        groups[a]=merged; del groups[b]
        for k in merged: owner[k]=a
    return [tuple(footprints[i].name for i in ids) for _,ids in sorted(groups.items())]
