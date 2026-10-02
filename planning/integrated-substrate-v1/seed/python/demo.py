#!/usr/bin/env python3
"""Toy end-to-end semantic witness; not a native repository integration test."""
import json
import numpy as np
from is1.effects import HybridAffine
from is1.packing import Problem,Process,Contribution,propose,support_cohorts,execute,vjp

def sequence_effect(sequence):
    effect=HybridAffine.identity(4,2)
    for base in sequence.upper():
        if base not in 'ACGT':
            local=HybridAffine(np.zeros(4,dtype=int),np.zeros((4,2,2)),np.zeros((4,2)))
        else:
            code='ACGT'.index(base)+1
            t=(np.arange(4)+code)%4
            A=np.repeat(np.array([[.7,.1],[0.,.8]])[None],4,axis=0)
            b=np.array([[.1*code,.025*s] for s in range(4)])
            local=HybridAffine(t,A,b)
        effect=effect.then(local)
    return effect.apply(0,np.zeros(2))[1]

sequences=['ACGT','AAAA','CGTA','NACG','TTGC','ACTA','CGCG','GATT']
features=np.array([sequence_effect(s) for s in sequences])
problem=Problem(2,1,2,(Process(0,'sum',(0,),0,(Contribution(0),)),
                       Process(1,'sum',(1,),1,(Contribution(0),))))
plan=propose(problem,support_cohorts,2)
truth=np.array([.5,-.2]);targets=features@truth;theta=np.zeros(2)
losses=[]
for step in range(250):
    grad=np.zeros(2);loss=0.
    for x,y in zip(features,targets):
        prediction=execute(problem,plan,x,theta)[0];error=prediction-y
        loss+=.5*error*error;_,g=vjp(problem,plan,x,theta,[error]);grad+=g
    losses.append(loss/len(targets));theta-=.25*grad/len(targets)
assert losses[-1]<losses[0]
print(json.dumps({'kind':'host_semantic_witness','native_repository_integration':False,
 'scientific_validation':False,'sequence_count':len(sequences),'source_strings_retained':True,
 'invalid_base_rule':'reset control and continuous state; toy convention only',
 'provider_family':plan.family,'initial_loss':losses[0],'final_loss':losses[-1],
 'interpretation':'Toy sequence effects feed a packed mathematical program and declared observation loss.'},indent=2))
