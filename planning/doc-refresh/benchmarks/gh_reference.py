#!/usr/bin/env python3
"""Analytical oracle only. This program does not call or benchmark GlassHelix."""
import json
import math

def expected():
    rows=[]
    for name,u,y in [('passive',0.0,0.0),('discriminating',1.0,1.0),('symmetric_control',1.0,0.0)]:
        predictions=[u,-u]
        logp=[math.log(0.5)-0.5*(y-p)**2 for p in predictions]
        m=max(logp)
        w=[math.exp(p-m) for p in logp]
        rows.append({'case':name,'input':u,'observed':y,'predictions':predictions,
                     'weights':[v/sum(w) for v in w]})
    return {'status':'reference_only_not_library_execution','rows':rows}

if __name__=='__main__':
    print(json.dumps(expected(),indent=2))
