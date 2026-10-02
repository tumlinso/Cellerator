"""Fixed-weight delta accumulation witness with generation invalidation."""
import numpy as np
class DeltaLinear:
    def __init__(self,weights,generation=1):
        self.w=np.asarray(weights,dtype=np.float64).copy()
        self.generation=generation; self.sent=None; self.cached=None
    def publish(self,weights,generation):
        if generation<=self.generation: raise ValueError('generation must increase')
        w=np.asarray(weights,dtype=np.float64)
        if w.shape!=self.w.shape: raise ValueError('structural edit needs a new plan')
        self.w=w.copy(); self.generation=generation; self.sent=None; self.cached=None
    def evaluate(self,x,threshold=0.0):
        x=np.asarray(x,dtype=np.float64)
        if x.shape!=(self.w.shape[1],) or threshold<0 or not np.isfinite(x).all():
            raise ValueError('invalid state or threshold')
        if self.sent is None:
            self.sent=x.copy(); self.cached=self.w @ x
        else:
            delta=x-self.sent  # NEVER threshold only x[t]-x[t-1]
            changed=np.flatnonzero(np.abs(delta)>threshold)
            self.cached += self.w[:,changed] @ delta[changed]
            self.sent[changed]=x[changed]
        # Exact in real arithmetic for threshold=0, rounding accumulates in FP.
        # For threshold>0 this bounds the instantaneous omitted linear output.
        bound=np.abs(self.w) @ np.abs(x-self.sent)
        return self.cached.copy(),bound
