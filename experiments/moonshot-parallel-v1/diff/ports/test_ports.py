import unittest
import numpy as np
try:
    from . import ops
except ImportError:
    import ops

class PortsTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(19)
        self.widths=np.array([1,3,2],dtype=np.int64)
        self.src=np.array([0,1,1,2,0],dtype=np.int64)
        self.dst=np.array([1,2,2,0,1],dtype=np.int64)
        self.x=[self.rng.normal(size=s).astype(np.float32)*.3 for s in (6,12,12,5)]
        self.x[3][2]=0  # zero-valued trainable edge still has a derivative
    def forward(self,x):
        return ops.forward(*x,self.widths,self.src,self.dst,generations=(1,2,3,4))
    def reference(self,x):
        h,e,d,w=x; off=np.r_[0,np.cumsum(self.widths)]; p=2
        m=np.stack([e[p*off[i]:p*off[i+1]].reshape(p,hi)@h[off[i]:off[i+1]] for i,hi in enumerate(self.widths)])
        z=np.zeros_like(m)
        for src,dst,wt in zip(self.src,self.dst,w): z[dst]+=wt*m[src]
        return np.concatenate([d[p*off[i]:p*off[i+1]].reshape(hi,p)@z[i] for i,hi in enumerate(self.widths)])
    def test_forward_heterogeneous_repeated_edges(self):
        y,_=self.forward(self.x)
        np.testing.assert_allclose(y,self.reference(self.x),rtol=2e-6,atol=2e-8)
    def test_vjp_all_input_parameter_coordinates(self):
        _,t=self.forward(self.x); g=self.rng.normal(size=6).astype(np.float32)
        grads=ops.vjp(t,g,current_generations=(1,2,3,4)); eps=np.float32(.001)
        for block,(x,grad) in enumerate(zip(self.x,grads)):
            numeric=np.empty_like(x)
            for j in range(x.size):
                plus=[a.copy() for a in self.x]; minus=[a.copy() for a in self.x]
                plus[block][j]+=eps; minus[block][j]-=eps
                numeric[j]=np.dot(g,(self.forward(plus)[0]-self.forward(minus)[0]))/(2*eps)
            np.testing.assert_allclose(grad,numeric,rtol=.003,atol=2e-5)
        self.assertNotEqual(float(grads[3][2]),0)
    def test_jvp_finite_difference_and_adjoint(self):
        _,t=self.forward(self.x); tangents=[self.rng.normal(size=a.size).astype(np.float32) for a in self.x]
        dy=ops.jvp(t,*tangents); eps=.0005
        numerical=(self.forward([a+eps*b for a,b in zip(self.x,tangents)])[0]-self.forward([a-eps*b for a,b in zip(self.x,tangents)])[0])/(2*eps)
        np.testing.assert_allclose(dy,numerical,rtol=.003,atol=3e-5)
        g=self.rng.normal(size=6).astype(np.float32); grads=ops.vjp(t,g)
        np.testing.assert_allclose(np.dot(g,dy),sum(np.dot(a,b) for a,b in zip(grads,tangents)),rtol=3e-6,atol=1e-7)
    def test_saved_immutable_primal_generations_and_no_mutation(self):
        y,t=self.forward(self.x); saved=[a.copy() for a in self.x]
        for a in self.x: a[:]=99
        np.testing.assert_array_equal(y,self.forward(saved)[0])
        for a in (t.h,t.e,t.d,t.weights,t.widths,t.src,t.dst):
            self.assertFalse(a.flags.writeable)
            with self.assertRaises(ValueError): a.setflags(write=True)
        before=[a.tobytes() for a in (t.h,t.e,t.d,t.weights)]
        ops.vjp(t,np.ones(6,dtype=np.float32)); ops.jvp(t,*[np.zeros_like(a) for a in saved])
        self.assertEqual(before,[a.tobytes() for a in (t.h,t.e,t.d,t.weights)])
        with self.assertRaises(ValueError): ops.vjp(t,np.ones(6,dtype=np.float32),current_generations=(1,2,3,5))
        with self.assertRaises(ValueError): ops.jvp(t,*[np.zeros_like(a) for a in saved],current_generations=(0,2,3,4))
    def test_direct_tape_constructor_admission(self):
        from dataclasses import replace
        _,t=self.forward(self.x)
        malformed = [
            dict(h=t.h[:-1]), dict(e=t.e[:-1]), dict(d=t.d[:-1]),
            dict(weights=t.weights[:-1]), dict(src=t.src[:-1]),
            dict(dst=np.array([1,2,2,0,3],np.int64)),
            dict(src=np.array([-1,1,1,2,0],np.int64)),
            dict(widths=np.array([1,-3,2],np.int64)),
            dict(widths=np.array([],np.int64)), dict(widths=t.widths.astype(np.int32)),
            dict(h=t.h.astype(np.float64)), dict(e=t.e.reshape(2,6)),
            dict(p=0), dict(p=-1), dict(p=1.5), dict(p=True), dict(p=3),
            dict(p=2**63), dict(generations=(0,0,0)), dict(generations=(True,0,0,0)),
            dict(generations=(0,-1,0,0)),
        ]
        for bad in malformed:
            with self.subTest(bad=tuple(bad)):
                with self.assertRaises(ValueError): replace(t,**bad)
        direct=ops.Tape(*self.x,self.widths,self.src,self.dst,(1,2,3,4),2)
        before=direct.h.copy(); self.x[0][:]=77
        np.testing.assert_array_equal(direct.h,before)
        with self.assertRaises(ValueError): direct.h.setflags(write=True)
        self.assertEqual(ops._library()(7,0,0,0,*([None]*15)),2)

    def test_no_edges_and_validation(self):
        y,t=ops.forward(*self.x[:3],np.empty(0,np.float32),self.widths,np.empty(0,np.int64),np.empty(0,np.int64))
        np.testing.assert_array_equal(y,0)
        for grad in ops.vjp(t,np.ones(6,np.float32)): np.testing.assert_array_equal(grad,0)
        with self.assertRaises(ValueError): ops.forward(*self.x,self.widths,self.src,np.array([1,2,3,0,1],np.int64))
        with self.assertRaises(ValueError): ops.forward(self.x[0].astype(np.float64),*self.x[1:],self.widths,self.src,self.dst)

if __name__=='__main__': unittest.main(verbosity=2)
