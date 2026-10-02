"""Small algebra checks, intentionally not a qualification campaign."""
import json
import unittest
from pathlib import Path
import numpy as np
from moonref.operators import *
from moonref.promotion import *
from moonref.packing import Footprint,pack,jaccard,minhash_candidates
from moonref.delta import DeltaLinear

class Algebra(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(123)
    def test_kronecker_single_vector(self):
        l=self.rng.normal(size=(4,4));x=self.rng.normal(size=(4,4));r=self.rng.normal(size=(4,4))
        np.testing.assert_allclose(matrix_patch(x,l,r).ravel(order='F'),kronecker_matrix(l,r)@x.ravel(order='F'),rtol=1e-12,atol=1e-12)
    def test_patch_derivatives(self):
        x,l,r,g=[self.rng.normal(size=(4,4)) for _ in range(4)]
        dx,dl,dr=[self.rng.normal(size=(4,4)) for _ in range(3)]
        vx,vl,vr=matrix_patch_vjp(x,l,r,g,True)
        j=matrix_patch_jvp(x,l,r,dx,dl,dr,True)
        self.assertAlmostEqual(float(np.sum(g*j)),float(np.sum(vx*dx)+np.sum(vl*dl)+np.sum(vr*dr)),places=10)
        eps=1e-6
        fd=(matrix_patch(x+eps*dx,l+eps*dl,r+eps*dr,True)-matrix_patch(x-eps*dx,l-eps*dl,r-eps*dr,True))/(2*eps)
        np.testing.assert_allclose(fd,j,rtol=2e-6,atol=2e-7)
    def test_local_bases_not_aligned(self):
        h=self.rng.normal(size=(3,4));e=self.rng.normal(size=(3,2,4));d=self.rng.normal(size=(3,4,2));a=self.rng.normal(size=(3,3))
        qs=np.stack([np.linalg.qr(self.rng.normal(size=(4,4)))[0] for _ in range(3)])
        hp,ep,dp=reframe_ports(h,e,d,qs)
        expected=np.einsum('ihk,ik->ih',qs,local_port_transport(h,e,d,a))
        np.testing.assert_allclose(local_port_transport(hp,ep,dp,a),expected,atol=1e-12)
    def test_zero_value_can_have_response(self):
        y,j,p=product_value_jvp([0,3],[1,0])
        self.assertEqual(y,0);self.assertEqual(j,3);np.testing.assert_equal(p,[3,0])
    def test_repeated_product_slots(self):
        y,j,p=product_value_jvp([2,2,3],[1,1,0])
        self.assertEqual(y,12);self.assertEqual(j,12)
    def test_quad_coordinate_bijection(self):
        seen=[set() for _ in range(4)]
        for lane in range(32):
            for i in range(8):
                g,r,c=quad_fragment_coordinates(lane,i)
                self.assertNotIn((r,c),seen[g]);seen[g].add((r,c))
        self.assertEqual([len(x) for x in seen],[64]*4)
    def test_coordinate_promotion_preserves_rollout(self):
        a=self.rng.normal(size=(4,4))*.1;o=self.rng.normal(size=(2,4))
        t=np.array([[1.,0,0,0],[0,0,1,0],[0,1,0,1],[0,1,0,-1]])
        b,read=conjugate_linear_model(a,o,t)
        s=self.rng.normal(size=4);q=t@s
        for _ in range(10):
            np.testing.assert_allclose(o@s,read@q,atol=1e-12)
            s=s+.01*(a@s);q=q+.01*(b@q)
    def test_quotient_known_equality(self):
        e=np.array([[1.,0,0],[0,0,1],[0,1,0],[0,0,1]])
        p=np.array([[1.,0,0,0],[0,0,1,0],[0,.5,0,.5]])
        g=self.rng.normal(size=(3,3));a=e@g@p;o=self.rng.normal(size=(2,4))
        new,read=quotient_linear_model(a,o,e,p)
        np.testing.assert_allclose(new,g,atol=1e-12)
        np.testing.assert_allclose(read,o@e)
        np.testing.assert_allclose(pullback_replicated_adjoint(e,[0,2,0,5]),[0,0,7])
        a[3,0]+=.5
        with self.assertRaises(ValueError):quotient_linear_model(a,o,e,p)
    def test_epoch_boundary(self):
        guard=EpochGuard();epoch=guard.acquire()
        with self.assertRaises(RuntimeError):guard.publish()
        guard.release(epoch);self.assertEqual(guard.publish(),2)
    def test_capacity_recycles_with_live_gradient(self):
        r=RecyclableResidual(4,3,2,seed=3);x=np.array([1.,2.,-1.,3.])
        np.testing.assert_equal(r.forward(x),[0,0])
        du,dv=r.vjp(x,np.array([1.,1.]))
        self.assertEqual(float(np.abs(du).sum()),0)
        self.assertGreater(float(np.abs(dv).sum()),0)
        r.m_v[:]=7;r.v_v[:]=9;r.recycle([1])
        np.testing.assert_equal(r.m_v[:,1],0);np.testing.assert_equal(r.v_v[:,1],0)
    def test_graph_placement_does_not_merge_identity(self):
        f=[Footprint('a0',frozenset([1,2]),frozenset([4])),Footprint('b3',frozenset([1,2]),frozenset([4]))]
        self.assertEqual(jaccard(f[0].reads,f[1].reads),1)
        self.assertEqual(minhash_candidates(f),[(0,1)])
        groups=pack(f)
        self.assertEqual(set(sum((list(x) for x in groups),[])),{'a0','b3'})
    def test_delta_accumulates_untransmitted_change(self):
        w=np.array([[1.,2.],[3.,4.]]);d=DeltaLinear(w)
        d.evaluate([0,0]);d.evaluate([.04,0],threshold=.1)
        y,bound=d.evaluate([.08,0],threshold=.1)
        np.testing.assert_equal(y,0)
        np.testing.assert_allclose(bound,[.08,.24])
        y,bound=d.evaluate([.12,0],threshold=.1)
        np.testing.assert_allclose(y,w@np.array([.12,0]))
        d.publish(w*2,2);y,_=d.evaluate([.12,0]);np.testing.assert_allclose(y,2*w@np.array([.12,0]))
    def test_low_rank_implicit(self):
        n,r=12,3
        d=self.rng.normal(size=n)*.1;u=self.rng.normal(size=(n,r))*.1;v=self.rng.normal(size=(n,r))*.1;b=self.rng.normal(size=(n,4));dt=.05
        expected=np.linalg.solve(np.eye(n)-dt*(np.diag(d)+u@v.T),b)
        np.testing.assert_allclose(low_rank_implicit_solve(d,u,v,b,dt),expected,rtol=1e-12,atol=1e-12)

    def test_quadratic_matrix_response(self):
        x,l,r,m,v,g=[self.rng.normal(size=(4,4)) for _ in range(6)]
        j=quadratic_matrix_jvp(x,l,r,m,v)
        vt=quadratic_matrix_vjp(x,l,r,m,g)
        self.assertAlmostEqual(float(np.sum(g*j)),float(np.sum(vt*v)),places=10)
        eps=1e-6
        fd=(quadratic_matrix_field(x+eps*v,l,r,m)-quadratic_matrix_field(x-eps*v,l,r,m))/(2*eps)
        np.testing.assert_allclose(fd,j,rtol=2e-6,atol=2e-7)
    def test_sylvester_flow_not_euler_steps(self):
        x=self.rng.normal(size=(4,4));l=np.array([-.1,-.2,-.3,-.4]);r=np.array([-.2,-.3,-.4,-.5]);t=.7
        el=np.diag(np.exp(t*l));er=np.diag(np.exp(t*r))
        expected=x*np.exp(t*(l[:,None]+r[None,:]))
        np.testing.assert_allclose(sylvester_flow(x,el,er),expected,atol=1e-12)
        half=sylvester_flow(x,np.diag(np.exp(t*l/2)),np.diag(np.exp(t*r/2)))
        np.testing.assert_allclose(sylvester_flow(half,np.diag(np.exp(t*l/2)),np.diag(np.exp(t*r/2))),expected,atol=1e-12)

if __name__=='__main__':unittest.main(verbosity=2)
