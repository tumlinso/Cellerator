import dataclasses
import unittest
import numpy as np
from numpy.testing import assert_allclose
from is1.effects import AffineWitness, HybridAffine
from is1.operators import (quadratic,quadratic_jvp,quadratic_vjp,quadratic_delta,
                          QuadraticLedger,schur_solve,solve_jvp,multilevel_apply)
from is1.packing import (Problem,Process,Contribution,Plan,identity,opcode_cohorts,
                         support_cohorts,propose,execute,vjp,validate_plan,proxy)

class EffectTests(unittest.TestCase):
    def setUp(self): self.rng=np.random.default_rng(4721)
    def effect(self,n=3,m=2):
        return AffineWitness(self.rng.normal(size=(n,n)),self.rng.normal(size=n),
                             self.rng.normal(size=(m,n)),self.rng.normal(size=m))
    def hybrid(self,q=7,n=3):
        return HybridAffine(self.rng.integers(0,q,size=q),self.rng.normal(size=(q,n,n)),
                            self.rng.normal(size=(q,n)))
    def test_affine_direct_and_order(self):
        a,b=self.effect(),self.effect();h=self.rng.normal(size=3);q=self.rng.normal(size=2)
        for x,y in zip(a.then(b).apply(h,q),b.apply(*a.apply(h,q))):assert_allclose(x,y)
        self.assertFalse(np.allclose(a.then(b).A,b.then(a).A))
    def test_affine_associativity(self):
        a,b,c=self.effect(),self.effect(),self.effect()
        for name in ('A','b','C','d'):assert_allclose(getattr(a.then(b).then(c),name),getattr(a.then(b.then(c)),name))
    def test_affine_identity(self):
        a=self.effect();e=AffineWitness.identity(3,2)
        for name in ('A','b','C','d'):
            assert_allclose(getattr(e.then(a),name),getattr(a,name))
            assert_allclose(getattr(a.then(e),name),getattr(a,name))
    def test_affine_vjp_all_inputs(self):
        a=self.effect();h=self.rng.normal(size=3);q=self.rng.normal(size=2)
        u=self.rng.normal(size=3);v=self.rng.normal(size=2);g=a.vjp(h,u,v);eps=1e-6
        def objective(f,x,z):
            hh,qq=f.apply(x,z);return float(u@hh+v@qq)
        for name in ('A','b','C','d'):
            direction=self.rng.normal(size=getattr(a,name).shape)
            lo=dataclasses.replace(a,**{name:getattr(a,name)-eps*direction})
            hi=dataclasses.replace(a,**{name:getattr(a,name)+eps*direction})
            assert_allclose((objective(hi,h,q)-objective(lo,h,q))/(2*eps),np.sum(g[name]*direction),rtol=2e-7,atol=2e-7)
        for name,x in [('h',h),('q',q)]:
            d=self.rng.normal(size=x.shape)
            hi=objective(a,h+eps*d,q) if name=='h' else objective(a,h,q+eps*d)
            lo=objective(a,h-eps*d,q) if name=='h' else objective(a,h,q-eps*d)
            assert_allclose((hi-lo)/(2*eps),g[name]@d,rtol=2e-7,atol=2e-7)
    def test_affine_shape_and_nonfinite(self):
        with self.assertRaises(ValueError):AffineWitness(np.eye(3),np.zeros(2),np.zeros((1,3)),np.zeros(1))
        with self.assertRaises(ValueError):self.effect().apply([np.nan]*3,[0,0])
        with self.assertRaises(ValueError):self.effect().then(self.effect(2,2))
    def test_defensive_copy(self):
        A=np.eye(2);f=AffineWitness(A,np.zeros(2),np.zeros((0,2)),np.zeros(0));A[0,0]=5
        self.assertEqual(f.A[0,0],1)
        with self.assertRaises(ValueError):f.A[0,0]=2
    def test_hybrid_all_control_states(self):
        a,b=self.hybrid(),self.hybrid();h=self.rng.normal(size=3)
        for s in range(7):
            direct=b.apply(*a.apply(s,h));combined=a.then(b).apply(s,h)
            self.assertEqual(direct[0],combined[0]);assert_allclose(direct[1],combined[1])
    def test_hybrid_associative_and_identity(self):
        a,b,c=self.hybrid(),self.hybrid(),self.hybrid()
        x=a.then(b).then(c);y=a.then(b.then(c))
        np.testing.assert_array_equal(x.transition,y.transition)
        assert_allclose(x.A,y.A);assert_allclose(x.b,y.b)
        e=HybridAffine.identity(7,3)
        for x in (e.then(a),a.then(e)):
            np.testing.assert_array_equal(x.transition,a.transition);assert_allclose(x.A,a.A);assert_allclose(x.b,a.b)
    def test_scalar32_cuda_layout_contract_on_host(self):
        a,b=self.hybrid(32,1),self.hybrid(32,1)
        composed=a.then(b)
        t=a.transition
        np.testing.assert_array_equal(composed.transition,b.transition[t])
        assert_allclose(composed.A[:,0,0],b.A[t,0,0]*a.A[:,0,0])
        assert_allclose(composed.b[:,0],b.A[t,0,0]*a.b[:,0]+b.b[t,0])
    def test_hybrid_rejects_invalid_and_noninteger(self):
        with self.assertRaises(ValueError):HybridAffine(np.array([0.,1.]),np.zeros((2,1,1)),np.zeros((2,1)))
        with self.assertRaises(ValueError):HybridAffine(np.array([0,2]),np.zeros((2,1,1)),np.zeros((2,1)))
        with self.assertRaises(ValueError):self.hybrid().apply(-1,np.zeros(3))
        with self.assertRaises(ValueError):self.hybrid().apply(True,np.zeros(3))
        with self.assertRaises(ValueError):self.hybrid().then(self.hybrid(8,3))

class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(8732)
        self.X,self.L,self.R,self.M,self.D,self.G=[self.rng.normal(size=(4,4))*.2 for _ in range(6)]
    def test_exact_delta(self):
        assert_allclose(quadratic(self.X+self.D,self.L,self.R,self.M)-quadratic(self.X,self.L,self.R,self.M),quadratic_delta(self.X,self.D,self.L,self.R,self.M),atol=1e-14)
    def test_delta_second_order_term_needed(self):
        exact=quadratic_delta(self.X,self.D,self.L,self.R,self.M)
        linear=quadratic_jvp(self.X,self.L,self.R,self.M,self.D)
        assert_allclose(exact-linear,self.D@self.M@self.D,atol=1e-14)
        self.assertGreater(np.linalg.norm(exact-linear),1e-5)
    def test_rank_one_delta(self):
        u=self.rng.normal(size=4);v=self.rng.normal(size=4);D=np.outer(u,v)
        factored=self.L@D+D@self.R+np.outer(u,v@self.M@self.X)+np.outer(self.X@self.M@u,v)+np.outer(u,v)*(v@self.M@u)
        assert_allclose(factored,quadratic_delta(self.X,D,self.L,self.R,self.M),atol=1e-13)
    def test_jvp_finite_difference(self):
        e=1e-6
        fd=(quadratic(self.X+e*self.D,self.L,self.R,self.M)-quadratic(self.X-e*self.D,self.L,self.R,self.M))/(2*e)
        assert_allclose(fd,quadratic_jvp(self.X,self.L,self.R,self.M,self.D),atol=1e-10)
    def test_vjp_all_parameters(self):
        g=quadratic_vjp(self.X,self.L,self.R,self.M,self.G);e=1e-6
        args=[self.X,self.L,self.R,self.M]
        for i,name in enumerate(('X','L','R','M')):
            d=self.rng.normal(size=(4,4));a=args.copy();b=args.copy();a[i]=args[i]+e*d;b[i]=args[i]-e*d
            fd=np.sum(self.G*(quadratic(*a)-quadratic(*b)))/(2*e)
            assert_allclose(fd,np.sum(g[name]*d),atol=1e-10)
    def test_adjoint_identity(self):
        lhs=np.sum(self.G*quadratic_jvp(self.X,self.L,self.R,self.M,self.D))
        rhs=np.sum(self.D*quadratic_vjp(self.X,self.L,self.R,self.M,self.G)['X'])
        assert_allclose(lhs,rhs,atol=1e-14)
    def test_zero_primal_live_derivative(self):
        X=np.zeros((4,4));assert_allclose(quadratic(X,self.L,self.R,self.M),0)
        self.assertGreater(np.linalg.norm(quadratic_jvp(X,self.L,self.R,self.M,self.D)),1e-5)
    def test_ledger_remembers_untransmitted_changes(self):
        zero=np.zeros((2,2));L=np.eye(2);r=QuadraticLedger(zero,L,zero,zero)
        for k in (1,2,3):assert_allclose(r.update(np.full((2,2),.1*k),.35),0)
        assert_allclose(r.update(np.full((2,2),.4),.35),.4)
    def test_ledger_full_updates(self):
        r=QuadraticLedger(self.X,self.L,self.R,self.M)
        for _ in range(8):
            x=self.rng.normal(size=(4,4));assert_allclose(r.update(x),quadratic(x,self.L,self.R,self.M),atol=2e-14)
    def test_ledger_parameter_generation(self):
        r=QuadraticLedger(self.X,self.L,self.R,self.M)
        with self.assertRaises(ValueError):r.rebind_parameters(self.L*2,self.R,self.M,0,self.X)
        assert_allclose(r.rebind_parameters(self.L*2,self.R,self.M,1,self.X),quadratic(self.X,self.L*2,self.R,self.M))
        with self.assertRaises(ValueError):r.rebind_parameters(self.L,self.R,self.M,0,self.X)
    def test_invalid_parameter_rebind_is_atomic(self):
        r=QuadraticLedger(self.X,self.L,self.R,self.M)
        saved=r.value.copy()
        with self.assertRaises(ValueError):
            r.rebind_parameters(self.L*2,self.R,self.M,1,np.zeros((2,2)))
        self.assertEqual(r.parameter_epoch,0)
        assert_allclose(r.L,self.L);assert_allclose(r.sent,self.X);assert_allclose(r.value,saved)
    def test_port_solve_and_recovery(self):
        Q=self.rng.normal(size=(8,8));A=Q.T@Q+np.eye(8);b=self.rng.normal(size=8)
        result=schur_solve(A,b,[6,1,3]);assert_allclose(result['x'],np.linalg.solve(A,b),atol=1e-13)
        self.assertLess(result['residual_norm'],1e-12)
    def test_port_conditioning_and_indices(self):
        A=np.eye(3);A[2,2]=0
        with self.assertRaises(ValueError):schur_solve(A,np.ones(3),[0])
        with self.assertRaises(ValueError):schur_solve(np.eye(3),np.ones(3),[0,0])
        with self.assertRaises(ValueError):schur_solve(np.eye(3),np.ones(3),[0.0])
    def test_solve_derivative(self):
        Q=self.rng.normal(size=(5,5));A=Q.T@Q+np.eye(5);b=self.rng.normal(size=5)
        dA=self.rng.normal(size=(5,5));db=self.rng.normal(size=5);x=np.linalg.solve(A,b);e=1e-6
        fd=(np.linalg.solve(A+e*dA,b+e*db)-np.linalg.solve(A-e*dA,b-e*db))/(2*e)
        assert_allclose(fd,solve_jvp(A,x,dA,db),atol=1e-9)
    def test_multilevel_declared_map(self):
        D=self.rng.normal(size=(5,5));P=self.rng.normal(size=(5,2));R=self.rng.normal(size=(2,5));K=self.rng.normal(size=(2,2));x=self.rng.normal(size=5)
        assert_allclose(multilevel_apply(x,D,P,K,R),(D+P@K@R)@x,atol=1e-13)

class PackingTests(unittest.TestCase):
    def setUp(self):
        self.problem=Problem(4,3,3,(
            Process(9,'product',(0,0,2),0,(Contribution(0),Contribution(0,.5))),
            Process(2,'sum',(2,1),1,(Contribution(1),Contribution(2,-.5))),
            Process(3,'product',(0,3),0,(Contribution(2),)),
            Process(7,'product',(1,3),2,(Contribution(1),))))
        self.x=np.array([0.,2.,3.,4.]);self.theta=np.array([.5,-.2,.7])
    def custom(self,p,width):
        base=identity(p,width)
        return Plan(base.problem_signature,'user-reverse',tuple(reversed(base.state_order)),tuple((i,) for i in reversed(base.work_order)))
    def test_four_strategies_same_program(self):
        expected=execute(self.problem,identity(self.problem),self.x,self.theta)
        for strategy in (identity,opcode_cohorts,support_cohorts,self.custom):
            plan=propose(self.problem,strategy,2)
            assert_allclose(execute(self.problem,plan,self.x,self.theta),expected)
            self.assertEqual(proxy(self.problem,plan)['output_contributions'],6)
    def test_vjp_with_zero_and_repeated_arguments(self):
        rng=np.random.default_rng(65);G=np.array([.7,-.4,1.1]);e=1e-6
        for strategy in (identity,opcode_cohorts,support_cohorts,self.custom):
            plan=propose(self.problem,strategy,2);dx,dp=vjp(self.problem,plan,self.x,self.theta,G)
            d=rng.normal(size=4);p=rng.normal(size=3)
            fd=G@(execute(self.problem,plan,self.x+e*d,self.theta+e*p)-execute(self.problem,plan,self.x-e*d,self.theta-e*p))/(2*e)
            assert_allclose(fd,dx@d+dp@p,atol=1e-9)
            self.assertNotEqual(dx[0],0.)
    def test_repeated_argument_multiplicity_away_from_zero(self):
        p=Problem(1,1,1,(Process(1,'product',(0,0),0,(Contribution(0),)),));plan=identity(p)
        dx,dp=vjp(p,plan,[3.],[2.],[1.]);assert_allclose(dx,[12]);assert_allclose(dp,[9])
    def test_parameters_can_change_without_repacking(self):
        plan=identity(self.problem);assert_allclose(execute(self.problem,plan,self.x,2*self.theta),2*execute(self.problem,plan,self.x,self.theta))
    def test_stale_structure_rejected(self):
        plan=identity(self.problem)
        other=dataclasses.replace(self.problem,output_extent=4)
        with self.assertRaises(ValueError):validate_plan(other,plan)
    def test_ownership_and_permutation_corruption(self):
        plan=identity(self.problem)
        for bad in (dataclasses.replace(plan,groups=((9,2,3),)),
                    dataclasses.replace(plan,groups=((9,2,3,7,7),)),
                    dataclasses.replace(plan,state_order=(0,0,2,3))):
            with self.assertRaises(ValueError):validate_plan(self.problem,bad)
    def test_placement_ids_are_not_float_or_boolean(self):
        p=identity(self.problem)
        for order in ((0.,1.,2.,3.),(False,True,2,3)):
            with self.assertRaises(ValueError):
                validate_plan(self.problem,dataclasses.replace(p,state_order=order))
    def test_empty_and_constant_work(self):
        p=Problem(0,0,0,());self.assertEqual(execute(p,identity(p),[],[]).size,0)
        p=Problem(0,1,1,(Process(1,'product',(),0,(Contribution(0),)),))
        assert_allclose(execute(p,identity(p),[],[4.]),[4.])
    def test_bounds_invalid_width_and_nonfinite(self):
        with self.assertRaises(ValueError):identity(self.problem,0)
        with self.assertRaises(ValueError):execute(self.problem,identity(self.problem),[np.nan]*4,self.theta)
        with self.assertRaises(ValueError):Problem(1,1,1,(Process(1,'sum',(2,),0,(Contribution(0),)),)).validate()
        with self.assertRaises(ValueError):propose(self.problem,support_cohorts,-1)

if __name__=='__main__':unittest.main()
