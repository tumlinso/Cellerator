"""Direct native derivative checks, independent mathematical oracles."""
import unittest
import numpy as np
from patch.ops import forward, vjp, jvp, Tape, _lib

class PatchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(821)
        self.args = tuple((rng.normal(size=(3, 3)) * .25).astype(np.float32) for _ in range(3))
        self.directions = tuple(rng.normal(size=(3, 3)).astype(np.float32) for _ in range(3))
        self.g = rng.normal(size=(3, 3)).astype(np.float32)

    def test_fp32_forward(self):
        y, tape = forward(*self.args)
        x, l, r = (a.astype(np.float64) for a in self.args)
        np.testing.assert_allclose(y, np.tanh(l @ x) @ r, rtol=2e-6, atol=1e-7)
        self.assertEqual(y.dtype, np.float32)
        self.assertFalse(np.shares_memory(y, self.args[0]))

    def test_all_vjp_elements_finite_difference(self):
        _, tape = forward(*self.args)
        derivatives = vjp(tape, self.g)
        epsilon = .002
        for index, gradient in enumerate(derivatives):
            for cell in np.ndindex(gradient.shape):
                plus = [a.copy() for a in self.args]
                minus = [a.copy() for a in self.args]
                plus[index][cell] += epsilon
                minus[index][cell] -= epsilon
                yp, _ = forward(*plus)
                ym, _ = forward(*minus)
                fd = np.sum((yp-ym).astype(np.float64)*self.g)/(2*epsilon)
                self.assertAlmostEqual(float(gradient[cell]), fd, delta=2e-5)

    def test_jvp_finite_difference_and_adjoint(self):
        _, tape = forward(*self.args)
        tangent = jvp(tape, *self.directions)
        epsilon = .001
        plus = [a + epsilon*d for a, d in zip(self.args, self.directions)]
        minus = [a - epsilon*d for a, d in zip(self.args, self.directions)]
        yp, _ = forward(*plus); ym, _ = forward(*minus)
        np.testing.assert_allclose(tangent, (yp-ym)/(2*epsilon), rtol=3e-4, atol=2e-5)
        pullbacks = vjp(tape, self.g)
        self.assertAlmostEqual(float(np.sum(tangent*self.g)), sum(float(np.sum(a*b)) for a,b in zip(pullbacks,self.directions)), delta=2e-6)

    def test_half_surrogate_and_adjoint(self):
        y, tape = forward(*self.args, policy='stored_half_ste')
        x,l,r = (a.astype(np.float16).astype(np.float32) for a in self.args)
        t = l @ x
        v = np.tanh(t).astype(np.float16).astype(np.float32)
        np.testing.assert_allclose(y, v @ r, rtol=3e-6, atol=1e-7)
        s = (1-np.tanh(tape.t.astype(np.float64))**2)
        h = (self.g.astype(np.float64) @ r.astype(np.float64).T)*s
        expected = (l.T @ h, h @ x.T, tape.v.astype(np.float64).T @ self.g)
        gradients = vjp(tape, self.g)
        for actual, oracle in zip(gradients,expected):
            np.testing.assert_allclose(actual,oracle,rtol=3e-6,atol=1e-7)
        tangent = jvp(tape,*self.directions)
        self.assertAlmostEqual(float(np.sum(tangent*self.g)), sum(float(np.sum(a*b)) for a,b in zip(gradients,self.directions)), delta=2e-6)

    def test_trainable_zero_support(self):
        x = np.eye(3,dtype=np.float32)
        l = np.zeros_like(x); r = x.copy(); g = np.ones_like(x)
        y,tape = forward(x,l,r)
        dx,dl,dr = vjp(tape,g)
        np.testing.assert_array_equal(y,0)
        np.testing.assert_array_equal(dx,0)
        np.testing.assert_array_equal(dr,0)
        np.testing.assert_array_equal(dl,g)
        np.testing.assert_array_equal(jvp(tape,np.zeros_like(x),g,np.zeros_like(x)),g)

    def test_frozen_snapshot_generation_and_no_mutation(self):
        args = [a.copy() for a in self.args]
        _,tape = forward(*args,generations=(1,2,3,4))
        saved = [a.copy() for a in (tape.x,tape.l,tape.r,tape.t,tape.v)]
        gradients = vjp(tape,self.g)
        for a in args:
            a.fill(99)
        for got,want in zip(vjp(tape,self.g,current_generations=(1,2,3,4)),gradients):
            np.testing.assert_array_equal(got,want)
        jvp(tape,*self.directions)
        for actual,expected in zip((tape.x,tape.l,tape.r,tape.t,tape.v),saved):
            np.testing.assert_array_equal(actual,expected)
            with self.assertRaises(ValueError):
                actual.setflags(write=True)
        for function in (lambda: vjp(tape,self.g,current_generations=(1,9,3,4)),lambda: jvp(tape,*self.directions,current_generations=(2,2,3,4))):
            with self.assertRaises(ValueError):
                function()

    def test_public_tape_admission_and_snapshot(self):
        _, valid = forward(*self.args)
        values = dict(x=valid.x.copy(), l=valid.l.copy(), r=valid.r.copy(),
                      t=valid.t.copy(), v=valid.v.copy(), policy='fp32', generations=(0,0,0,0))
        for field in ('x','l','r','t','v'):
            for malformed in (np.zeros((2,2),dtype=np.float32), np.zeros((3,3),dtype=np.float64),
                              np.zeros((3,3),dtype=np.float32).T, np.full((3,3),np.nan,dtype=np.float32),
                              np.ndarray((3,3),dtype=np.float32,buffer=bytearray(37),offset=1)):
                case = dict(values); case[field] = malformed
                with self.assertRaises((ValueError,TypeError)):
                    Tape(**case)
        for field,bad in [('policy','unknown'),('generations',(1,2)),('generations',(0,-1,0,0))]:
            case = dict(values); case[field] = bad
            with self.assertRaises((ValueError,TypeError)):
                Tape(**case)
        rebuilt = Tape(**values)
        expected = rebuilt.x.copy()
        values['x'].fill(42)
        np.testing.assert_array_equal(rebuilt.x, expected)
        with self.assertRaises(ValueError):
            rebuilt.x.setflags(write=True)
        # Deliberate frozen-dataclass bypass still cannot submit bad extents.
        object.__setattr__(rebuilt,'l',np.zeros((1,1),dtype=np.float32))
        with self.assertRaises(ValueError):
            vjp(rebuilt,self.g)
        with self.assertRaises(ValueError):
            jvp(rebuilt,*self.directions)

    def test_private_native_basic_admission(self):
        import ctypes
        lib = _lib()
        null = ctypes.POINTER(ctypes.c_float)()
        for name,count in [('patch_forward',6),('patch_output',3),('patch_vjp',9),('patch_jvp',9)]:
            fn = getattr(lib,name)
            for n in (-1,0,46341,3):
                self.assertEqual(fn(n,*([null]*count)),1)

    def test_admission(self):
        x,l,r = self.args
        for bad in (x.astype(np.float64),x.T,x[:, :2],np.zeros((0,0),dtype=np.float32)):
            with self.assertRaises((ValueError,TypeError)):
                forward(bad,l,r)
        for bad in ('fp64','stored_half'):
            with self.assertRaises(ValueError):
                forward(x,l,r,policy=bad)
        with self.assertRaises(ValueError):
            forward(x,l,r,generations=(1,2,3))
        with self.assertRaises(ValueError):
            forward(np.full_like(x,1e9),l,r,policy='stored_half_ste')
        _,tape = forward(x,l,r)
        with self.assertRaises(ValueError):
            vjp(tape,np.zeros((2,2),dtype=np.float32))
        with self.assertRaises(ValueError):
            jvp(tape,np.zeros((2,2),dtype=np.float32),l,r)

if __name__ == '__main__':
    unittest.main()
