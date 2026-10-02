"""Compare native-backed adapters against independent Torch expressions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from adapter import patch, process, ports, patch_jvp, process_jvp, ports_jvp

torch.set_num_threads(1)

def half_ste(t):
    return t + (t.half().float() - t).detach()

def patch_reference(x,l,r,policy='fp32'):
    if policy == 'stored_half_ste':
        x,l,r = [half_ste(t) for t in (x,l,r)]
        return half_ste(torch.tanh(l @ x)) @ r
    return torch.tanh(l @ x) @ r

def ports_reference(h,e,d,w,widths,src,dst):
    p=e.numel()//h.numel(); hs=h.split(widths)
    es=e.split([p*hi for hi in widths]); ds=d.split([p*hi for hi in widths])
    messages=[ei.reshape(p,hi) @ state for hi,ei,state in zip(widths,es,hs)]
    incoming=[sum((w[k]*messages[s] for k,(s,t) in enumerate(zip(src,dst)) if t==i),torch.zeros(p)) for i in range(len(widths))]
    return torch.cat([di.reshape(hi,p)@z for hi,di,z in zip(widths,ds,incoming)])

class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(52)
    def values(self,shapes):
        return [(torch.randn(shape)*.2).requires_grad_() for shape in shapes]
    def compare(self,actual,reference,values):
        copies=[v.detach().clone().requires_grad_() for v in values]
        before=[v.detach().clone() for v in values]
        y=actual(*values); expected=reference(*copies)
        torch.testing.assert_close(y,expected,rtol=3e-5,atol=2e-7)
        g=torch.randn_like(y)
        grads=torch.autograd.grad(y,values,g)
        expected_grads=torch.autograd.grad(expected,copies,g)
        for result,want,original,prior in zip(grads,expected_grads,values,before):
            torch.testing.assert_close(result,want,rtol=5e-5,atol=3e-7)
            torch.testing.assert_close(original,prior,rtol=0,atol=0)
        return grads
    def test_patch_fp32(self):
        self.compare(patch,patch_reference,self.values([(3,3)]*3))
    def test_patch_stored_half_ste(self):
        self.compare(lambda *x:patch(*x,policy='stored_half_ste'),
                     lambda *x:patch_reference(*x,policy='stored_half_ste'),self.values([(3,3)]*3))
    def test_product_repeated_and_zero(self):
        x,k=self.values([(4,),(4,)])
        with torch.no_grad(): x[0]=0; k[2]=0
        a=[0,1,2,1]; b=[1,1,3,2]
        grads=self.compare(lambda x,k:process(x,k,a,b),lambda x,k:k*x[a]*x[b],[x,k])
        self.assertNotEqual(grads[0][0].item(),0)
        self.assertNotEqual(grads[1][2].item(),0)
    def test_ports_heterogeneous_repeated_zero_edge(self):
        widths=[1,3,2]; src=[0,1,1,2,0]; dst=[1,2,2,0,1]
        values=self.values([(6,),(12,),(12,),(5,)])
        with torch.no_grad(): values[-1][2]=0
        grads=self.compare(lambda *x:ports(*x,widths,src,dst),
                           lambda *x:ports_reference(*x,widths,src,dst),values)
        self.assertNotEqual(grads[-1][2].item(),0)
    def test_selected_jvp_adjoint_all_routes(self):
        p=self.values([(3,3)]*3); product=self.values([(4,),(3,)])
        actor=self.values([(3,),(6,),(6,),(2,)])
        routes=[(p,lambda *x:patch(*x),lambda x,t:patch_jvp(*x,*t)),
                (p,lambda *x:patch(*x,policy='stored_half_ste'),lambda x,t:patch_jvp(*x,*t,policy='stored_half_ste')),
                (product,lambda *x:process(*x,[0,1,1],[1,1,2]),lambda x,t:process_jvp(*x,[0,1,1],[1,1,2],*t)),
                (actor,lambda *x:ports(*x,[1,2],[0,0],[1,1]),lambda x,t:ports_jvp(*x,[1,2],[0,0],[1,1],*t))]
        for values,f,j in routes:
            tangents=[torch.randn_like(v) for v in values]; y=f(*values); g=torch.randn_like(y)
            grads=torch.autograd.grad(y,values,g)
            torch.testing.assert_close((g*j(values,tangents)).sum(),sum((a*b).sum() for a,b in zip(grads,tangents)),rtol=1e-5,atol=1e-6)
    def test_saved_versions_reject_mutation_all_routes(self):
        routes=[(self.values([(2,2)]*3),lambda *x:patch(*x)),
                (self.values([(3,),(2,)]),lambda *x:process(*x,[0,1],[1,2])),
                (self.values([(3,),(6,),(6,),(1,)]),lambda *x:ports(*x,[1,2],[0],[1]))]
        for values,f in routes:
            for index in range(len(values)):
                y=f(*values)
                with torch.no_grad(): values[index].add_(1)
                with self.assertRaisesRegex(RuntimeError,'modified by an inplace operation'):
                    y.sum().backward()
    def test_optimizer_composition(self):
        x=torch.randn(2,2)*.2; l=torch.nn.Parameter(torch.eye(2)); r=torch.nn.Parameter(torch.eye(2))
        optimizer=torch.optim.SGD([l,r],lr=.3); losses=[]
        for _ in range(30):
            optimizer.zero_grad(); loss=patch(x,l,r).square().mean(); losses.append(loss.item())
            loss.backward(); optimizer.step()
        self.assertLess(losses[-1],losses[0]*.8)
    def test_support_rejections(self):
        x=torch.ones(2,2)
        with self.assertRaisesRegex(ValueError,'CPU float32'): patch(x.double(),x,x)
        with self.assertRaisesRegex(ValueError,'contiguous'): patch(x.T,x,x)
        with self.assertRaisesRegex(ValueError,'integer'): process(torch.ones(2),torch.ones(1),[0.],[1])
        with self.assertRaisesRegex(ValueError,'CPU float32'): patch(x.to('meta'),x,x)
    def test_higher_order_is_unsupported(self):
        x,l,r=self.values([(2,2)]*3)
        dx,=torch.autograd.grad(patch(x,l,r).sum(),x,create_graph=True)
        self.assertFalse(dx.requires_grad)
        with self.assertRaises(RuntimeError): torch.autograd.grad(dx.sum(),x)

if __name__ == '__main__': unittest.main(verbosity=2)
