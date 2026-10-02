"""Tiny CPU/autograd check only. No fitting experiment or biological claims."""
import torch
from actor_fields import MatrixPatchField,ActorPortField,RegrowthBranch

def run():
    torch.manual_seed(1)
    for model,x in [
      (MatrixPatchField(2,4).double(),torch.randn(2,4,4,dtype=torch.double)),
      (ActorPortField(3,4,2,([0,1,2],[1,2,0])).double(),torch.randn(3,4,dtype=torch.double))]:
        x.requires_grad_(True)
        y=model(x);y.square().sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in model.named_parameters() if n!='readout')
    r=RegrowthBranch(4,3,2).double();x=torch.tensor([1.,2.,-1.,3.],dtype=torch.double)
    assert torch.equal(r(x),torch.zeros(2,dtype=torch.double))
    r(x).sum().backward()
    assert r.outside.weight.grad.abs().sum()>0
    print('Torch forward/backward and live regrowth: passed (CPU only)')
if __name__=='__main__':run()

from actor_fields import SylvesterFlow
flow=SylvesterFlow(4)
x=torch.randn(4,4,requires_grad=True)
y=flow(x,.2)
y.square().sum().backward()
assert x.grad is not None and flow.left.grad is not None and flow.right.grad is not None
print("Torch exact linear-flow parameter gradients: passed (CPU)")
