"""Tiny single-cell rollout fitting example, not biological evidence or timing.

Use the existing compatible Torch installation. Native CE/CT substitution is
an integration task, not something this reference silently claims to perform.
"""
from __future__ import annotations
import argparse,json,torch
from actor_fields import MatrixPatchField,ActorPortField

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['patch','ports'],default='patch')
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.steps<=0: p.error('steps must be positive')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA unavailable')
    torch.manual_seed(11)
    make=(lambda:MatrixPatchField(2,4)) if a.family=='patch' else (lambda:ActorPortField(4,4,3,([0,1,2,3,0],[1,2,3,0,2])))
    teacher=make().to(a.device);student=make().to(a.device)
    x=torch.randn((2,4,4) if a.family=='patch' else (4,4),device=a.device)*.2
    def rollout(model):
        state=x
        history=[]
        for _ in range(4):
            state=state+.05*model(state);history.append(state)
        return torch.stack(history)
    with torch.no_grad():target=rollout(teacher)
    optimizer=torch.optim.Adam(student.parameters(),lr=.01)
    first=None
    for _ in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        loss=(rollout(student)-target).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite toy loss')
        if first is None:first=float(loss.detach())
        loss.backward();optimizer.step()
    with torch.no_grad():last=float((rollout(student)-target).square().mean())
    print(json.dumps({'family':a.family,'device':a.device,'steps':a.steps,'first_loss':first,'last_loss':last,'meaning':'synthetic engineering example; not a benchmark or biology result'},sort_keys=True))

if __name__=='__main__':main()
