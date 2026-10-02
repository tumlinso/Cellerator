"""Ordinary Torch witnesses for two optional model families.

No CE kernel duplicates, no inference of biological identity. Native integration
belongs in CelleraTorch. These references do not silently choose native execution.
"""
from __future__ import annotations
import torch
from torch import nn

class MatrixPatchField(nn.Module):
    """One cell contains patches; patch is NOT a biological batch dimension."""
    def __init__(self,patches:int,width:int=16):
        super().__init__()
        if patches<=0 or width<=0: raise ValueError('positive shape required')
        scale=width**-0.5
        self.left=nn.Parameter(torch.randn(patches,width,width)*scale)
        self.right=nn.Parameter(torch.randn(patches,width,width)*scale)
        self.decay=nn.Parameter(torch.zeros(patches,1,1))
    def forward(self,state):
        if state.shape!=self.left.shape: raise ValueError('one-cell patch shape mismatch')
        return torch.tanh(self.left @ state) @ self.right-torch.nn.functional.softplus(self.decay)*state

class ActorPortField(nn.Module):
    """Local embeddings communicate through learned ports, not shared columns.

    Reference encoder maps can be arbitrary per actor. Choosing a palette/shared
    encoder for faster native contractions would be a separate model assumption.
    """
    def __init__(self,actors:int,width:int,ports:int,edges):
        super().__init__()
        if min(actors,width,ports)<=0: raise ValueError('positive dimensions required')
        src,dst=edges
        src=torch.as_tensor(src,dtype=torch.long);dst=torch.as_tensor(dst,dtype=torch.long)
        if src.shape!=dst.shape or src.ndim!=1 or bool(((src<0)|(src>=actors)|(dst<0)|(dst>=actors)).any()):
            raise ValueError('invalid declared edges')
        self.register_buffer('source',src);self.register_buffer('destination',dst)
        self.encoder=nn.Parameter(torch.randn(actors,width,ports)*width**-0.5)
        self.decoder=nn.Parameter(torch.randn(actors,ports,width)*ports**-0.5)
        self.local=nn.Parameter(torch.randn(actors,width,width)*width**-0.5)
        self.edge=nn.Parameter(torch.ones(src.numel())*0.1)
        self.readout=nn.Parameter(torch.randn(actors,width)*width**-0.5)
    def forward(self,state):
        if state.shape!=self.encoder.shape[:2]: raise ValueError('one-cell actor/slot shape mismatch')
        messages=torch.einsum('ih,ihp->ip',state,self.encoder)
        received=torch.zeros_like(messages).index_add(0,self.destination,self.edge[:,None]*messages[self.source])
        return torch.tanh(torch.einsum('ih,ihk->ik',state,self.local))-state+torch.einsum('ip,iph->ih',received,self.decoder)
    def observe(self,state):
        return (state*self.readout).sum(-1)

class RegrowthBranch(nn.Module):
    """A new branch with zero initial effect but a live output-weight gradient."""
    def __init__(self,input_width:int,capacity:int,output_width:int):
        super().__init__()
        self.inside=nn.Linear(input_width,capacity,bias=False)
        self.outside=nn.Linear(capacity,output_width,bias=False)
        nn.init.zeros_(self.outside.weight)
    def forward(self,state):
        return self.outside(torch.tanh(self.inside(state)))


class SylvesterFlow(nn.Module):
    """One-cell exact linear-flow family, not a general biological solver.

    Scalar state is arranged into square patches. Torch computes the matrix
    exponentials; a CE adapter may cache them only across unchanged L,R,dt.
    Parameter derivatives pass through matrix_exp, not a detached flow cache.
    """
    def __init__(self, width=16):
        super().__init__()
        self.left=nn.Parameter(-.1*torch.eye(width))
        self.right=nn.Parameter(-.1*torch.eye(width))
    def forward(self, x, dt):
        return torch.matrix_exp(dt*self.left) @ x @ torch.matrix_exp(dt*self.right)
