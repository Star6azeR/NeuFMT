import math
import numpy as np
import torch
import time


def positional_encoding_default(x: torch.Tensor, L=4) -> torch.Tensor:
    """
    Maps coordinates [x, y, z] in 3-dim to
    [sin(2^i*x), sin(2^i*y), sin(2^i*z), cos(2^i*x), cos(2^i*y), cos(2^i*y)] (i: 0->L) in (6*L)-dim
    """
    pe = []
    for i in range(L):
        pe.append(torch.sin(2.**i * x))
        pe.append(torch.cos(2.**i * x))
    return torch.cat(pe, dim=1)


def positional_encoding_altz(p, L, Lz) -> torch.Tensor:
    """
    Should return same result as defalut.\n
    Maps coordinates [x, y, z] in 3-dim to
    [sin(2^i*x), sin(2^i*y), sin(2^i*z), cos(2^i*x), cos(2^i*y), cos(2^i*y)] (i: 0->L) in (6*L)-dim
    """
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).cuda()
    logseq_z = torch.logspace(start=0, end=Lz-1, steps=Lz, base=2).cuda()

    xsin = torch.sin((logseq*math.pi).reshape([1,-1]) * p[:,0].reshape([-1, 1]))
    ysin = torch.sin((logseq*math.pi).reshape([1,-1]) * p[:,1].reshape([-1, 1]))
    zsin = torch.sin((logseq_z*math.pi).reshape([1,-1]) * p[:,2].reshape([-1, 1]))
    xcos = torch.cos((logseq*math.pi).reshape([1,-1]) * p[:,0].reshape([-1, 1]))
    ycos = torch.cos((logseq*math.pi).reshape([1,-1]) * p[:,1].reshape([-1, 1]))
    zcos = torch.cos((logseq_z*math.pi).reshape([1,-1]) * p[:,2].reshape([-1, 1]))

    return torch.cat((xsin,xcos,ysin,ycos,zsin,zcos), dim=1)