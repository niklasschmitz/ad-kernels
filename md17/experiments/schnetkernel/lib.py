import jax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.dlpack
import torch.utils.dlpack
import torch

import schnetpack.properties as properties

import functools
from ase import Atoms
from ase.neighborlist import neighbor_list


# JAX - PyTorch AD interop

def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch2jax(x):
    # https://github.com/google/jax/issues/7657
    # return jax.device_put(x.numpy()) # this copies to host and back to device, safe but slow
    # TODO remove .contiguous() once fixed, to avoid potential copy
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))

def torch_apply(nn, x):
    x = jax2torch(x)
    nnx = nn(x)
    nnx = torch2jax(nnx)
    return nnx

def torch_jvp(nn, x, v):
    x = jax2torch(x)
    v = jax2torch(v)
    nnx, nnv = torch.autograd.functional.jvp(nn, x, v)
    nnx = torch2jax(nnx)
    nnv = torch2jax(nnv)
    return nnx, nnv

def torch_vjp(nn, x, v):
    x = jax2torch(x)
    v = jax2torch(v)
    nnx, nnv = torch.autograd.functional.vjp(nn, x, v)
    nnx = torch2jax(nnx)
    nnv = torch2jax(nnv)
    return nnx, nnv


# Jacobian preaccumulation
  
def batch_jacobian(f, xbatch):
    # 'trick' to compute all jacobians in a batch jointly
    # TODO use vmap once stable
    B, *in_shape = xbatch.shape
    fsum = lambda x: f(x).sum(axis=0)
    jacs = torch.autograd.functional.jacobian(fsum, xbatch)
    out_shape = f(xbatch).shape
    jacs = torch.moveaxis(jacs, len(out_shape) - 1, 0)
    assert jacs.shape == (*out_shape, *in_shape)
    return jacs # (batch, n_atoms, n_features, n_atoms, 3)
  
def torch_jacobians(nn, x):
    x = jax2torch(x)
    jacs = batch_jacobian(nn, x)
    jacs = torch2jax(jacs)
    return jacs


# SchNet inputs

@torch.no_grad()
@functools.lru_cache(maxsize=1000) # compute training set neighbor lists once
def _neighbor_list(r, cutoff, device):
    positions = r.cpu().numpy()
    idx_i, idx_j = neighbor_list("ij", Atoms(positions=positions), cutoff=cutoff)
    idx_i = torch.tensor(idx_i, device=device)
    idx_j = torch.tensor(idx_j, device=device)
    return idx_i, idx_j

def schnet_input_dict(r, z, cutoff): # dev-brach schnetpack@83935fb5001cacea1b414c33fb54b6a94eb74893
    idx_i, idx_j = _neighbor_list(r, cutoff, r.device)
    n_atoms, n_dims = r.shape
    return {
        properties.Z: z,
        properties.Rij: r[idx_i] - r[idx_j],
        properties.idx_i: idx_i,
        properties.idx_j: idx_j,
        properties.n_atoms: torch.tensor([n_atoms])
    }
