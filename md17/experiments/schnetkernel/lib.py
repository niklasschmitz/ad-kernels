import jax
import jax.numpy as jnp
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.dlpack
import torch.utils.dlpack
import torch

import schnetpack as spk
import schnetpack.properties as properties

import functools
from functools import partial
from ase import Atoms
from ase.neighborlist import neighbor_list


# derivative kernel (jax) with nn features (torch) interop example

def make_inner(kappa, xs1, xs2):
    def kvp(x1, x2, v2):
        g = lambda x2: jax.grad(kappa)(x1, x2)
        return jax.jvp(g, (x2,), (v2,))[1]
    def inner(alpha):
        return jax.vmap(jax.vmap(kvp, (None, 0, 0)), (0, None, None))(xs1, xs2, alpha)
    return inner

def make_mvm(kappa, xs1, xs2):
    inner = make_inner(kappa, xs1, xs2)
    def mvm(alpha):
        return jnp.sum(inner(alpha), axis=1)
    return mvm

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

def make_mvm_nn(kappa, nn, xs1, xs2):
    nn_xs1 = torch_apply(nn, xs1)
    nn_xs2 = torch_apply(nn, xs2)
    @jax.jit # an inner function that doesn't close-over xs (to not capture too much memory with JIT)
    def _inner_mvm(nn_xs1, nn_xs2, alpha2):
        return make_mvm(kappa, nn_xs1, nn_xs2)(alpha2)
    def inner_mvm(alpha2):
        return _inner_mvm(nn_xs1, nn_xs2, alpha2)
    def mvm(alpha):
        nn_xs2, alpha2 = torch_jvp(nn, xs2, alpha)
        alpha3 = inner_mvm(alpha2)
        _, alpha4 = torch_vjp(nn, xs1, alpha3)
        return alpha4
    return mvm

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

# Explicit derivative kernelmatrix with preaccumulated Jacobians

@partial(jax.jit, static_argnums=(0,))
def matrix(kappa, nn_xs1, nn_xs2, jacs1, jacs2):
    B1, _, _, N1, D1 = jacs1.shape
    B2, _, _, N2, D2 = jacs2.shape
    inner = make_inner(kappa, nn_xs1, nn_xs2)
    KJ = jax.vmap(jax.vmap(inner, (3,)), (3,))(jacs2)
    JKJ = jnp.einsum("abcde,fgaibc->adeifg", jacs1, KJ) # TODO clean up
    JKJ = JKJ.reshape(B1*N1*D1, B2*N2*D2)
    return JKJ

def kernelmatrix_jacobian_preaccumulated(kappa, nn, xs1, xs2, *, jacs1=None, jacs2=None):
    nn_xs1 = torch_apply(nn, xs1)
    nn_xs2 = torch_apply(nn, xs2)
    if jacs1 is None: jacs1 = torch_jacobians(nn, xs1)
    if jacs2 is None: jacs2 = torch_jacobians(nn, xs2)
    return matrix(kappa, nn_xs1, nn_xs2, jacs1, jacs2)

@jax.jit
def psdsolve(K, y, reg):
    K = K.at[jnp.diag_indices_from(K)].add(reg)
    alpha = jax.scipy.linalg.solve(K, y, sym_pos=True)
    return alpha

def mae(y, yp):
    return jnp.mean(jnp.abs(y - yp))

def batched(f, batch_size):
    def batched_f(xs):
        batch_indices = jnp.split(jnp.arange(len(xs)), len(xs) / batch_size)
        return jnp.vstack([f(xs[idx]) for idx in batch_indices])
    return batched_f


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

def demo():

    def untrained_schnet(n_features=128, cutoff=5.0):
        radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
        representation = spk.representation.SchNet(
            n_atom_basis=n_features,
            n_interactions=3,
            radial_basis=radial_basis,
            cutoff_fn=spk.nn.CosineCutoff(cutoff),
        )
        return representation

        
    rbf = lambda x1, x2: jnp.exp(-0.5*jnp.sum((x1-x2)**2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cutoff = 5.0
    schnet = untrained_schnet(cutoff=cutoff).to(device, dtype=torch.float64)
    z = torch.ones(9, device=device, dtype=torch.int)

    def nn(xs):
        n_batch, n_atoms, n_dims = xs.shape
        batch = [schnet_input_dict(x, z, cutoff) for x in xs]
        batch = spk.data.loader._atoms_collate_fn(batch)
        return schnet(batch)['scalar_representation'].reshape(n_batch, n_atoms, -1)

    xs1   = torch2jax(torch.randn(5,9,3,device=device, dtype=torch.float64))
    xs2   = torch2jax(torch.randn(3,9,3,device=device, dtype=torch.float64))
    alpha = torch2jax(torch.randn(3,9,3,device=device, dtype=torch.float64))

    mvm = make_mvm_nn(rbf, nn, xs1, xs2)
    mvm_alpha = mvm(alpha)
    print(mvm_alpha)
    print(mvm_alpha.shape)
    print(type(mvm_alpha))

if __name__=='__main__':
    demo()
