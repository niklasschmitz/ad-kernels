import jax.numpy as jnp
import numpy as np
from jax import vmap
from gdml_jax.kernels.simplekernels import rbf
from gdml_jax.kernels.composite import DescriptorKernel
from gdml_jax.kernels.atomistic import GlobalSymmetryKernel


def GDMLDescriptor(shape, *, metric_fn=None):
    """Returns a function that computes the pairwise inverse distance descriptor as used in GDML."""

    if metric_fn is None:
        metric_fn = lambda ri, rj: jnp.linalg.norm(ri - rj)

    pair_fn = lambda ri, rj: 1.0 / metric_fn(ri, rj)
    idx1, idx2 = np.triu_indices(shape[0], k=1)

    def descriptor(r):
        """(N,3) -> (N choose 2)"""
        return vmap(lambda i, j: pair_fn(r[i], r[j]))(idx1, idx2)
    
    return descriptor

def GDMLKernel(shape, *, metric_fn=None, kappa=rbf):
    """Returns a function that computes the scalar basekernel as used in GDML."""
    descriptor = GDMLDescriptor(shape, metric_fn=metric_fn)
    return DescriptorKernel(descriptor, kappa)

def sGDMLKernel(shape, *, metric_fn=None, kappa=rbf, perms=None):
    """Returns a function that computes the scalar basekernel as used in sGDML."""
    descriptor = GDMLDescriptor(shape, metric_fn=metric_fn)
    return GlobalSymmetryKernel(descriptor, kappa=kappa, perms=perms)
