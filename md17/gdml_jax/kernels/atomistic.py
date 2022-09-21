import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from functools import partial
from gdml_jax.kernels.composite import DescriptorKernel
from gdml_jax.kernels import rbf


def AtomicDoubleSumKernel(z, descriptor, k_atomic=rbf):
    """Returns a function that computes a scalar Kernel between two molecules.
    This is a general form of the GAP (also used in FCHL) kernels allowing 
    1) other descriptors and
    2) other atomic kernels ('k_atomic') instead of the Gaussian kernel.
    k(x1, x2) := Σ_i Σ_j (z[i] == z[j]) * k_atomic(descriptor(x1)[i], descriptor(x2)[j])
    
    Args:
        z: Array[int] representing nuclear charges.
        descriptor: A function that given atomic positions computes features
            of the form (N_atoms, N_atomfeatures...) and promised to be
            invariant w.r.t. to permutation of neighborhoods for each atom.
            Examples include ACSF, FCHL, SchNet.
        k_atomic: A scalar kernel function, comparing atomic features.
    """
    def kappa(d1, d2, **kwargs):
        k_partial = partial(k_atomic, **kwargs)
        allpairs = jax.vmap(jax.vmap(k_partial, (None, 0)), (0, None))

        # only compare same-species atoms
        res =  sum(
            jnp.sum(allpairs(d1[z == atom_type], d2[z == atom_type]))
            for  atom_type in np.unique(z)
        )
        return res

    return DescriptorKernel(descriptor, kappa)


def GlobalSymmetryKernel(descriptor, kappa, perms, is_atomwise=False):
    """Returns a function that computes a scalar kernel between two molecules.
    This is a generalization of the sGDML kernel allowing 
    1) other descriptors instead of inverse pairwise distances and
    2) other head kernels ('kappa') instead of Matern52.
    ksym(x1, x2) := 1 / |{P}| Σ_P kappa(descriptor(x1), descriptor(P x2))
    
    Args:
        descriptor: A function that computes features from atomic positions.
        kappa: A scalar head kernel function. The user needs to ensure kappa
            to be invariant w.r.t simultaneous permutation of the inputs, i.e.
            `kappa(P x1, P x2) == kappa(x1, x2)` for all P in perms.
        perms: An array of index permutations. See also `get_symmetries`.
        is_atomwise: Boolean, whether the descriptor is atom-wise, i.e.
            of the form (N_atoms, N_atomfeatures...) and promised to be
            invariant w.r.t. to permutation of neighborhoods for each atom.
            Examples include ACSF, FCHL, SchNet.
    """
    if is_atomwise:
        # special case for atom-wise feature descriptors: can compute descriptors once,
        # and just permute the result, instead of re-computing the descriptor.
        def kappasym(d1, d2, **kwargs):
            return jnp.sum(lax.map(lambda p: kappa(d1, d2[p], **kwargs), perms)) / len(perms)
        return DescriptorKernel(descriptor, kappasym)
    else:
        def basekernel(x1, x2, **kwargs):
            descriptor_kwargs = kwargs.get("descriptor_kwargs", {})
            kappa_kwargs = {key: val for (key, val) in kwargs.items() if key != "descriptor_kwargs"}
            descriptor_p = partial(descriptor, **descriptor_kwargs)
            d1 = descriptor_p(x1)
            return jnp.sum(lax.map(lambda p: kappa(d1, descriptor_p(x2[p]), **kappa_kwargs), perms)) / len(perms)
        return basekernel
