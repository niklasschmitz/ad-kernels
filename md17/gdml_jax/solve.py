import jax
import jax.numpy as jnp
import jax.scipy
from jax import vmap
import numpy as np
from functools import partial
from plum import dispatch
from gdml_jax.kernels.composite import DescriptorKernel, KernelSum


@jax.jit
def _flatten(k_nn):
    # k_nn: (n_xs, n_xs2, n_features..., n_features...)
    n_axes = len(k_nn.shape)
    k_nn = jnp.moveaxis(k_nn, 1, n_axes//2)
    shape = k_nn.shape
    a = np.prod(shape[:n_axes//2])
    b = np.prod(shape[n_axes//2:])
    return k_nn.reshape(a, b)


@partial(jax.jit, static_argnames="basekernel")
def _kernelmatrix(xs, xs2, kernel_kwargs={}, basekernel=None):
    def matrixkernel(x1, x2):
        return jax.jacfwd(jax.grad(partial(basekernel, **kernel_kwargs)), argnums=1)(x1, x2)
    k_fn = vmap(vmap(matrixkernel, (None, 0)), (0, None))
    return k_fn(xs, xs2)

@dispatch
def dkernelmatrix(basekernel, xs, **kwargs):
    return dkernelmatrix(basekernel, xs, None, **kwargs)

@dispatch
def dkernelmatrix(basekernel, xs, xs2, *, batch_size=-1, batch_size2=-1, kernel_kwargs={}, flatten=True, store_on_device=True):
    """Constructs an explicit gradient-gradient-kernelmatrix. Used by closed-form solver."""
    if xs2 is None:
        xs2 = xs
    if batch_size == -1:
        matrix = _kernelmatrix(xs, xs2, kernel_kwargs=kernel_kwargs, basekernel=basekernel)
    elif batch_size2 == -1: # batching along rows only
        device = xs.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices = np.split(np.arange(len(xs)), len(xs) / batch_size)
        matrix = jnp.concatenate([
            jax.device_put(_kernelmatrix(xs[idx], xs2, kernel_kwargs=kernel_kwargs, basekernel=basekernel), device) 
            for idx in batch_indices
        ])
    else: # batching along both rows and columns
        device = xs.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices1 = np.split(np.arange(len(xs)), len(xs) / batch_size)
        batch_indices2 = np.split(np.arange(len(xs2)), len(xs2) / batch_size2)
        matrix = jnp.concatenate([
            jnp.concatenate([
                jax.device_put(_kernelmatrix(xs[idx], xs2[idx2], kernel_kwargs=kernel_kwargs, basekernel=basekernel), device)
                for idx2 in batch_indices2
            ], axis=1) 
            for idx in batch_indices1
        ])
    if flatten:
        matrix = _flatten(matrix)
    return matrix


@dispatch
def dkernelmatrix(basekernel: KernelSum, xs1, xs2, **kwargs):
    print("calculating kernelmatrices seperately for", basekernel)
    return sum(dkernelmatrix(kernel, xs1, xs2, **kwargs) for kernel in basekernel.kernels)


@partial(jax.jit, static_argnames="descriptor")
def preaccumulate_descriptors_and_jacobians(descriptor, xs):
    descriptors = jax.vmap(descriptor)(xs)
    jacobians = jax.vmap(jax.jacfwd(descriptor))(xs)
    return descriptors, jacobians


@dispatch
def dkernelmatrix(basekernel: DescriptorKernel, xs1, xs2, **kwargs):
    print("preaccumulating descriptors and jacobians for", basekernel)
    descriptor = basekernel.descriptor
    kappa = basekernel.kappa
    phi_xs1, jacs_xs1 = preaccumulate_descriptors_and_jacobians(descriptor, xs1)
    if xs2 is not None:
        phi_xs2, jacs_xs2 = preaccumulate_descriptors_and_jacobians(descriptor, xs2)
    else:
        phi_xs2, jacs_xs2 = phi_xs1, jacs_xs1
    return  dkernelmatrix_preaccumulated_batched(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, **kwargs)


def dkernelmatrix_preaccumulated_batched(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, *, batch_size=-1, batch_size2=-1, store_on_device=True, **kwargs):
    if batch_size == -1:
        return dkernelmatrix_preaccumulated(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, **kwargs)
    elif batch_size2 == -1: # batching along rows only
        device = phi_xs1.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices = np.split(np.arange(len(phi_xs1)), len(phi_xs1) / batch_size)
        return np.concatenate([
            jax.device_put(dkernelmatrix_preaccumulated(kappa, phi_xs1[idx], phi_xs2, jacs_xs1[idx], jacs_xs2, **kwargs), device) 
            for idx in batch_indices
        ])
    else: # batching along both rows and columns
        device = phi_xs1.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices1 = np.split(np.arange(len(phi_xs1)), len(phi_xs1) / batch_size)
        batch_indices2 = np.split(np.arange(len(phi_xs2)), len(phi_xs2) / batch_size2)
        return np.concatenate([
            np.concatenate([
                jax.device_put(dkernelmatrix_preaccumulated(kappa, phi_xs1[idx], phi_xs2[idx2], jacs_xs1[idx], jacs_xs2[idx2], **kwargs), device) 
                for idx2 in batch_indices2
            ], axis=1)
            for idx in batch_indices1
        ])
        
        
@partial(jax.jit, static_argnames=("kappa", "flatten"))
def dkernelmatrix_preaccumulated(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, *, kernel_kwargs={}, flatten=True):
    # Perf optimization in the case of preaccumulated Jacobians.
    # These could also come from non-JAX code.
    # TODO clean up
    assert phi_xs1.shape[1:] == phi_xs2.shape[1:]
    assert phi_xs1.shape == jacs_xs1.shape[:len(phi_xs1.shape)]
    assert phi_xs2.shape == jacs_xs2.shape[:len(phi_xs2.shape)]
    kappa = partial(kappa, **kernel_kwargs)
    def kvp(x1, x2, v2):
        g = lambda x2: jax.grad(kappa)(x1, x2)
        return jax.jvp(g, (x2,), (v2,))[1]
    def inner(alpha):
        return vmap(vmap(kvp, (None, 0, 0)), (0, None, None))(phi_xs1, phi_xs2, alpha)
    def _recurse(f, x, i: int):
        # apply f i-times to x: f(f(f(...f(x))))
        return x if i <= 0 else _recurse(f, f(x), i-1)
    pshape = phi_xs2.shape
    jshape = jacs_xs2.shape
    KJ = _recurse(lambda f: vmap(f, (len(phi_xs2.shape),)), inner, len(jshape) - len(pshape))(jacs_xs2)
    if len(jacs_xs1.shape) == 4: # TODO generalize the pattern
        JKJ = jnp.einsum("abcd,ghaeb->aecdgh", jacs_xs1, KJ)
    else:
        JKJ = jnp.einsum("abicd,ghaebi->aecdgh", jacs_xs1, KJ)
    if flatten:
        JKJ = _flatten(JKJ)
    return JKJ


@jax.jit
def _solve_closed(train_k, train_y, reg):
    train_k = train_k.at[jnp.diag_indices_from(train_k)].add(reg)
    y = train_y.reshape(-1)
    alpha = jax.scipy.linalg.solve(train_k, y, sym_pos=True)
    alpha = alpha.reshape(train_y.shape)
    return alpha


def solve_closed(basekernel, train_x, train_y, reg=1e-10, batch_size=-1, batch_size2=-1, kernel_kwargs={}):
    train_k = dkernelmatrix(
        basekernel, train_x, train_x, 
        batch_size=batch_size, batch_size2=batch_size2, kernel_kwargs=kernel_kwargs,
        store_on_device=(batch_size == -1)
    )
    alpha = _solve_closed(train_k, train_y, reg)
    params = dict(alpha=alpha, kernel_kwargs=kernel_kwargs)
    return params