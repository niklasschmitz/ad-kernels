import logging
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

@partial(jax.jit, static_argnums=0)
def _kernelmatrix(basekernel, xs, xs2, kernel_kwargs={}):
    def matrixkernel(x1, x2):
        return jax.jacfwd(jax.grad(partial(basekernel, **kernel_kwargs)), argnums=1)(x1, x2)
    k_fn = vmap(vmap(matrixkernel, (None, 0)), (0, None))
    return k_fn(xs, xs2)


@dispatch
def dkernelmatrix(basekernel, xs, **kwargs):
    return dkernelmatrix(basekernel, xs, None, **kwargs)


@dispatch
def dkernelmatrix(basekernel, xs, xs2, *, batch_size=-1, batch_size2=-1, kernel_kwargs={},
                  flatten=True, store_on_device=True, **unused_kwargs):
    """Constructs an explicit gradient-gradient-kernelmatrix. Used by closed-form solver."""
    _kernelmatrix_checkpointed = jax.checkpoint(_kernelmatrix, static_argnums=0)
    if xs2 is None:
        xs2 = xs
    if batch_size == -1:
        matrix = _kernelmatrix(basekernel, xs, xs2, kernel_kwargs)
    elif batch_size2 == -1: # batching along rows only
        batch_indices = np.array(np.split(np.arange(len(xs)), len(xs) / batch_size))
        if store_on_device:
            device = xs.device()
            batch_indices = jax.device_put(batch_indices, device)
            matrix = jax.lax.map(
                lambda idx: jax.device_put(_kernelmatrix_checkpointed(basekernel, xs[idx], xs2, kernel_kwargs), device),
                batch_indices
            )
            matrix = matrix.reshape(matrix.shape[0]*matrix.shape[1], *matrix.shape[2:])
        else:
            device = jax.devices('cpu')[0]
            matrix = jnp.concatenate([
                jax.device_put(_kernelmatrix_checkpointed(basekernel, xs[idx], xs2, kernel_kwargs), device)
                for idx in batch_indices
            ])
    else: # batching along both rows and columns
        batch_indices1 = np.array(np.split(np.arange(len(xs)), len(xs) / batch_size))
        batch_indices2 = np.array(np.split(np.arange(len(xs2)), len(xs2) / batch_size2))
        if store_on_device:
            device = xs.device()
            batch_indices1 = jax.device_put(batch_indices1, device)
            batch_indices2 = jax.device_put(batch_indices2, device)
            matrix = jax.lax.map(
                lambda idx: (
                    jax.lax.map(
                        lambda idx2: jax.device_put(_kernelmatrix_checkpointed(basekernel, xs[idx], xs2[idx2], kernel_kwargs), device),
                        batch_indices2
                    )
                ),
                batch_indices1
            )
            matrix = matrix.swapaxes(1, 2)
            matrix = matrix.reshape(matrix.shape[0]*matrix.shape[1], matrix.shape[2]*matrix.shape[3], *matrix.shape[4:])
        else:
            device = jax.devices('cpu')[0]
            matrix = jnp.concatenate([
                jnp.concatenate([
                    jax.device_put(_kernelmatrix_checkpointed(basekernel, xs[idx], xs2[idx2], kernel_kwargs), device)
                    for idx2 in batch_indices2
                ], axis=1)
                for idx in batch_indices1
            ])
    if flatten:
        matrix = _flatten(matrix)
    return matrix


@dispatch
def dkernelmatrix(basekernel: KernelSum, xs1, xs2, *, verbose=True, **kwargs):
    if verbose: logging.info(f"[gdml_jax]: calculating kernelmatrices seperately for {basekernel}")
    return sum(jax.device_get(dkernelmatrix(kernel, xs1, xs2, **kwargs)) for kernel in basekernel.kernels)


@partial(jax.jit, static_argnames="descriptor")
def preaccumulate_descriptors_and_jacobians(descriptor, xs, descriptor_kwargs={}):
    descriptor_p = partial(descriptor, **descriptor_kwargs)
    descriptors = jax.vmap(descriptor_p)(xs)
    jacobians = jax.vmap(jax.jacfwd(descriptor_p))(xs)
    return descriptors, jacobians


@dispatch
def dkernelmatrix(basekernel: DescriptorKernel, xs1, xs2, *, verbose=True, kernel_kwargs={}, **kwargs):
    if verbose: logging.info(f"[gdml_jax]: preaccumulating descriptors and jacobians for {basekernel}")
    descriptor = basekernel.descriptor
    descriptor_kwargs = kernel_kwargs.get("descriptor_kwargs", {})
    kappa = basekernel.kappa
    kappa_kwargs = {key: val for (key, val) in kernel_kwargs.items() if key != "descriptor_kwargs"}
    phi_xs1, jacs_xs1 = preaccumulate_descriptors_and_jacobians(descriptor, xs1, descriptor_kwargs)
    if xs2 is not None:
        phi_xs2, jacs_xs2 = preaccumulate_descriptors_and_jacobians(descriptor, xs2, descriptor_kwargs)
    else:
        phi_xs2, jacs_xs2 = phi_xs1, jacs_xs1
    return dkernelmatrix_preaccumulated_batched(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, kernel_kwargs=kappa_kwargs, **kwargs)


def dkernelmatrix_preaccumulated_batched(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, *, batch_size=-1, batch_size2=-1, store_on_device=True, **kwargs):
    if batch_size == -1:
        return dkernelmatrix_preaccumulated(kappa, phi_xs1, phi_xs2, jacs_xs1, jacs_xs2, **kwargs)
    elif batch_size2 == -1: # batching along rows only
        device = phi_xs1.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices = np.split(np.arange(len(phi_xs1)), len(phi_xs1) / batch_size)
        return jnp.concatenate([
            jax.device_put(dkernelmatrix_preaccumulated(kappa, phi_xs1[idx], phi_xs2, jacs_xs1[idx], jacs_xs2, **kwargs), device)
            for idx in batch_indices
        ])
    else: # batching along both rows and columns
        device = phi_xs1.device() if store_on_device else jax.devices('cpu')[0]
        batch_indices1 = np.split(np.arange(len(phi_xs1)), len(phi_xs1) / batch_size)
        batch_indices2 = np.split(np.arange(len(phi_xs2)), len(phi_xs2) / batch_size2)
        return jnp.concatenate([
            jnp.concatenate([
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
    alpha = jax.scipy.linalg.solve(train_k, y, assume_a='pos')
    alpha = alpha.reshape(train_y.shape)
    return alpha


def solve_closed(basekernel, train_x, train_y, reg=1e-10, kernel_kwargs={}, verbose=True,
                 batch_size=-1, batch_size2=-1, store_on_device=None, solve_on_device=None):
    if store_on_device is None:
        store_on_device = (batch_size == -1)
    if solve_on_device is None:
        solve_on_device = store_on_device
    train_k = dkernelmatrix(
        basekernel, train_x, train_x,
        batch_size=batch_size, batch_size2=batch_size2, kernel_kwargs=kernel_kwargs,
        store_on_device=store_on_device, verbose=verbose,
    )
    if not solve_on_device:
        cpu_device = jax.devices("cpu")[0]
        train_k = jax.device_put(train_k, cpu_device)
        train_y = jax.device_put(train_y, cpu_device)
        reg = jax.device_put(reg, cpu_device)
    alpha = _solve_closed(train_k, train_y, reg)
    alpha = jax.device_put(alpha, train_x.device())
    params = dict(alpha=alpha, kernel_kwargs=kernel_kwargs)
    return params
