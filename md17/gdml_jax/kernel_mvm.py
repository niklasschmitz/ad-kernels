"""MVM computations for prediction in GDML models.

These functions are not necessarily user-facing, but instead used
by `GDMLPredict` and `GDMLPredictEnergy`, respectively.

Example:
>>> import jax.numpy as jnp
>>> from gdml_jax.kernel_mvm import kernel_mvm, kernel_mvm_integrate
>>> # Define some scalar differentiable kernel function.
>>> rbf = lambda x1, x2: jnp.exp(-jnp.sum(jnp.square(x1 - x2)))
>>> # Generate dummy input data
>>> batch_x1, batch_x2, alpha = jnp.ones((30, 5, 3)), jnp.ones((50, 5, 3)), jnp.ones((50, 5, 3))
>>> f = kernel_mvm(rbf, batch_x1, batch_x2, alpha)
>>> f.shape
(30, 5, 3)
>>> e = kernel_mvm_integrate(rbf, batch_x1, batch_x2, alpha)
>>> e.shape
(30,)
"""

import jax.numpy as jnp
from jax import grad, jvp
from jax import vmap, jit
from functools import partial
from plum import dispatch


@dispatch
@partial(jit, static_argnums=(0,))
def kernel_mvm(basekernel, batch_x1, batch_x2, alpha, *, kernel_kwargs={}):
    """Efficient derivative-kernelmatrix-vector-multiplication.

    outᵢ = ∑ⱼ ∇ₓᵢ k(xᵢ, xⱼ) ∇ₓⱼᵀαⱼ

    This is used for matrix-free prediction of derivatives (forces).
    It can also be used for iterative training procedures such as
    conjugate gradient methods.

    Args:
        basekernel: A function that produces a scalar kernel value,
            `basekernel(x1, x2)`. Here `x1` and `x2` are `jnp.ndarray`s
            of shape `input_shape`.
        batch_x1: Batch of (test) inputs. A `jnp.ndarray` of shape 
            `(M1,) + input_shape`
        batch_x2: Batch of (train) inputs. A `jnp.ndarray` of shape 
            `(M2,) + input_shape`
        alpha: Linear coefficients. A `jnp.ndarray` of shape 
            `(M2,) + input_shape`
        kernel_kwargs: A dict of keyword arguments to be passed to
            `basekernel(x1, x2, **kernel_kwargs)`.
    Returns:
        The result of shape `(M1,) + input_shape`
    """
    k_partial = partial(basekernel, **kernel_kwargs)

    def kvp(x1, x2, v2):
        # "forward over reverse AD" kernel vector product
        g = lambda x2: grad(k_partial)(x1, x2)
        return jvp(g, (x2,), (v2,))[1]

    def blockrow_vector_product(x):
        prods = vmap(kvp, (None, 0, 0))(x, batch_x2, alpha)
        return jnp.sum(prods, axis=0)

    return vmap(blockrow_vector_product)(batch_x1)

@dispatch
def kernel_mvm(basekernel, batch_x1, batch_x2, alpha, batch_size1, batch_size2, kernel_kwargs={}):
    indices1 = jnp.array(jnp.split(jnp.arange(len(batch_x1)), len(batch_x1) / batch_size1))
    indices2 = jnp.array(jnp.split(jnp.arange(len(batch_x2)), len(batch_x2) / batch_size2))
    def f(idx1):
        def g(idx2):
            return kernel_mvm(basekernel, batch_x1[idx1], batch_x2[idx2], alpha[idx2], kernel_kwargs=kernel_kwargs)
        prods = lax.map(g, indices2)
        return jnp.sum(prods, axis=0)
    return jnp.concatenate(lax.map(f, indices1))


@partial(jit, static_argnums=(0,))
def kernel_mvm_integrate(basekernel, batch_x1, batch_x2, alpha, kernel_kwargs={}):
    """Efficient gradient-kernelmatrix-vector-multiplication.
    
    outᵢ = ∑ⱼ k(xᵢ, xⱼ) ∇ₓⱼᵀαⱼ

    This is used for matrix-free prediction of integrated derivatives (energies).
    Note that the integration constant is estimated by `GDMLPredictEnergy`.

    Args:
        basekernel: A function that produces a scalar kernel value,
            `basekernel(x1, x2)`. Here `x1` and `x2` are `jnp.ndarray`s
            of shape `input_shape`.
        batch_x1: Batch of (test) inputs. A `jnp.ndarray` of shape 
            `(M1,) + input_shape`
        batch_x2: Batch of (train) inputs. A `jnp.ndarray` of shape 
            `(M2,) + input_shape`
        alpha: Linear coefficients. A `jnp.ndarray` of shape 
            `(M2,) + input_shape`
        kernel_kwargs: A dict of keyword arguments to be passed to
            `basekernel(x1, x2, **kernel_kwargs)`.
    Returns:
        The result of shape `(M1,)`
    """
    k_partial = partial(basekernel, **kernel_kwargs)

    def gvp(x1, x2, v2):
        # gradient vector product
        kx1 = lambda x2: k_partial(x1, x2)
        return jvp(kx1, (x2,), (v2,))[1]

    def blockrow_vector_product(x):
        grads = vmap(gvp, (None, 0, 0))(x, batch_x2, alpha)
        return jnp.sum(grads)

    return vmap(blockrow_vector_product)(batch_x1)
