"""GDML prediction models.

This module contains the `GDMLPredict` and `GDMLPredictEnergy` models,
which are essentially wrappers around efficient kernel MVM methods for
predicting forces and energies, given parameters. The parameters also
encopass the linear coefficients `alpha` obtained by some optimization 
procedure or a closed-form linear solve. Note that the solve itself is 
not part of this module.

For a full example see `examples/ethanol_example.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from gdml_jax.kernel_mvm import kernel_mvm, kernel_mvm_integrate


def GDMLPredict(basekernel, train_x, batch_size=-1):
    """Returns a function that predicts forces on input geometries.
    
    Note that the returned function closes over the kernel `basekernel` 
    and training inputs `train_x`, but not over `params`. This way it
    can also be used for iterative training.
    The optional `batch_size` can be used to evaluate expensive kernels 
    on memory-limited machines, trading off speed.

    Args:
        basekernel: A function that produces a scalar kernel value,
            `basekernel(x1, x2)`. Here `x1` and `x2` are `jnp.ndarray`s
            of shape `input_shape`.
        train_x: Training inputs. A `jnp.ndarray` of shape 
            `(M2,) + input_shape`
        batch_size: Integer. Should divide len(batch_x). (optional)
    Returns:
        A new function `predict_fn` that predicts forces on batches of
        new inputs `batch_x` given parameters `params`.
    """

    @jit
    def predict_fn(params, batch_x):
        alpha = params["alpha"]
        kernel_kwargs = params.get("kernel_kwargs", {})
        return kernel_mvm(basekernel, batch_x, train_x, alpha, kernel_kwargs=kernel_kwargs)

    if batch_size > 0:

        @jit
        def batched_predict_fn(params, batch_x):
            """Computes prediction serially in minibatches to save memory"""
            batch_indices = np.array(np.split(np.arange(len(batch_x)), len(batch_x) / batch_size))
            preds = jax.lax.map(lambda mb: predict_fn(params, batch_x[mb]), batch_indices)
            preds = preds.reshape(batch_x.shape)
            return preds

        return batched_predict_fn

    return predict_fn


def GDMLPredictEnergy(basekernel, train_x, train_e, params, batch_size=-1):

    kernel_kwargs = params.get("kernel_kwargs", {})

    # calculate integration constant c
    alpha = params["alpha"]
    if batch_size > 0:
        @jit
        def batched_integrate(alpha, train_x):
            batch_indices = np.array(np.split(np.arange(len(train_x)), len(train_x) / batch_size))
            train_e_preds = jax.lax.map(
                lambda mb: kernel_mvm_integrate(basekernel, train_x[mb], train_x, alpha, kernel_kwargs=kernel_kwargs), 
                batch_indices
            )
            train_e_preds = train_e_preds.reshape(-1)
            return train_e_preds
        train_e_preds = batched_integrate(alpha, train_x)
    else:
        train_e_preds = kernel_mvm_integrate(basekernel, train_x, train_x, alpha, kernel_kwargs=kernel_kwargs)
    c = jnp.mean(train_e + train_e_preds)

    @jit
    def energy_fn(batch_x):
        integrals = kernel_mvm_integrate(basekernel, batch_x, train_x, alpha, kernel_kwargs=kernel_kwargs)
        energies = -integrals + c
        return energies

    if batch_size > 0:

        @jit
        def batched_energy_fn(batch_x):
            """Computes prediction serially in minibatches to save memory"""
            batch_indices = np.array(np.split(np.arange(len(batch_x)), len(batch_x) / batch_size))
            preds = jax.lax.map(lambda mb: energy_fn(batch_x[mb]), batch_indices)
            preds = preds.reshape(-1)
            return preds

        return batched_energy_fn

    return energy_fn
