import jax.numpy as jnp


def rbf(x1, x2, lengthscale=1.0):
    return jnp.exp(-0.5 * jnp.sum((x1 - x2) ** 2) / lengthscale**2)
