import jax.numpy as jnp


def mse(y, yp):
    return jnp.mean((y - yp) ** 2)


def mae(y, yp):
    return jnp.mean(jnp.abs(y - yp))
