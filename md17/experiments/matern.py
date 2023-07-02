import jax
import jax.numpy as jnp
from jax import custom_jvp

# make matern52 jax-differentiable at 0

@custom_jvp
def _matern(r):
  # r: xi - xj
  d2 = jnp.vdot(r, r)
  d = jnp.sqrt(d2)
  return (1.0+jnp.sqrt(5)*d+5*d2/3)*jnp.exp(-jnp.sqrt(5)*d)
  
@_matern.defjvp
def _matern_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  g = _grad_matern(x)
  return _matern(x), jnp.vdot(g, x_dot)

@custom_jvp
def _grad_matern(x):
  # this is needed to predict energies (but not for forces)
  d = jnp.sqrt(jnp.vdot(x, x))
  return x * (-5*(d*jnp.sqrt(5)+1)/3 * jnp.exp(-jnp.sqrt(5)*d))

@_grad_matern.defjvp
def _grad_matern_jvp(primals, tangents):
  r, = primals
  r_dot, = tangents
  d = jnp.sqrt(jnp.vdot(r, r))
  s = 5. / 3. * jnp.exp(-jnp.sqrt(5)*d)
  res = 5.0 * r * jnp.vdot(r, r_dot)
  res -= (1.0 + jnp.sqrt(5) * d) * r_dot
  res *= s
  return _grad_matern(r), res


# exported:

def matern52(x1, x2, lengthscale=1.0):
  return _matern((x1 - x2) / lengthscale)

def matern52tp(x1, x2, lengthscale=1.0):
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    return jnp.prod(jax.vmap(lambda x1, x2: matern52(x1, x2, lengthscale))(x1, x2))
