import jax.numpy as jnp
from jax import custom_jvp

# make matern52 jax-differentiable at 0

@custom_jvp
def _matern(r):
  # r: xi - xj
  d2 = jnp.dot(r, r)
  d = jnp.sqrt(d2)
  return (1.0+jnp.sqrt(5)*d+5*d2/3)*jnp.exp(-jnp.sqrt(5)*d)
  
@_matern.defjvp
def _matern_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  g = _grad_matern(x)
  return _matern(x), jnp.dot(g, x_dot)

@custom_jvp
def _grad_matern(x):
  # this is needed to predict energies (but not for forces)
  d = jnp.sqrt(jnp.dot(x, x))
  return x * (-5*(d*jnp.sqrt(5)+1)/3 * jnp.exp(-jnp.sqrt(5)*d))

@_grad_matern.defjvp
def _grad_matern_jvp(primals, tangents):
  r, = primals
  r_dot, = tangents
  d = jnp.sqrt(jnp.dot(r, r))
  s = 5. / 3. * jnp.exp(-jnp.sqrt(5)*d)
  res = 5.0 * jnp.dot(r, jnp.dot(r, r_dot))
  res -= (1.0 + jnp.sqrt(5) * d) * r_dot
  res *= s
  return _grad_matern(r), res


# exported:

def matern52(xi, xj, lengthscale=1.0):
  return _matern((xi - xj) / lengthscale)

