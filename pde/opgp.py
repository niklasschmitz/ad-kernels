import jax
import jax.numpy as jnp
import jax.scipy.sparse
import jax.flatten_util
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)  # enable double precision (very important)

from collections import OrderedDict
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# define the generic operator kernel model

def kvp(k, L1, L2, x1, x2, alpha):
    return L1(lambda a: jnp.vdot(L2(lambda b: k(a, b))(x2), alpha))(x1)

# optionally define custom operator-specializing dispatches of kvp
# e.g. for grad operators to use forward-over-reverse AD instead of reverse-over-reverse

def make_mvm(k, operators1, operators2, xs1, xs2): # TODO clean up
    if isinstance(xs1, jnp.ndarray):
        xs1 = jax.tree_map(lambda _: xs1, operators1)
    if isinstance(xs2, jnp.ndarray):
        xs2 = jax.tree_map(lambda _: xs2, operators2)
    def _make_mvm(L1, L2, xsL1, xsL2):
        def _kvp(x1, x2, alpha):
            return kvp(k, L1, L2, x1, x2, alpha)
        def _mvm(alphas2):
            prods = vmap(vmap(_kvp, (None, 0, 0)), (0, None, None))(xsL1, xsL2, alphas2)
            return jnp.sum(prods, axis=1)
        return _mvm
    def mvm(alphas):
        return {
            key1: sum(_make_mvm(L1, L2, xs1[key1], xs2[key2])(alphas[key2]) for (key2, L2) in operators2.items())
            for (key1, L1) in operators1.items() # TODO consider jax.tree_map
        }
    return mvm

def build_solve(k, operators, *, solver="cg"):
    implementations = {
        "cg": build_solve_cg,
        "cholesky": build_solve_closed
    }
    return implementations[solver](k, operators)

def build_solve_cg(k, operators):
    def solve(x_train, y_train, reg=1e-10):
        G = make_mvm(k, operators, operators, x_train, x_train)
        G_reg = lambda alphas: tree_add(G(alphas), tree_scale(reg, alphas))
        alphas, _ = jax.scipy.sparse.linalg.cg(G_reg, y_train)
        return alphas
    return jax.jit(solve)

def build_predict(k, operators_test, operators_train):
    def predict(x_test, x_train, alphas):
        G_test = make_mvm(k, operators_test, operators_train, x_test, x_train)
        return G_test(alphas)
    return jax.jit(predict)

def build_predict_scalar(k, operators_train, x_train, alphas):
    def predict_scalar(x):
        x_test = jnp.expand_dims(x, axis=0)
        G_test = make_mvm(k, {"id": identity}, operators_train, x_test, x_train)
        res = G_test(alphas)
        y = res["id"][0]
        return y
    return jax.jit(predict_scalar)

# some pytree utils

def tree_add(x, y):
    return jax.tree_map(jnp.add, x, y)

def tree_scale(s, x):
    scale = jax.tree_util.Partial(jnp.multiply, s)
    return jax.tree_map(scale, x)

def tree_show(x, msg=""):
    print(msg, jax.tree_map(jnp.shape, x))
    return x

#===========================================#
# explicit kernelmatrix utils for debugging #

def build_solve_closed(k, operators):
    # CAUTION: jax.tree_utils do not keep dict key order
    # https://github.com/google/jax/issues/4085
    operators = OrderedDict(sorted(operators.items()))
    structure = jax.tree_structure(operators)

    def flatmatrix(arr, vshape):
        # arr.shape = (n1, n2, feats1..., feats2...)
        # vshape = (n2, feats2...)
        n_axes = len(arr.shape)
        arr = jnp.moveaxis(arr, 1, n_axes-len(vshape))
        shape = arr.shape
        a = int(np.prod(shape[:(n_axes-len(vshape))]))
        b = int(np.prod(shape[(n_axes-len(vshape)):]))
        return arr.reshape(a, b)

    def make_matrix(k, operators, xs, ys):
        def _make_matrix(L1, L2, xsL1, xsL2):
            def _matrixkernel(x1, x2, v):
                basis = jax._src.api._std_basis(v) # [v.size, v.shape...]
                m = vmap(partial(kvp, k, L1, L2, x1, x2))(basis) # [v.size, y1.shape...]
                m = jnp.moveaxis(m, 0, -1).reshape(*m.shape[1:], *v.shape) # [y1.shape..., v.shape...]
                return m
            def _matrix(vs):
                mm = vmap(vmap(_matrixkernel, (None, 0, 0)), (0, None, None))(xsL1, xsL2, vs)
                mm = flatmatrix(mm, vs.shape)
                return mm
            return _matrix
        matrix = {
            key1: {key2: _make_matrix(L1, L2, xs[key1], xs[key2])(ys[key2]) for (key2, L2) in operators.items()}
            for (key1, L1) in operators.items()
        }
        matrix = jnp.vstack([jnp.hstack(row.values()) for row in matrix.values()])
        return matrix

    def solve_closed(x, observations, *, reg=1e-10):
        if isinstance(x, jnp.ndarray):
            x = jax.tree_map(lambda _: x, operators)
        x = OrderedDict(sorted(x.items()))
        observations = OrderedDict(sorted(observations.items()))
        assert jax.tree_structure(x) == structure
        assert jax.tree_structure(observations) == structure
        matrix = make_matrix(k, operators, x, observations)
        matrix = matrix + reg*jnp.eye(len(matrix))
        y, unravel = jax.flatten_util.ravel_pytree(observations)
        alpha = jax.scipy.linalg.solve(matrix, y, sym_pos=True)
        alpha = unravel(alpha)
        return alpha
        
    return jax.jit(solve_closed)

#==================#
# downstream stuff #

# define base kernels

def rbf(x1, x2, lengthscale=1.0):
    return jnp.exp(-0.5 * jnp.sum((x1-x2)**2) / lengthscale**2)

# define operators

def identity(f):
    return f

def gradient(f):
    return jax.grad(f)

def hessian(f):
    return jax.hessian(f)

def make_jvp_operator(v):
    def jvp(f):
        return lambda x: jax.jvp(f, (x,), (v,))[1]
    return jvp

def compose(L2, L1):
    return lambda f: L2(L1(f))

def div(f):
    # f: R^n -> R^n
    def divf(x):
        return jnp.trace(jax.jacobian(f)(x))
    return divf

def laplacian(f):
    return div(gradient(f))

# other utilities for demo

def residual(u, x, operators, observations):
    mae = lambda a, b: jnp.mean(jnp.abs(a-b))
    preds = {key: jax.vmap(op(u))(x[key]) for key, op in operators.items()}
    return jax.tree_map(mae, observations, preds)

def make_data(
    f_true,
    n_samples=5,
    xlims=(-5.0, 5.0),
    ylims=(-5.0, 5.0),
    operators={"identity": identity},
    rng=jax.random.PRNGKey(0),
):
    rng1, rng2 = jax.random.split(rng)
    x1 = jax.random.uniform(rng1, (n_samples,), minval=xlims[0], maxval=xlims[1])
    x2 = jax.random.uniform(rng2, (n_samples,), minval=ylims[0], maxval=ylims[1])
    x = jnp.stack([x1, x2]).T
    observations = {key: jax.vmap(op(f_true))(x) for key, op in operators.items()}
    return x, observations

def plot_operators(f, operators):
    x = jnp.stack(jnp.meshgrid(jnp.linspace(-5, 5, 10), jnp.linspace(-5, 5, 10)), axis=-1).reshape(-1, 2)
    fig, axs = plt.subplots(ncols=len(operators), figsize=[len(operators)*5, 5], squeeze=False)
    axs = axs[0,:]
    for (ax, (op_key, op)) in zip(axs, operators.items()):
        ax.set_title(op_key)
        y = jax.vmap(op(f))(x) # (num_samples, obsdim1, obsim2, ...)
        obsdim = y.ndim - 1
        if obsdim == 0: # scalars
            ax.tricontourf(*x.T, y, levels=20)
        elif obsdim == 1: # vectors
            ax.quiver(*x.T, *y.T, width=0.005)
        elif obsdim == 2: # matrices
            ax.quiver(*x.T, *y[:,:,0].T, width=0.005)
            ax.quiver(*x.T, *y[:,:,1].T, width=0.005)
        else:
            print("no plot recipe for obsdim", obsdim)
    plt.close()
    return fig

def demo():
    k = rbf
    operators = {
        "identity": identity,
        "gradient": jax.grad,
        "hessian": jax.hessian,
        "jvp": make_jvp_operator(jnp.array([1.0, 2.0])),
        "jvpjvp": compose(
            make_jvp_operator(jnp.array([1.0, 2.0])), 
            make_jvp_operator(jnp.array([2.0, 3.0]))
        ),
        "hvp": compose(
            make_jvp_operator(jnp.array([1.0, 2.0])), 
            jax.grad
        )
    }

    def himmelblau(x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    x, observations = make_data(himmelblau, operators=operators)
    solve = build_solve(k, operators)
    predict = build_predict(k, operators, operators)
    alphas = solve(x, observations)
    p = predict(x, x, alphas)
    mse = lambda a, b: jnp.mean((a - b)**2)
    print("MSE: ", jax.tree_map(mse, p, observations))

if __name__ == '__main__':
    demo()
