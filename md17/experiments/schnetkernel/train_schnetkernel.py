import jax
import jax.numpy as jnp
from jax import lax
from jax.config import config
from functools import partial

from gdml_jax.solve import dkernelmatrix_preaccumulated_batched, _solve_closed
from gdml_jax import losses
from gdml_jax.util.datasets import get_symmetries

import argparse
import logging

# enable double precision
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description="SchNetKernel force field demo")
parser.add_argument("--jacs_train", type=str, required=True)
parser.add_argument("--jacs_test", type=str, required=True)
parser.add_argument("--lengthscale", type=float, required=True)
parser.add_argument("--molecule", type=str, default="ethanol")
parser.add_argument("--reg", type=float, default=1e-10)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--sym", type=eval, choices=[True, False], default=True)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--batch_size2", type=int, default=-1)
parser.add_argument("--loglevel", type=int, default=logging.INFO)
parser.add_argument("--logfile", type=str, default="")
args = parser.parse_args()

def config_logger(args):
    delim = "=============================="
    filename = args.logfile or f"{args.molecule}_train{args.n_train}_l{args.lengthscale}_reg{args.reg}"
    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{filename}.log"),
            logging.StreamHandler()
        ]
    )
    logging.info(delim)
    logging.info(args)
    logging.info(delim)
    logging.info(f"logging to {filename}.log")
    return filename
filename = config_logger(args)


trainset = jnp.load(args.jacs_train)
testset = jnp.load(args.jacs_test)

nn_train = jnp.array(trainset['features'])[:args.n_train]
y_train = jnp.array(trainset['y'])[:args.n_train]
jacs_train = jnp.array(trainset['jacs'])[:args.n_train]

nn_test = jnp.array(testset['features'])
y_test = jnp.array(testset['y'])
jacs_test = jnp.array(testset['jacs'])

def make_inner(kappa, xs1, xs2):
    def kvp(x1, x2, v2):
        g = lambda x2: jax.grad(kappa)(x1, x2)
        return jax.jvp(g, (x2,), (v2,))[1]
    def inner(alpha):
        return jax.vmap(jax.vmap(kvp, (None, 0, 0)), (0, None, None))(xs1, xs2, alpha)
    return inner

def make_mvm(kappa, xs1, xs2):
    inner = make_inner(kappa, xs1, xs2)
    def mvm(alpha):
        return jnp.sum(inner(alpha), axis=1)
    return mvm

@partial(jax.jit, static_argnums=(0,))
def mvm_nn(kappa, nn_xs1, nn_xs2, jacs1, jacs2, alpha, *, kappa_kwargs={}):
    kappa = partial(kappa, **kappa_kwargs)
    inner_mvm = make_mvm(kappa, nn_xs1, nn_xs2)
    alpha = jnp.einsum("abcde,ade->abc", jacs2, alpha)
    alpha = inner_mvm(alpha)
    alpha = jnp.einsum("abcde,abc->ade", jacs1, alpha)
    return alpha


def rbf(x1, x2, lengthscale=1.0): 
    return jnp.exp(-0.5*jnp.sum((x1-x2)**2) / lengthscale**2)

kappa = rbf

if args.sym:
    def symmetrize(kappa, perms):
        def kappasym(x1, x2, **kwargs):
            return jnp.sum(lax.map(lambda p: kappa(x1, x2[p], **kwargs), perms)) / len(perms)
        return kappasym

    perms = get_symmetries(args.molecule)
    rbf_sym = symmetrize(rbf, perms)
    kappa = rbf_sym

kappa_kwargs = {"lengthscale": args.lengthscale}

y = jax.device_get(y_train)
K = dkernelmatrix_preaccumulated_batched(
    kappa, nn_train, nn_train, jacs_train, jacs_train, 
    batch_size=args.batch_size, batch_size2=args.batch_size2, store_on_device=False, kernel_kwargs=kappa_kwargs
)
alpha = _solve_closed(K, y, args.reg)

preds_train = mvm_nn(kappa, nn_train, nn_train, jacs_train, jacs_train, alpha, kappa_kwargs=kappa_kwargs)
logging.info("forces:")
logging.info(f"train MSE: {losses.mse(y_train, preds_train)}")
logging.info(f"train MAE: {losses.mae(y_train, preds_train)}")

preds_test = mvm_nn(kappa, nn_test, nn_train, jacs_test, jacs_train, alpha, kappa_kwargs=kappa_kwargs)
logging.info(f"test MSE: {losses.mse(y_test, preds_test)}")
logging.info(f"test MAE: {losses.mae(y_test, preds_test)}")


