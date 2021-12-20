import jax
import jax.numpy as jnp
from jax.config import config
import numpy as np
from functools import partial

import argparse
import logging

# enable double precision
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description="SchNetKernel force field demo")
parser.add_argument("--molecule", type=str, default="ethanol")
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--name", type=str, default="schnet")
parser.add_argument("--loglevel", type=int, default=logging.INFO)
args = parser.parse_args()

CPU = jax.devices('cpu')[0]
LENGTHSCALES = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
REG_RANGE = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]

trainset = jnp.load(f"precompute/jacobians_{args.name}_{args.molecule}_train.npz")
testset  = jnp.load(f"precompute/jacobians_{args.name}_{args.molecule}_test.npz")

nn_train = jnp.array(trainset['features'])[:args.n_train]
y_train = jnp.array(trainset['y'])[:args.n_train]
jacs_train = jnp.array(trainset['jacs'])[:args.n_train]

nn_test = jnp.array(testset['features'])
y_test = jnp.array(testset['y'])
jacs_test = jnp.array(testset['jacs'])

logging.basicConfig(
    level=args.loglevel,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"kernelsearch_{args.name}_{args.molecule}_n{args.n_train}.log"),
        logging.StreamHandler()
    ],
    force=True
)
logging.info(f"representation: {args.name}")
logging.info(f"molecule: {args.molecule}")
logging.info(f"n_train: {args.n_train}")
logging.info(f"loglevel: {args.loglevel}")
logging.info(f"lengthscales: {LENGTHSCALES}")
logging.info(f"regularizations: {REG_RANGE}")

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

@partial(jax.jit, static_argnums=(0,))
def matrix(kappa, nn_xs1, nn_xs2, jacs1, jacs2, *, kappa_kwargs={}):
    kappa = partial(kappa, **kappa_kwargs)
    B1, _, _, N1, D1 = jacs1.shape
    B2, _, _, N2, D2 = jacs2.shape
    inner = make_inner(kappa, nn_xs1, nn_xs2)
    KJ = jax.vmap(jax.vmap(inner, (3,)), (3,))(jacs2)
    JKJ = jnp.einsum("abcde,fgaibc->adeifg", jacs1, KJ) # TODO clean up
    JKJ = JKJ.reshape(B1*N1*D1, B2*N2*D2)
    return JKJ

# @jax.jit
@partial(jax.jit, backend="cpu")
def psdsolve(K, y, reg):
    K = K.at[jnp.diag_indices_from(K)].add(reg)
    alpha = jax.scipy.linalg.solve(K, y, sym_pos=True)
    return alpha

# @partial(jax.jit, static_argnums=0, static_argnames="batch_size")
def batched_matrix(kappa, nn_xs1, jacs_xs1, nn_xs2, jacs_xs2, *, batch_size=None, batch_size2=None, kappa_kwargs):
    if batch_size is None:
        # K = jax.device_get(matrix(kappa, nn_xs1, nn_xs2, jacs_xs1, jacs_xs2).block_until_ready()) # pull to CPU
        K = jax.device_put(matrix(kappa, nn_xs1, nn_xs2, jacs_xs1, jacs_xs2, kappa_kwargs=kappa_kwargs).block_until_ready(), CPU)
    else:
        batch_indices = np.array(np.split(np.arange(len(nn_xs1)), len(nn_xs1) / batch_size))
        # K = jax.lax.map(lambda idx: jax.device_get(matrix(kappa, nn_xs1[idx], nn_xs2, jacs_xs1[idx], jacs_xs2)), batch_indices)
        # K = [jax.device_get(matrix(kappa, nn_xs1[idx], nn_xs2, jacs_xs1[idx], jacs_xs2).block_until_ready()) for idx in batch_indices]
        K = [batched_matrix(kappa, nn_xs2, jacs_xs2, nn_xs1[idx], jacs_xs1[idx], kappa_kwargs=kappa_kwargs, batch_size=batch_size2).T # caution: tricky # TODO undo 
            for idx in batch_indices]
        K = np.vstack(K)
    return K

def mae(y, yp):
    return jnp.mean(jnp.abs(y - yp))

def validate_preaccumulated(kappa, kappa_kwargs, nn_train, jacs_train, y_train, nn_val, jacs_val, y_val, *,
                            regs, batch_size=None, batch_size2=None):
    # temporary util function to work with fully precomputed features and pre-accumulated Jacobians (no explicit PyTorch anymore)
    y = y_train.reshape(-1)

    # TODO refactor (could re-use kernel submatrices across folds)
    K = batched_matrix(
        kappa, nn_train, jacs_train, nn_train, jacs_train, 
        batch_size=batch_size, batch_size2=batch_size2, kappa_kwargs=kappa_kwargs
    )

    losses = []
    for reg in regs:
        alpha = psdsolve(K, y, reg).reshape(y_train.shape)
        logging.info(f"reg {reg}")
        train_mae = mae(y_train, mvm_nn(kappa, nn_train, nn_train, jacs_train, jacs_train, alpha, kappa_kwargs=kappa_kwargs))
        logging.info(f"train MAE {train_mae}")
        val_mae = mae(y_val,  mvm_nn(kappa, nn_val, nn_train, jacs_val, jacs_train, alpha, kappa_kwargs=kappa_kwargs))
        logging.info(f"valid MAE {val_mae}")
        losses += [val_mae]
    return losses

def cross_validate(kappa, kappa_kwargs, nn_train, jacs_train, y_train, *,
                   regs=jnp.logspace(-10,-2,5), batch_size=None, batch_size2=None, n_folds=5):
    splits = np.array(np.split(np.arange(len(nn_train)), n_folds))
    cv_losses = []
    for train_idx in splits:
        val_idx  = np.delete(np.arange(len(nn_train)), train_idx)
        losses = validate_preaccumulated(
            kappa, kappa_kwargs, 
            nn_train[train_idx], jacs_train[train_idx], y_train[train_idx], 
            nn_train[val_idx], jacs_train[val_idx], y_train[val_idx],
            regs=regs, batch_size=batch_size, batch_size2=batch_size2
        )
        cv_losses += [losses]
    means = np.mean(cv_losses, axis=0)
    logging.info(f"kappa_kwargs={kappa_kwargs} [(reg, mean_mae)]={list(zip(regs, means))}")
    return means

def gridsearch(kappa, lengthscale_range, reg_range):
    losses = [
        cross_validate(kappa, {"lengthscale": lengthscale}, nn_train, jacs_train, y_train, regs=reg_range, batch_size=1)
        for lengthscale in lengthscale_range
    ]

    # select lengthscale and reg with smallest validation loss
    losses = np.array(losses)
    lengthscale_id, reg_id = np.unravel_index(np.nanargmin(losses), losses.shape)
    result = {
        "best_mae": np.nanmin(losses),
        "lengthscale": lengthscale_range[lengthscale_id],
        "reg": reg_range[reg_id],
    }
    return result


def rbf(x1, x2, lengthscale=1.0): 
    return jnp.exp(-0.5*jnp.sum((x1-x2)**2) / lengthscale**2)

from jax import lax
from gdml_jax.util.datasets import get_symmetries

def symmetrize(kappa, perms):
    def kappasym(x1, x2, **kwargs):
        return jnp.sum(lax.map(lambda p: kappa(x1, x2[p], **kwargs), perms)) / len(perms)
    return kappasym

perms = get_symmetries(args.molecule)
rbf_sym = symmetrize(rbf, perms)

for kappa in [rbf, rbf_sym]:
    logging.info(">< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< >< ><")
    logging.info(f"starting gridsearch for {kappa.__name__}")
    result = gridsearch(kappa, LENGTHSCALES, REG_RANGE)
    logging.info(f"gridsearch result: {result}")
    logging.info(f"retraining the best model on the whole training set...")
    validate_preaccumulated(
        kappa, {"lengthscale": result["lengthscale"]}, 
        nn_train, jacs_train, y_train, nn_test, jacs_test, y_test,
        regs=[result["reg"]], batch_size=1
    )
