import argparse
import logging
import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
from gdml_jax.util.datasets import load_md17, get_symmetries
from gdml_jax.kernels import rbf, DescriptorKernel, GlobalSymmetryKernel
from gdml_jax.models import GDMLPredict, GDMLPredictEnergy
from gdml_jax.solve import solve_closed
from gdml_jax import losses
from distutils.util import strtobool
import optax

# enable double precision
config.update("jax_enable_x64", True)


def print_callback(i, loss, params):
    logging.info(f"step={i:4d} loss={loss:.8f} {jax.tree_map(lambda p: f'{p:.5f}', params)}")

def fit(loss_fn, params, optimizer, steps, cb=print_callback):

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss
    
    opt_state = optimizer.init(params)
    for i in range(steps):
        params, opt_state, loss = train_step(params, opt_state)
        cb(i, loss, params)
    
    return params

def powered_coulomb_descriptor(r, power=1.0):
    """(N,3) -> (N choose 2)"""
    n_atoms, n_dim = r.shape
    idx1, idx2 = np.triu_indices(n_atoms, k=1)
    pair_fn = lambda ri, rj: 1.0 / jnp.linalg.norm(ri - rj) ** power
    return jax.vmap(lambda i, j: pair_fn(r[i], r[j]))(idx1, idx2)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="GDML-JAX MD17 hyper example")
    parser.add_argument("--lengthscale_init", type=jnp.float64)
    parser.add_argument("--power_init", type=jnp.float64, default=jnp.float64(1.0))
    parser.add_argument("--reg", type=jnp.float64, default=jnp.float64(1e-10))
    parser.add_argument("--molecule", type=str, default="ethanol")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_test", type=int, default=2000)
    parser.add_argument("--validation_split", type=jnp.float64, default=jnp.float64(0.8))
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--batch_size2", type=int, default=-1)
    parser.add_argument("--store_on_device", type=lambda x: bool(strtobool(x)), nargs='?', const=True)
    parser.add_argument("--solve_on_device", type=lambda x: bool(strtobool(x)), nargs='?', const=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step_size", type=jnp.float64, default=jnp.float64(1e-2))
    parser.add_argument("--datadir", type=str, default="data/train")
    parser.add_argument("--loglevel", type=int, default=logging.INFO)
    parser.add_argument("--logfile", type=str, default="")
    args = parser.parse_args()

    filename = args.logfile or f"hyper_{args.molecule}_train{args.n_train}_l{args.lengthscale_init}_reg{args.reg}"
    logging.basicConfig(
        level=args.loglevel,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{filename}.log"),
            logging.StreamHandler()
        ],
        force=True,
    )
    logging.info(args)

    perms = get_symmetries(args.molecule)
    basekernel = GlobalSymmetryKernel(powered_coulomb_descriptor, kappa=rbf, perms=perms)

    # data loading
    trainset, testset, meta = load_md17(args.molecule, args.n_train, args.n_test, args.datadir)
    train_x, train_e, train_y = trainset
    n_atoms, _ = meta["shape"]

    # split train data into train and validation part
    split = int(np.floor(args.validation_split * args.n_train))
    train_x, val_x = train_x[:split], train_x[split:]
    train_y, val_y = train_y[:split], train_y[split:]

    # define hyperparameter loss
    def loss_from_kernel(basekernel, kernel_kwargs, reg=args.reg):
        params = solve_closed(basekernel, train_x, train_y, reg=reg, kernel_kwargs=kernel_kwargs, 
                              batch_size=args.batch_size, batch_size2=args.batch_size2, verbose=False,
                              store_on_device=args.store_on_device, solve_on_device=args.solve_on_device)
        force_fn = GDMLPredict(basekernel, train_x)
        preds_y = force_fn(params, val_x)
        return losses.mae(val_y, preds_y)

    def loss_fn(params):
        return loss_from_kernel(basekernel, params)

    # fit hyperparameters
    initial_params = {
        "lengthscale": args.lengthscale_init or jnp.float64(n_atoms), 
        "descriptor_kwargs": {
            "power": args.power_init,
        },
    }

    optimizer = optax.adam(args.step_size)
    kernel_kwargs = fit(
        loss_fn, 
        initial_params, 
        optimizer,
        args.steps,
    )

    # fit final model
    train_x, train_e, train_y = trainset
    params = solve_closed(basekernel, train_x, train_y, reg=args.reg, kernel_kwargs=kernel_kwargs, 
                          batch_size=args.batch_size, batch_size2=args.batch_size2, verbose=False,
                          store_on_device=args.store_on_device, solve_on_device=args.solve_on_device)
    force_fn = GDMLPredict(basekernel, train_x, batch_size=args.batch_size)

    # evaluate on training data
    preds_y = force_fn(params, train_x)
    logging.info("forces:")
    logging.info(f"train MSE: {losses.mse(train_y, preds_y)}")
    logging.info(f"train MAE: {losses.mae(train_y, preds_y)}")

    energy_fn = GDMLPredictEnergy(basekernel, train_x, train_e, params, args.batch_size)
    preds_e = energy_fn(train_x)
    logging.info("energies:")
    logging.info(f"train MSE: {losses.mse(train_e, preds_e)}")
    logging.info(f"train MAE: {losses.mae(train_e, preds_e)}")

    # evaluate on test data
    test_x, test_e, test_y = testset
    preds_y = force_fn(params, test_x)
    logging.info("forces:")
    logging.info(f"test MSE: {losses.mse(test_y, preds_y)}")
    logging.info(f"test MAE: {losses.mae(test_y, preds_y)}")

    preds_e = energy_fn(test_x)
    logging.info("energies:")
    logging.info(f"test MSE: {losses.mse(test_e, preds_e)}")
    logging.info(f"test MAE: {losses.mae(test_e, preds_e)}")
