import logging
import argparse
from functools import partial
from util import config_logger

import jax
import jax.numpy as jnp
from jax.config import config
from gdml_jax.util.datasets import load_md17, get_symmetries
from gdml_jax.models import GDMLPredict, GDMLPredictEnergy
from gdml_jax.solve import dkernelmatrix
from gdml_jax import losses

from gdml_jax.kernels import GDMLKernel, sGDMLKernel
from fchl import FCHL19Kernel
from fchl import FCHL19GlobalKernelWithSymmetries
from matern import matern52

import time

# enable double precision
config.update("jax_enable_x64", True)

N_ITER = 10

# arg parsing
parser = argparse.ArgumentParser(
    description="GDML-JAX MD17 force prediction benchmark"
)
parser.add_argument('--modelfile', type=str, default="")
parser.add_argument("--n_test", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--datadir", type=str, default="data/train")
parser.add_argument("--loglevel", type=int, default=logging.INFO)
parser.add_argument("--logfile", type=str, default="benchforces")
args = parser.parse_args()

m = jnp.load(args.modelfile, allow_pickle=True)
params = m['params'].item()
modelargs = m['args'].item()
result = m['result'].item()

# logging
delim = "=============================="
filename = config_logger(args)
logging.info(modelargs)
logging.info(delim)

# data loading
trainset, testset, meta = load_md17(
    modelargs.molecule, modelargs.n_train, args.n_test, args.datadir
)
train_x, train_e, train_y = trainset
shape = meta["shape"]
z = meta["z"]
perms = get_symmetries(modelargs.molecule)
params['alpha'] = params['alpha'].reshape(-1, *shape)

if modelargs.kernel == "GDML":
    KernelMaker = partial(GDMLKernel, shape=shape)
elif modelargs.kernel == "sGDML":
    KernelMaker = partial(sGDMLKernel, shape=shape, perms=perms)
elif modelargs.kernel == "GDMLmatern":
    KernelMaker = partial(GDMLKernel, shape=shape, kappa=matern52)
elif modelargs.kernel == "sGDMLmatern":
    KernelMaker = partial(sGDMLKernel, shape=shape, kappa=matern52, perms=perms)
elif modelargs.kernel == "FCHL19":
    KernelMaker = partial(FCHL19Kernel, z=z)
elif modelargs.kernel == "sGDMLFCHL19":
    KernelMaker = partial(
        FCHL19GlobalKernelWithSymmetries, z=z, perms=perms
    )
else:
    logging.error(f"Kernel identifier '{args.kernel}' not recognized.")

basekernel = partial(KernelMaker(), lengthscale=result["lengthscale"])
predict_fn = GDMLPredict(basekernel, train_x, args.batch_size)

test_x, test_e, test_y = testset
logging.info("evaluate test data fit...")
logging.info(f"test_x: {test_x.shape}, test_y: {test_y.shape}")
preds = predict_fn(params, test_x)
logging.info(f"preds: {preds.shape}")
logging.info(f"test MAE: {losses.mae(test_y, preds)}")

def time_forces(i, force_fn):
    start = time.time()
    forces = force_fn(test_x).block_until_ready()
    elapsed = time.time() - start
    logging.info(f"iteration {i+1}/{N_ITER}: predicted f in {elapsed} seconds")
    return elapsed

@jax.jit
def force_fn_fast(test_x):
    return predict_fn(params, test_x)

@jax.jit
def force_fn_slow(test_x):
    K = dkernelmatrix(basekernel, train_x, test_x, batch_size=modelargs.batch_size, batch_size2=args.batch_size, flatten=False)
    preds = jnp.einsum("jiabcd,jab->icd", K, params['alpha'])
    return preds

for force_fn in [force_fn_fast, force_fn_slow]:
    logging.info(f"#=== {force_fn.__name__} ===#")
    # warm-up
    logging.info("first warm-up construction including JIT compilation.")
    time_forces(-1, force_fn)
    logging.info("done.")
    # benchmark
    times = jnp.array([time_forces(i, force_fn) for i in range(N_ITER)])
    logging.info(f"times {times}")
    logging.info(f"mean {times.mean()}, std {times.std()}")
