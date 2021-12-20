import argparse
import logging
from jax.config import config

from gdml_jax.util.datasets import load_md17, get_symmetries
from gdml_jax.models import GDMLPredict, GDMLPredictEnergy
from gdml_jax.solve import solve_closed
from gdml_jax import losses
from gdml_jax.kernels import rbf, GDMLKernel, sGDMLKernel, GlobalSymmetryKernel, KernelSum
from matern import matern52
from fchl import FCHL19Kernel, FCHL19Representation

# enable double precision
config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description="GDML-JAX MD17")
parser.add_argument("--kernel", type=str, default="sGDML")
parser.add_argument("--lengthscale", type=float, required=True)
parser.add_argument("--reg", type=float, default=1e-10)
parser.add_argument("--molecule", type=str, default="ethanol")
parser.add_argument("--n_train", type=int, default=200)
parser.add_argument("--n_test", type=int, default=2000)
parser.add_argument("--batch_size", type=int, default=-1)
parser.add_argument("--batch_size2", type=int, default=-1)
parser.add_argument("--datadir", type=str, default="../data/train")
parser.add_argument("--loglevel", type=int, default=logging.INFO)
parser.add_argument("--logfile", type=str, default="")
args = parser.parse_args()

def config_logger(args):
    delim = "=============================="
    filename = args.logfile or f"{args.molecule}_train{args.n_train}_{args.kernel}_l{args.lengthscale}_reg{args.reg}"
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

# data loading
trainset, testset, meta = load_md17(args.molecule, args.n_train, args.n_test, args.datadir)
train_x, train_e, train_y = trainset
shape = meta["shape"]
z = meta["z"]
perms = get_symmetries(args.molecule)
basekernel = None


def FCHL19GlobalKernelWithSymmetries(z, perms, kappa=rbf):
    """Returns a function that computes a scalar kernel between two molecules.

    This is a variant of the sGDML kernel with FCHL19 as descriptors as opposed
    to inverse pairwise distances. Only a few physically plausible permutation 
    symmetries are taken into account to increase efficiency. """
    return GlobalSymmetryKernel(FCHL19Representation(z), kappa, perms, is_atomwise=True)

def SGDMLPlusFCHL19Kernel(shape, z, perms, kappa1=matern52, kappa2=rbf):
    k_sgdml = sGDMLKernel(shape=shape, perms=perms, kappa=kappa1)
    k_fchl19 = FCHL19Kernel(z=z, kappa=kappa2)
    return KernelSum((k_sgdml, k_fchl19))


if args.kernel == "GDML":
    basekernel = GDMLKernel(shape=shape)
elif args.kernel == "sGDML":
    basekernel = sGDMLKernel(shape=shape, perms=perms)
elif args.kernel == "GDMLmatern":
    basekernel = GDMLKernel(shape=shape, kappa=matern52)
elif args.kernel == "sGDMLmatern":
    basekernel = sGDMLKernel(shape=shape, kappa=matern52, perms=perms)
elif args.kernel == "FCHL19":
    basekernel = FCHL19Kernel(z=z)
elif args.kernel == "sGDMLFCHL19":
    basekernel = FCHL19GlobalKernelWithSymmetries(z=z, perms=perms)
elif args.kernel == "SGDMLmaternPlusFCHL19rbf":
    basekernel = SGDMLPlusFCHL19Kernel(shape=shape, z=z, perms=perms, kappa1=matern52, kappa2=rbf)
elif args.kernel == "SGDMLrbfPlusFCHL19rbf":
    basekernel = SGDMLPlusFCHL19Kernel(shape=shape, z=z, perms=perms, kappa1=rbf, kappa2=rbf)
elif args.kernel == "SGDMLrbfPlusFCHL19matern":
    basekernel = SGDMLPlusFCHL19Kernel(shape=shape, z=z, perms=perms, kappa1=rbf, kappa2=matern52)
elif args.kernel == "SGDMLmaternPlusFCHL19matern":
    basekernel = SGDMLPlusFCHL19Kernel(shape=shape, z=z, perms=perms, kappa1=matern52, kappa2=matern52)
else:
    logging.error(f"Kernel identifier '{args.kernel}' not recognized.")
    exit(1)

predict_fn = GDMLPredict(basekernel, train_x)
kernel_kwargs = {"lengthscale": args.lengthscale}

# solve in closed form
params = solve_closed(basekernel, train_x, train_y,
                      batch_size=args.batch_size, batch_size2=args.batch_size2,
                      reg=args.reg, kernel_kwargs=kernel_kwargs)

# evaluate on training data
preds = predict_fn(params, train_x)
logging.info("forces:")
logging.info(f"train MSE: {losses.mse(train_y, preds)}")
logging.info(f"train MAE: {losses.mae(train_y, preds)}")

energy_fn = GDMLPredictEnergy(basekernel, train_x, train_e, params, args.batch_size)
preds = energy_fn(train_x)
logging.info("energies:")
logging.info(f"train MSE: {losses.mse(train_e, preds)}")
logging.info(f"train MAE: {losses.mae(train_e, preds)}")

# evaluate on test data
test_x, test_e, test_y = testset
preds = predict_fn(params, test_x)
logging.info("forces:")
logging.info(f"test MSE: {losses.mse(test_y, preds)}")
logging.info(f"test MAE: {losses.mae(test_y, preds)}")

preds = energy_fn(test_x)
logging.info("energies:")
logging.info(f"test MSE: {losses.mse(test_e, preds)}")
logging.info(f"test MAE: {losses.mae(test_e, preds)}")
