# Preaccumulating Jacobians in PyTorch for easier debugging and re-use in JAX
# (until JVP / VJP interop between AD systems becomes more stable and efficient)
import argparse
import os
import numpy as np
import schnetpack as spk
import torch

from gdml_jax.util.datasets import load_md17
from lib import schnet_input_dict, batch_jacobian

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser(description="SchNet Jacobian precomputation")
parser.add_argument("--molecule", type=str, default="ethanol")
parser.add_argument("--n_test", type=int, default=2000)
parser.add_argument("--modeldir", type=str, required=True)
parser.add_argument("--datadir", type=str, default="../data/train")
parser.add_argument("--name", type=str, default="schnet")
args = parser.parse_args()


split = np.load(os.path.join(args.modeldir, "split.npz"))
train_idx = np.hstack([split['train_idx'], split['val_idx']])
test_idx = split['test_idx'][:args.n_test]

# data loading
trainset, testset, meta = load_md17(args.molecule, datadir=args.datadir, train_idx=train_idx, test_idx=test_idx)
x_train, e_train, y_train = trainset
x_test, e_test, y_test = testset
shape = meta['shape']

x_train = np.array(x_train.reshape(-1,*shape))
y_train = np.array(y_train.reshape(-1,*shape))
x_test = np.array(x_test.reshape(-1, *shape))
y_test = np.array(y_test.reshape(-1, *shape))


# load pre-trained schnet
schnetmodel = torch.load(
    os.path.join(args.modeldir, "best_inference_model"),
    map_location=device
)
schnet = schnetmodel.representation.double()
cutoff = 5.0 # TODO load from model
z = torch.tensor(meta["z"], device=device).int()

def nn(xs):
    n_batch, n_atoms, n_dims = xs.shape
    batch = [schnet_input_dict(x, z, cutoff) for x in xs]
    batch = spk.data.loader._atoms_collate_fn(batch)
    return schnet(batch)['scalar_representation'].reshape(n_batch, n_atoms, -1)

def torch_apply(f, x):
    x = torch.tensor(x, device=device)
    return f(x).detach().cpu().numpy()

def torch_jacobians(f, x):
    x = torch.tensor(x, device=device)
    return batch_jacobian(f, x).detach().cpu().numpy()

def batched(f, batch_size):
    def batched_f(xs):
        batch_indices = np.split(np.arange(len(xs)), len(xs) / batch_size)
        return np.vstack([f(xs[idx]) for idx in batch_indices])
    return batched_f

# precompute features and Jacobians
features_train = batched(lambda xs: torch_apply(nn, xs), batch_size=5)(x_train)
jacs_train = batched(lambda xs: torch_jacobians(nn, xs), batch_size=5)(x_train)
np.savez(f"jacobians_{args.name}_{args.molecule}_train.npz", x=x_train, y=y_train, features=features_train, jacs=jacs_train, modeldir=args.modeldir)

features_test = batched(lambda xs: torch_apply(nn, xs), batch_size=5)(x_test)
jacs_test = batched(lambda xs: torch_jacobians(nn, xs), batch_size=5)(x_test)
np.savez(f"jacobians_{args.name}_{args.molecule}_test.npz", x=x_test, y=y_test, features=features_test, jacs=jacs_test, modeldir=args.modeldir)

