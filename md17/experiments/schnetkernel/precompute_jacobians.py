# Preaccumulating Jacobians in PyTorch for easier debugging and re-use in JAX
# (until JVP / VJP interop between AD systems becomes more stable and efficient)
import os
from lib import schnet_input_dict, batch_jacobian

from gdml_jax.util.datasets import load_md17

import numpy as np
import torch
import schnetpack as spk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


schnet_dirs = {
    "aspirin": "../../pretrained-schnet/runs/d2ac7fa6-1dcc-11ec-821d-851e0d7b7bc4",
    "benzene": None,
    "ethanol": "../../pretrained-schnet/runs/809579c4-1dcd-11ec-a520-cf1e00acefe3",
    "malonaldehyde": "../../pretrained-schnet/runs/24d75046-25be-11ec-bf5a-256ccc7eda07",
    "naphtalene": "../../pretrained-schnet/runs/e2a8a086-1f72-11ec-9b29-cf60b7e3b52f",
    "salicylicacid": "../../pretrained-schnet/runs/66e9ee4e-25b9-11ec-a5b6-7b82d6a8d769",
    "toluene": "../../pretrained-schnet/runs/bfb996d8-1de8-11ec-8a7c-cb9cb95bd25b",
    "uracil": "../../pretrained-schnet/runs/7147319e-1dcb-11ec-8b42-f125aefaa936"
}

painn_dirs = {
    "aspirin": "../../pretrained-schnet/runs/f75e126a-1f8e-11ec-b8eb-6df15493b23e",
    "benzene": None,
    "ethanol": "../../pretrained-schnet/runs/7dd09f2a-1fad-11ec-a697-63854ae244e3",
    "malonaldehyde": "../../pretrained-schnet/runs/304bdc94-274e-11ec-80c4-75f1ff93d93c",
    "naphtalene": "../../pretrained-schnet/runs/c3efea56-1fad-11ec-8c4c-85785d8aefd1",
    "salicylicacid": "../../pretrained-schnet/runs/a3d3f1da-1f90-11ec-981f-53b84b72ec12",
    "toluene": "../../pretrained-schnet/runs/cc8c134c-1fad-11ec-8f0a-f3e9c5b7db72",
    "uracil": "../../pretrained-schnet/runs/e82ad86a-1f8d-11ec-bde9-07661e398e42"
}

model_dirs = {
    "schnet": schnet_dirs,
    "painn": painn_dirs
}

# args
DATA_DIR = '../../data/train'
N_TEST = 1000

# REPRESENTATION = "schnet"
# REPRESENTATION = "painn"
REPRESENTATION = os.environ["REPRESENTATION"]

# MOLECULE = "aspirin"
# MOLECULE = "benzene"
# MOLECULE = "ethanol"
# MOLECULE = "malonaldehyde"
# MOLECULE = "naphtalene"
# MOLECULE = "salicylicacid"
# MOLECULE = "toluene"
# MOLECULE = "uracil"
MOLECULE = os.environ["MOLECULE"]

model_dir = model_dirs[REPRESENTATION][MOLECULE]
split = np.load(os.path.join(model_dir, "split.npz"))
train_idx = np.hstack([split['train_idx'], split['val_idx']])
test_idx = split['test_idx'][:N_TEST]

# data loading
trainset, testset, meta = load_md17(MOLECULE, datadir=DATA_DIR, train_idx=train_idx, test_idx=test_idx)
x_train, e_train, y_train = trainset
x_test, e_test, y_test = testset
shape = meta['shape']

x_train = np.array(x_train.reshape(-1,*shape))
y_train = np.array(y_train.reshape(-1,*shape))
x_test = np.array(x_test.reshape(-1, *shape))
y_test = np.array(y_test.reshape(-1, *shape))


# load pre-trained schnet
schnetmodel = torch.load(
    os.path.join(model_dir, "best_inference_model"),
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
# features_train = torch_apply(nn, x_train)
# jacs_train = torch_jacobians(nn, x_train)
features_train = batched(lambda xs: torch_apply(nn, xs), batch_size=5)(x_train)
jacs_train = batched(lambda xs: torch_jacobians(nn, xs), batch_size=5)(x_train)
np.savez(f"jacobians_{REPRESENTATION}_{MOLECULE}_train.npz", x=x_train, y=y_train, features=features_train, jacs=jacs_train, modeldir=model_dir)

# features_test = torch_apply(nn, x_test)
# jacs_test = torch_jacobians(nn, x_test)
features_test = batched(lambda xs: torch_apply(nn, xs), batch_size=5)(x_test)
jacs_test = batched(lambda xs: torch_jacobians(nn, xs), batch_size=5)(x_test)
np.savez(f"jacobians_{REPRESENTATION}_{MOLECULE}_test.npz", x=x_test, y=y_test, features=features_test, jacs=jacs_test, modeldir=model_dir)

