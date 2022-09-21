import os
import urllib.request
import urllib.parse
import posixpath
import logging
import jax.numpy as jnp
import numpy as np


DEFAULT_DIR = '/tmp/gdml_jax/data/train'
MD17_URLS = {
    "ethanol": 'http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz',
    "aspirin": 'http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz',
    "benzene": 'http://quantum-machine.org/gdml/data/npz/benzene_old_dft.npz',
    "uracil": 'http://quantum-machine.org/gdml/data/npz/uracil_dft.npz',
    "naphtalene": 'http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz',
    "salicylicacid": 'http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz',
    "malonaldehyde": 'http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz',
    "toluene": "http://quantum-machine.org/gdml/data/npz/toluene_dft.npz"
}

def load_md17(molecule, n_train=100, n_test=200, datadir=DEFAULT_DIR, train_idx=None, test_idx=None):
    url = MD17_URLS[molecule]
    urlpath = urllib.parse.urlsplit(url).path  # gdml/data/npz/aspirin_dft.npz
    filename = posixpath.basename(urlpath)  # aspirin_dft.npz
    filepath = os.path.join(datadir, filename)  # {datadir}/aspirin_dft.npz
    
    if not os.path.isdir(datadir):
        os.makedirs(datadir)

    if not os.path.isfile(filepath):
        logging.info(f"[gdml_jax]: Downloading \'{molecule}\' MD17 dataset...")
        urllib.request.urlretrieve(url, filepath)
        logging.info("[gdml_jax]: done")
        
    data = jnp.load(filepath)
    
    r = jnp.array(data.get("R"))             # atomic coordinates
    e = jnp.array(data.get("E")).reshape(-1) # energies
    f = jnp.array(data.get("F"))             # forces

    # the shape (n_atoms, n_spatial_dim) is needed for pairwise descriptors
    shape = r.shape[1:]

    # sample observations without replacement
    assert n_train > 0 and n_test > 0
    np.random.seed(0)
    indices = np.random.choice(len(r), n_train + n_test, replace=False)
    train_indices, test_indices = indices[:n_train], indices[n_train:]

    # if indices explicitly passed by user, overrule random indices
    if train_idx is not None:
        train_indices = train_idx
    if test_idx is not None:
        test_indices = test_idx

    train_r = r[train_indices]
    train_e = e[train_indices]
    train_f = f[train_indices]
    test_r = r[test_indices]
    test_e = e[test_indices]
    test_f = f[test_indices]

    trainset = (train_r, train_e, train_f)
    testset = (test_r, test_e, test_f)
    
    # collect metadata
    meta = {key: data[key] for key in ["name", "theory", "z", "type", "md5"]}
    meta["train_indices"] = train_indices
    meta["test_indices"] = test_indices
    meta["shape"] = shape

    return trainset, testset, meta

def get_symmetries(molecule):
    """sGDML symmetries precomputed for MD17 (https://github.com/stefanch/sGDML)."""
    symmetries = {
        "aspirin": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 18, 20],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 18, 19],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 19, 18],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 18],
            ]
        ),
        "benzene": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [0, 5, 4, 3, 2, 1, 6, 11, 10, 9, 8, 7],
                [1, 0, 5, 4, 3, 2, 7, 6, 11, 10, 9, 8],
                [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9],
                [2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 6, 7],
                [3, 2, 1, 0, 5, 4, 9, 8, 7, 6, 11, 10],
                [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
                [4, 3, 2, 1, 0, 5, 10, 9, 8, 7, 6, 11],
                [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6],
                [5, 0, 1, 2, 3, 4, 11, 6, 7, 8, 9, 10],
                [4, 5, 0, 1, 2, 3, 10, 11, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6],
            ]
        ),
        "ethanol": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 2, 4, 3, 5, 7, 6, 8],
                [0, 1, 2, 4, 3, 6, 5, 7, 8],
                [0, 1, 2, 4, 3, 7, 6, 5, 8],
                [0, 1, 2, 3, 4, 7, 5, 6, 8],
                [0, 1, 2, 3, 4, 6, 7, 5, 8],
            ]
        ),
        "malonaldehyde": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [0, 1, 2, 3, 4, 5, 7, 6, 8],
                [2, 1, 0, 4, 3, 8, 6, 7, 5],
                [2, 1, 0, 4, 3, 8, 7, 6, 5],
            ]
        ),
        "naphtalene": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
                [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 13, 12, 11, 10, 17, 16, 15, 14],
                [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 14, 15, 16, 17, 10, 11, 12, 13],
                [8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 17, 16, 15, 14, 13, 12, 11, 10]
            ]
        ),
        "salicylicacid": jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]),
        "toluene": jnp.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 9, 8, 7, 10, 11, 12, 13, 14],
                [0, 1, 6, 5, 4, 3, 2, 7, 8, 9, 14, 13, 12, 11, 10],
                [0, 1, 6, 5, 4, 3, 2, 7, 9, 8, 14, 13, 12, 11, 10],
                [0, 1, 6, 5, 4, 3, 2, 8, 7, 9, 14, 13, 12, 11, 10],
                [0, 1, 6, 5, 4, 3, 2, 9, 8, 7, 14, 13, 12, 11, 10],
                [0, 1, 2, 3, 4, 5, 6, 9, 7, 8, 10, 11, 12, 13, 14],
                [0, 1, 2, 3, 4, 5, 6, 8, 9, 7, 10, 11, 12, 13, 14],
                [0, 1, 6, 5, 4, 3, 2, 9, 7, 8, 14, 13, 12, 11, 10],
                [0, 1, 6, 5, 4, 3, 2, 8, 9, 7, 14, 13, 12, 11, 10],
            ]
        ),
        "uracil": jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]),
    }
    return symmetries[molecule]
