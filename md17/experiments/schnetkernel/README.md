# SchNetKernel

Prerequisites:
- JAX
- PyTorch 
- schnetpack ([commit 83935fb](https://github.com/atomistic-machine-learning/schnetpack/tree/83935fb))

Steps:
1. (PyTorch) Train a SchNet representation on an MD17 data set. Example:

```
spktrain experiment=md17 model/representation=schnet data.molecule=ethanol
```

2. (PyTorch) Precompute features and Jacobians for the train and validation set. For this, update the `model_dir` variable in `precompute_jacobians.py` with the path to the pretrained model and then run
```
python precompute_jacobians.py
```

3. (JAX) Fit the force field kernel to forces
```
python train_schnetkernel.py
```
