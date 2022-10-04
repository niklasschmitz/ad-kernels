# ad-kernels
Supplementary material for **"Algorithmic Differentiation for Automatized Modelling of Machine Learned Force Fields"** by Niklas Frederik Schmitz, Klaus-Robert Müller and Stefan Chmiela.

The preprint is available here: https://arxiv.org/abs/2208.12104

<div align="middle">
<img src="_figures/graphical_toc.png" width=50%>
</div>

![](_figures/boxology.png)

## Speedups

Even on the small molecules of the [MD17 data set](http://www.sgdml.org/#datasets), we observe practical relative **speedups up to 50x** for force prediction timings using our efficient higher-order AD contraction approach over naive dense higher-order AD:

|               | Ethanol (N=9) |             |             | Aspirin (N=21) |             |            |
|---------------|---------------|-------------|-------------|----------------|-------------|------------|
|               | dense         | contraction | **speedup** | dense          | contraction | **speedup**|
| sGDML         | 00.0780       | 0.0018      | **43.3x**   | 000.2832       | 00.0087     | **x32.5**  |
| FCHL19GPR     | 57.9439       | 0.9815      | **59.0x**   | 414.0216       | 10.1438     | **x40.8**  |
| sGDML[RBF]    | 00.0800       | 0.0015      | **53.3x**   | 000.2811       | 00.0074     | **x37.9**  |
| global-FCHL19 | 57.5194       | 0.9653      | **59.5x**   | 458.5465       | 10.0728     | **x45.5**  |
| sFCHL19       | 60.3951       | 1.0704      | **56.4x**   | 419.3778       | 10.3336     | **x40.5**  |

> Benchmarked force prediction times (s) for different kernels. Each model was
> trained with 1000 training points, and evaluated for 10 points at the same time. All
> timings are averaged over 10 runs (excluding an initial run for just-in-time compilation). Our approach (contraction) yields consistent speedups by a up to two orders
> of magnitude over the direct (dense) implementation of the constrained models. All
> measurements are done on a single Nvidia Titan RTX 24 GB GPU.

See our [preprint](https://arxiv.org/abs/2208.12104) for details of
where such speedups come from when considering algorithmic choices
of higher-order AD and operator-transformed Gaussian processes.

## Content

1. [md17/](md17/)
    - [gdml_jax](md17/gdml_jax/)    
      - a small toolbox focused on gradients and molecular force fields
    - [train_sgdml.py](md17/experiments/train_sgdml.py)
      - code for fitting different models to MD17 forces
    - [experiments/schnetkernel](md17/experiments/schnetkernel/)
      - utils for schnetpack interop
    - [hyper_coulomb.py](md17/experiments/hyper_coulomb.py)
      - example on efficiently optimizing kernel hyperparameters by another outer level of AD
    - [benchmark_forces.py](md17/experiments/md17_benchmark_forces.py)
      - benchmark for dense instantiation vs fast operator kernel contraction
2. [pde/](pde/) more general differential operators (gradients, Hessians, VJP, ...)
   - [opgp_demo.ipynb](pde/opgp_demo.ipynb) 
     - A 2D regession problem using gradients and Hessians
   - [laplace.ipynb](pde/laplace.ipynb) 
     - A toy example solving Laplace's equation on an annulus
   - [wave_eq.ipynb](pde/wave_eq.ipynb)
     - A toy example solving a wave equation in one dimension


## Requirements

- python 3.8
- [JAX](https://github.com/google/jax#installation)

Below is an example setup:

```
conda create -n ad-kernels python=3.8
conda activate ad-kernels
cd md17 && pip install -e .
```

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2208.12104,
  author = {Schmitz, Niklas Frederik and Müller, Klaus-Robert and Chmiela, Stefan},
  title = {Algorithmic Differentiation for Automatized Modelling of Machine Learned Force Fields},
  publisher = {arXiv},
  year = {2022},
  doi = {10.48550/ARXIV.2208.12104},
  url = {https://arxiv.org/abs/2208.12104},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
