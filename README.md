# ad-kernels
Supplementary material for **"Algorithmic Differentiation for Automatized Modelling of Machine Learned Force Fields"** by Niklas Frederik Schmitz, Klaus-Robert Müller and Stefan Chmiela.

The preprint is available here: https://arxiv.org/abs/2208.12104

## Requirements

- python 3.8
- [JAX](https://github.com/google/jax#installation)

Below is an example setup:

```
conda create -n ad-kernels python=3.9
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
