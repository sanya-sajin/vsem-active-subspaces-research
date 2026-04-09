# Modular Uncertainty Propagation for Surrogate-Based Bayesian Inverse Problems 

This repo contains the code for reproducing the experiments in the paper 
[Propagating Surrogate Uncertainty in Bayesian Inverse Problems](https://www.arxiv.org/abs/2601.03532), as well as a review/synthesis
paper on uncertainty quantification/active learning for probabilistic surrogates (to appear on arXiv soon). The organization of the 
repo is still slightly in flux. The primary dependencies are packages from the JAX ecosystem (JAX, NumPyro, GPjax, blackjax).

## Dependencies
To replicate the environment used to run the experiments:
```bash
uv python pin 3.13.8
uv sync --frozen
```

This will create a virtual environment; all Python code should be run using
this virtual environment, which can be activated using
```bash
source .venv/bin/activate
```

## Organization
The core code underlying the experiments is housed within the `uncprop` ("uncertainty propagation") directory.
The sub-directories are organized as follows.
- `core`
    * Inverse problem and probability distribution abstractions used in experiments
    * Sampling algorithms, written more-or-less in the [blackjax](https://blackjax-devs.github.io/blackjax/) style
    * Gaussian process surrogate model: `GPJaxSurrogate` is a wrapper around a [gpjax](https://docs.jaxgaussianprocesses.com/) model with some added functionality
    * Abstractions for random measures induced by GP surrogates
- `models`
    * Each "model" typically corresponds to one of the numerical experiments
    * Models consist of an underlying mechanistic model, a statistical inverse problem model, and a surrogate model
    * Currently the code setting up the experiments for these models is also housed here, but this will be moved outside of `uncprop` shortly
- `utils`
    * Abstractions for a single experimental "replicate" and an experiments consisting of many replicates
    * Extensions to gpjax models, including vectorized fitting of independent multi-output GPs not natively supported in gpjax
    * Other helpers for plotting, working with probability distributions, and working with uniformly spaced grids (for toy examples over 2d parameter space)
