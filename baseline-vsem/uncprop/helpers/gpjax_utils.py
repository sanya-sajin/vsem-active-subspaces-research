from __future__ import annotations

import pprint
from typing import Dict, Sequence, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions.transforms as npt
from flax import nnx
import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.parameters import DEFAULT_BIJECTION, transform, Parameter
from gpjax.parameters import PositiveReal  # Parameter helper class
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from gpjax.kernels import RBF
import optax as ox


def make_interval_bijector(low, high):
    """Return a NumPyro ComposeTransform mapping R -> (low, high)."""
    return npt.ComposeTransform([npt.SigmoidTransform(), 
                                 npt.AffineTransform(jnp.asarray(low), jnp.asarray(high) - jnp.asarray(low))])


def compute_bounds_from_design(design):
    """
    Derive anisotropic per-dimension lengthscale bounds and y-variance based bounds
    for kernel variance and noise from your Dataset-like object with attributes
    .X (N,D) and .y (N,).
    """
    X = np.asarray(design.X, dtype=float)
    y = np.asarray(design.y).ravel()
    N, D = X.shape

    lengthscale_low = np.empty(D, dtype=float)
    lengthscale_high = np.empty(D, dtype=float)

    for d in range(D):
        col = X[:, d]
        diffs = np.abs(col[:, None] - col[None, :]).reshape(N * N)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            min_nonzero = 1e-6
            max_pair = max(1.0, float(np.ptp(col)))
        else:
            min_nonzero = float(np.min(diffs))
            max_pair = float(np.max(diffs))
        lengthscale_low[d] = max(min_nonzero * 0.1, 1e-9)
        lengthscale_high[d] = max(max_pair * 2.0, lengthscale_low[d] * 10.0)

    y_var = float(np.var(y, ddof=0)) if y.size > 1 else 1.0
    kernel_var_low = max(y_var * 1e-3, 1e-9)
    kernel_var_high = max(y_var * 10.0, kernel_var_low * 10.0)

    noise_low = 1e-9
    noise_high = max(y_var * 0.2, 1e-9)

    return dict(
        lengthscale_low=lengthscale_low,
        lengthscale_high=lengthscale_high,
        kernel_var_low=kernel_var_low,
        kernel_var_high=kernel_var_high,
        noise_low=noise_low,
        noise_high=noise_high,
    )


def average_pairwise_distance_per_dim(x):
    """
    Compute average distance between pairwise distinct points, dim-by-dim.
    """
    n, d = x.shape
    diffs = x[:, None, :] - x[None, :, :]  # (n, n, d)
    abs_diffs = np.abs(diffs)

    # Discard diagonal (i==j), only keep i != j pairs
    mask = ~np.eye(n, dtype=bool)
    abs_diffs_pairs = abs_diffs[mask].reshape(n * (n - 1), d)
    avg_dist_per_dim = abs_diffs_pairs.mean(axis=0)
    return avg_dist_per_dim


# -----------------------------------------------------------------------------
# Modified version of gpjax fit_scipy() function 
# -----------------------------------------------------------------------------

import typing as tp 
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten_with_path
from scipy.optimize import minimize
from gpjax.objectives import Objective
from numpyro.distributions.transforms import Transform
from gpjax.parameters import (
    DEFAULT_BIJECTION,
    Parameter,
    transform,
)

from gpjax.typing import (
    Array,
    KeyArray,
    ScalarFloat,
)

Model = tp.TypeVar("Model", bound=nnx.Module)


def _check_model(model: tp.Any) -> None:
    """Check that the model is a subclass of nnx.Module."""
    if not isinstance(model, nnx.Module):
        raise TypeError(
            "Expected model to be a subclass of nnx.Module. "
            f"Got {model} of type {type(model)}."
        )


def _check_train_data(train_data: tp.Any) -> None:
    """Check that the train_data is of type gpjax.Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError(
            "Expected train_data to be of type gpjax.Dataset. "
            f"Got {train_data} of type {type(train_data)}."
        )


def _check_optim(optim: tp.Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError(
            "Expected optim to be of type optax.GradientTransformation. "
            f"Got {optim} of type {type(optim)}."
        )


def _check_num_iters(num_iters: tp.Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError(
            "Expected num_iters to be of type int. "
            f"Got {num_iters} of type {type(num_iters)}."
        )

    if num_iters <= 0:
        raise ValueError(f"Expected num_iters to be positive. Got {num_iters}.")


def _check_log_rate(log_rate: tp.Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError(
            "Expected log_rate to be of type int. "
            f"Got {log_rate} of type {type(log_rate)}."
        )

    if not log_rate > 0:
        raise ValueError(f"Expected log_rate to be positive. Got {log_rate}.")


def _check_verbose(verbose: tp.Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError(
            "Expected verbose to be of type bool. "
            f"Got {verbose} of type {type(verbose)}."
        )


def _check_batch_size(batch_size: tp.Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError(
            "Expected batch_size to be of type int. "
            f"Got {batch_size} of type {type(batch_size)}."
        )

    if not batch_size == -1 and not batch_size > 0:
        raise ValueError(f"Expected batch_size to be positive or -1. Got {batch_size}.")


def custom_fit_scipy(  # noqa: PLR0913
    *,
    model: Model,
    objective: Objective,
    train_data: Dataset,
    trainable: nnx.filterlib.Filter = Parameter,
    params_bijection: dict[str, Transform] = DEFAULT_BIJECTION, ### MODIFICATION: added argument
    max_iters: int = 500,
    verbose: bool = True,
    safe: bool = True,
) -> tuple[Model, Array]:
    r"""Train a Module model with respect to a supplied Objective function.
    Optimisers used here should originate from Optax. todo

    Args:
        model: the model Module to be optimised.
        objective: The objective function that we are optimising with
            respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        max_iters (int): The maximum number of optimisation steps to run. Defaults
            to 500.
        verbose (bool): Whether to print the information about the optimisation. Defaults
            to True.

    Returns:
        A tuple comprising the optimised model and training history.
    """
    if safe:
        # Check inputs.
        _check_model(model)
        _check_train_data(train_data)
        _check_num_iters(max_iters)
        _check_verbose(verbose)

    # Model state filtering
    graphdef, params, *static_state = nnx.split(model, trainable, ...)

    #### Starting modifications
    # Issue is that nnx.split wraps variables as VariableState, but gpjax transform()
    # only looks for Parameter objects. Thus ignores VariableState and ends up always
    # defaulting to identity transformations. Here we simply unwrap the nodes to 
    # extract the underlying variables from the VariableStates.
    params = jax.tree_util.tree_map(
        lambda x: x.to_variable() if isinstance(x, nnx.VariableState) else x,
        params,
        is_leaf=lambda x: isinstance(x, (nnx.VariableState, Parameter)),
    )

    # Store structure of the tree so can be recreated later
    param_template = tree_flatten_with_path(params, is_leaf = lambda x: isinstance(x, Parameter))
    #### Ending modifications

    # Parameters bijection to unconstrained space
    params = transform(params, params_bijection, inverse=True)

    # Loss definition
    def loss(params) -> ScalarFloat:
        params = transform(params, params_bijection)
        model = nnx.merge(graphdef, params, *static_state)
        return objective(model, train_data)

    # convert to numpy for interface with scipy
    x0, scipy_to_jnp = ravel_pytree(params)

    @jax.jit
    def scipy_wrapper(x0):
        ### MODIFICATION: replacing scipy_to_jnp(jnp.array(x0)) with call to _scipy_to_nnx_state
        param_state = _scipy_to_nnx_state(jnp.array(x0), scipy_to_jnp, param_template)
        value, grads = jax.value_and_grad(loss)(param_state)
        scipy_grads = ravel_pytree(grads)[0]
        return value, scipy_grads

    history = [scipy_wrapper(x0)[0]]
    result = minimize(
        fun=scipy_wrapper,
        x0=x0,
        jac=True,
        callback=lambda X: history.append(scipy_wrapper(X)[0]),
        options={"maxiter": max_iters, "disp": verbose},
    )
    history = jnp.array(history)

    # convert back to nnx.State with JAX arrays
    params = _scipy_to_nnx_state(result.x, scipy_to_jnp, param_template)

    # Parameters bijection to constrained space
    print('unconstrained:')
    print(params)
    params = transform(params, params_bijection)
    print('constrained')
    print(params)

    # Reconstruct model
    model = nnx.merge(graphdef, params, *static_state)

    return model, history


def _scipy_to_nnx_state(x, scipy_to_jnp, param_template):
    """
    Addressing issue that scipy_to_jnp strips the Parameter class from the 
    tree, which means that tranform() does not recognize the values as
    parameters and defaults to identity transform.
    """
    leaf_template, treedef_template = param_template
    tree = scipy_to_jnp(x)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    leaf_template_keys = [l[0] for l in leaf_template]

    updated_leaves = []
    for leaf_info in leaves:
        key, val = leaf_info
        idx = leaf_template_keys.index(key)
        target_structure = leaf_template[idx][1]
        updated_leaf = _wrap_as_parameter(val, target_structure)
        updated_leaves.append(updated_leaf)

    updated_pytree = jax.tree_util.tree_unflatten(treedef, updated_leaves)
    return nnx.State(updated_pytree)


def _wrap_as_parameter(val, target_structure):
    cls = type(target_structure)
    return cls(value=val, tag=target_structure.tag)


if __name__ == '__main__':
    # Bijections
    lengthscale_bijection = make_interval_bijector(bounds["lengthscale_low"], bounds["lengthscale_high"])
    kernel_var_bijection = make_interval_bijector(bounds["kernel_var_low"], bounds["kernel_var_high"])
    likelihood_noise_bijection = make_interval_bijector(bounds["noise_low"], bounds["noise_high"])

    param_path_tag_map = {
        'likelihood/obs_stddev': 'lik_noise',
        'prior/kernel/lengthscale': 'kernel_ls',
        'prior/kernel/variance': 'kernel_var'
    }

    bijection = dict(DEFAULT_BIJECTION)

    print('Default:')
    print(bijection)

    bijection['lik_noise'] = likelihood_noise_bijection
    bijection['kernel_ls'] = lengthscale_bijection
    bijection['kernel_var'] = kernel_var_bijection

    print('Custom:')
    print(bijection)

    # Update tags
    gp_untuned_posterior.prior.kernel.lengthscale.tag = 'kernel_ls'
    gp_untuned_posterior.prior.kernel.variance.tag = 'kernel_var'
    gp_untuned_posterior.likelihood.obs_stddev.tag = 'lik_noise'
