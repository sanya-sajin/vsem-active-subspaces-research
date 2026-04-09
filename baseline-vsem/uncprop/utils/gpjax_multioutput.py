# uncprop/utils/gpjax_multioutput.py
"""
Helpers for vectorized batch independent multioutput GPs.
"""
from __future__ import annotations
from typing import Protocol
from dataclasses import dataclass
from collections.abc import Callable, Sequence

from flax import nnx
import jax
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, jit

import optax
import gpjax as gpx
from gpjax import Dataset
from gpjax.parameters import transform, DEFAULT_BIJECTION
from gpjax.gps import ConjugatePosterior, _build_fourier_features_fn
from gpjax.kernels import AbstractKernel
from gpjax.kernels.computations import AbstractKernelComputation
from gpjax.parameters import NonNegativeReal, PositiveReal

from numpyro.distributions.transforms import Transform
from numpyro.distributions import MultivariateNormal

from uncprop.custom_types import Array, ArrayLike, PRNGKey
# from uncprop.core.surrogate import GPJaxSurrogate # TODO: fix circular import
from uncprop.core.distribution import GaussianFromNumpyro


Bijection = dict

# -----------------------------------------------------------------------------
# Batched kernels
#
#   gpjax does not natively support kernels batching over hyperparameters.
#   As I found out the hard way, gpjax kernels will still run without error
#   with batched parameters, but silently give the wrong result.
#
#   Here we define a batch kernel as a function k(x, y) that returns a 
#   (q,) vector, where q is the batch size. BatchDenseKernelComputation
#   does nothing more than take such a function and vectorize it to 
#   return (q, n, m) when x, y have leading batch dimensions n and 
#   m, respectively.
#
#   We then generalize gpjax's stationary kernels to support batched 
#   hyperparameters, such that it can be used with 
#   BatchDenseKernelComputation.
# -----------------------------------------------------------------------------


class BatchDenseKernelComputation(AbstractKernelComputation):
    """
    Dense kernel computation for vector-valued kernels; i.e., kernels
    that return shape (q,) when evaluated at a single pair of points.
    Batch dimension is always the leading dimension.

    Notes: 
        Identical to gpjax DenseKernelComputation but explicitly assumes
        q-valued kernel, always returns 3d array, and moves batch index
        to leading dimension.
    """

    def _cross_covariance(
        self,
        kernel,
        x: Array,
        y: Array,
    ) -> Array:
        """
        For x (n,d) and y (m, d) returns (q, n, m). Note that even if kernel 
        is scalar-valued, return shape will be (1, n, m).
        """
        x = jnp.atleast_2d(x)
        y = jnp.atleast_2d(y)

        kxy = vmap(lambda x: vmap(lambda y: kernel(x, y))(y))(x) # (n, m, b) or (n, m)

        kxy = jnp.atleast_3d(kxy) # (n, m, b)
        return jnp.moveaxis(kxy, -1, 0)


def batched_squared_distance(x, y):
    """
    x, y: (..., D)
    returns: (...)
    """
    return jnp.sum((x - y) ** 2, axis=-1)


class BatchedStationaryKernel(AbstractKernel):
    """
    A batched version of gpjax's stationary kernel. To keep things simple,
    we require users to specify the batch dimension q and input dimension
    d. The lengthscale and variance parameters must be broadcastable to 
    (q, d) and (q,), respectively.
    """

    def __init__(
        self,
        batch_dim: int,
        input_dim: int, 
        active_dims=None,
        lengthscale: ArrayLike | nnx.Variable = 1.0,
        variance: ArrayLike | nnx.Variable = 1.0,
        compute_engine: AbstractKernelComputation = BatchDenseKernelComputation(),
    ):
        """Initializes the kernel.

        Args:
            active_dims: The indices of the input dimensions that the kernel operates on.
            lengthscale: the lengthscale(s) of the kernel ℓ. If a scalar or an array of
                length 1, the kernel is isotropic, meaning that the same lengthscale is
                used for all input dimensions. If an array with length > 1, the kernel is
                anisotropic, meaning that a different lengthscale is used for each input.
            variance: the variance of the kernel σ.
            n_dims: The number of input dimensions. If `lengthscale` is an array, this
                argument is ignored.
            compute_engine: The computation engine that the kernel uses to compute the
                covariance matrix.
        """

        super().__init__(active_dims, input_dim, compute_engine)

        lengthscale, variance = _validate_batch_kernel_params(
            batch_dim=batch_dim,
            input_dim=input_dim,
            lengthscale=lengthscale,
            variance=variance,
        )

        self.batch_dim = batch_dim
        self.n_dims = input_dim
        self.lengthscale: nnx.Variable = lengthscale
        self.variance: nnx.Variable  = variance


    def _scaled_squared_distance(self, x, y):
        """
        x, y: (..., D)
        returns: (...)
        """
        x = self.slice_input(x)
        y = self.slice_input(y)
        return batched_squared_distance(x / self.lengthscale.value,
                                        y / self.lengthscale.value)


def _validate_batch_kernel_params(batch_dim, input_dim, lengthscale, variance):

    # broadcast lengthscale shape
    if isinstance(lengthscale, nnx.Variable):
        val = lengthscale.get_value()
        val = jnp.broadcast_to(val, (batch_dim, input_dim))
        lengthscale.set_value(val)
    else:
        lengthscale = jnp.broadcast_to(lengthscale, (batch_dim, input_dim))
        lengthscale = PositiveReal(lengthscale)


    # broadcast variance shape
    if isinstance(variance, nnx.Variable):
        val = variance.get_value()
        val = jnp.broadcast_to(val, (batch_dim,))
        variance.set_value(val)
    else:
        variance = jnp.broadcast_to(variance, (batch_dim,))
        variance = NonNegativeReal(variance)

    return lengthscale, variance


class BatchedRBF(BatchedStationaryKernel):
    """
    Generalization of gpjax RBF kernel that supports batched kernel
    parameters. The kernel variance and lengthscales are (q,) and
    (q, d) respectively.
    """
    name = "BatchedRBF"

    def __call__(self, x, y):
        sqdist = self._scaled_squared_distance(x, y)
        return self.variance.get_value() * jnp.exp(-0.5 * sqdist)


# -----------------------------------------------------------------------------
# Utilities for batched hyperparameter optimization
# -----------------------------------------------------------------------------

class SingleOutputGPFactory(Protocol):
    def __call__(self, dataset: Dataset) -> ConjugatePosterior:
        """ Returns a gpjax posterior for a single-output GP"""
        pass


class BatchIndependentGP:
    """ Stores a set of independent gpjax GP posterior objects
    
    This class provides the functionality to map between two different 
    representations of the batch GP:
        (1) a list of single-output GP posteriors
        (2) a single batch GP posterior

    The batch GP posterior is a tree in which the parameter leaves have an
    added batch dimension (the first dimension). This is the representation
    that allows for vectorized hyperparameter optimization and prediction.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 posterior_list: Sequence[ConjugatePosterior] | None = None,
                 batch_posterior: ConjugatePosterior | None = None):
        """
        Notes:
            dataset is the shared multi-output dataset across all GPs.
        """
        if not ((posterior_list is None) ^ (batch_posterior is None)):
            raise ValueError('BatchIndependentGP requires exactly one of posterior_list or batch_posterior.')
        
        self.dataset = dataset
        self.dim_out = dataset.y.shape[1]

        if(posterior_list is None):
            self.posterior_list, self.tree_info = _posterior_batch_to_list(batch_posterior, self.dim_out)
            self.batch_posterior = batch_posterior
        else:
            self.batch_posterior, self.tree_info = _posterior_list_to_batch(posterior_list)
            self.posterior_list = posterior_list

    @property
    def graphdef(self):
        return self.tree_info[0]

    @property
    def params(self):
        return self.tree_info[1]
    
    @property
    def static_state(self):
        return self.tree_info[2]


def get_batch_gp_from_template(gp_factory: SingleOutputGPFactory,
                               dataset: Dataset) -> BatchIndependentGP:
    """
    Args:
        gp_factory: callable that returns one single-output GP posterior
        dataset: gpjax Dataset, with "y" of shape (N, Q) where Q is the number
                    of GPs in the batch.
    """
    out_dim = dataset.y.shape[1]
    posteriors = []
    
    for i in range(out_dim):
        dataset_i = _get_single_output_dataset(dataset, i)
        posterior_i = gp_factory(dataset_i)
        posteriors.append(posterior_i)
    
    return BatchIndependentGP(dataset=dataset, posterior_list=posteriors)


def _get_single_output_dataset(dataset: Dataset, output_idx: int):
    return Dataset(dataset.X, dataset.y[:, [output_idx]])


def _make_batched_loss_and_grad(batch_gp: BatchIndependentGP,
                                objective,
                                bijection: Bijection,
                                dataset: Dataset):
    X = dataset.X
    Y = dataset.y
    graphdef = batch_gp.graphdef
    static = batch_gp.static_state

    # loss/grad for a single-output GP
    def single_loss(param, y):
        param_constr = transform(param, bijection)
        model = nnx.merge(graphdef, param_constr, *static)
        return objective(model, Dataset(X, y[:, None]))
    
    single_grad = jax.grad(single_loss, argnums=0)

    # batched independent loss/gradient computation
    loss_vect = jax.vmap(single_loss, in_axes=(0, 1))
    grad_vect = jax.vmap(single_grad, in_axes=(0, 1))
                         
    def loss_and_grad(params):
        return loss_vect(params, Y), grad_vect(params, Y)

    return loss_and_grad


def fit_batch_independent_gp(
    batch_gp: BatchIndependentGP,
    objective,
    optim: optax.GradientTransformation,
    *,
    bijection: Bijection = DEFAULT_BIJECTION,
    num_iters: int = 100,
    key: jax.Array = jax.random.PRNGKey(0),
    unroll: int = 1,
):
    """
    Fit a batch of independent GP models using vectorized optimisation.

    Args:
        batch_gp: BatchIndependentGP instance.
        objective: Scalar loss function (e.g. negative log marginal likelihood).
        optim: Optax optimizer.
        num_iters: Number of optimisation steps.
        key: PRNGKey.
        unroll: Scan unroll factor.

    Notes:
        This function is defined specifically for a BatchIndependentGP, which uses 
        a standard gpjax kernel with batched parameters. Vectorized evaluation of 
        gram/cross covariance matrices will not work properly for such a kernel, but
        this function 
        

    Returns:
        Updated BatchIndependentGP with trained posteriors.
        Training loss history of shape (num_iters, Q).
    """
    params_unconstr = transform(batch_gp.params, bijection, inverse=True)

    # initialize batch optimizer state and create batch loss
    opt_state = jax.vmap(optim.init)(params_unconstr)
    loss_and_grad_fn = _make_batched_loss_and_grad(batch_gp, objective, bijection, batch_gp.dataset)

    @jit
    def step(carry, _):
        params, opt_state = carry
        loss, grads = loss_and_grad_fn(params)
        updates, opt_state = jax.vmap(optim.update)(grads, opt_state, params)
        params = jax.vmap(optax.apply_updates)(params, updates)
        carry = (params, opt_state)
        return carry, loss

    # Optimisation loop
    keys = jax.random.split(key, num_iters)
    (params_unconstr, _), history = jax.lax.scan(step, (params_unconstr, opt_state), keys, unroll=unroll)

    # Parameters bijection to constrained space
    params = transform(params_unconstr, bijection)

    # Return batch posterior with updated parameters
    new_tree = nnx.merge(batch_gp.graphdef, params, *batch_gp.static_state)
    new_batch_gp = BatchIndependentGP(dataset=batch_gp.dataset, batch_posterior=new_tree)

    return new_batch_gp, history


def _posterior_batch_to_list(posterior_batch, dim_out) -> tuple[Sequence[ConjugatePosterior], tuple]:
    """
    dim_out is the number of outputs in the multioutput GP (i.e., the length of
    the batch dimension). Assumes parameters have leading batch dimension.
    """

    graphdef, params, *static = nnx.split(posterior_batch, gpx.parameters.Parameter, ...)

    posterior_list = []
    for i in range(dim_out):
        param = jax.tree.map(lambda p: p[i], params)
        post = nnx.merge(graphdef, param, *static)
        posterior_list.append(post)

    info = (graphdef, params, static)
    return posterior_list, info


def _posterior_list_to_batch(
        posterior_list: Sequence[ConjugatePosterior]
) -> tuple[ConjugatePosterior, tuple]:
    """Split and stack independent GP posteriors into a batched PyTree.
    
    Returns:
        tuple:
            - batch posterior: gpjax ConjugatePosterior with batch parameters
            - info: tuple with graphdef, batch_params, static that can be combined to 
                    form the batch posterior using nnx.merge.
    """
    graphdefs = []
    params = []
    static_states = []

    for post in posterior_list:
        graphdef, param, *static = nnx.split(post, gpx.parameters.Parameter, ...)
        graphdefs.append(graphdef)
        params.append(param)
        static_states.append(tuple(static))

    # All graphdefs / static states must be identical
    if not all(g == graphdefs[0] for g in graphdefs):
        raise ValueError("All GP posteriors must share the same structure.")
    if not all(s == static_states[0] for s in static_states):
        raise ValueError("All GP static states must be identical.")
    
    graphdef = graphdefs[0]
    static_state = static_states[0]

    # transpose tree - values of parameter leaves are now batch values
    batch_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params)
    batch_post = nnx.merge(graphdef, batch_params, *static_state)
    info = (graphdef, batch_params, static_state)
    
    return batch_post, info