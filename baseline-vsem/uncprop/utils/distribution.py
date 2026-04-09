# uncprop/utils/distribution.py

import jax
import jax.random as jr
import jax.numpy as jnp
import jax.scipy as jsp
from jax.scipy.stats import norm
from jax.scipy.linalg import solve_triangular

from uncprop.custom_types import Array, ArrayLike, PRNGKey


# -----------------------------------------------------------------------------
# Clipped Gaussian and LogNormal mean
# -----------------------------------------------------------------------------

def clipped_gaussian_mean(m: ArrayLike,
                          s2: ArrayLike,
                          b: ArrayLike) -> Array:
    """Compute E[min(X, b)] for X ~ Normal(m, s2), vectorized and JAX-compatible.

    The inputs `m`, `s2`, and `b` may be scalars or arrays of the same shape
    (or broadcastable shapes). The result is the elementwise expectation
    E[min(X, b)] computed using the analytic formula:

        E[min(X, b)] = b + (m - b) * Phi(z) - sigma * phi(z),
        where z = (b - m) / sigma, sigma = sqrt(s2).

    The degenerate case s2 == 0 is handled: E[min(X,b)] = min(m, b).

    Args:
        m: Mean(s) of the normal distribution (scalar or array).
        s2: Variance(s) of the normal distribution (scalar or array).
        b: Upper clipping bound(s) (scalar or array).

    Returns:
        An array (or scalar) of the same broadcasted shape containing
        E[min(X, b)].
    """
    m = jnp.asarray(m)
    s2 = jnp.asarray(s2)
    b = jnp.asarray(b)

    # safe_sigma used only to compute z when sigma == 0 (avoid division by zero).
    # Value chosen arbitrarily (1.0) because z won't be used in the degenerate branch.
    sigma = jnp.sqrt(s2)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    z = (b - m) / safe_sigma

    # Standard normal CDF via erf for numerical stability and JAX compatibility
    sqrt2 = jnp.sqrt(2.0)
    Phi = 0.5 * (1.0 + jsp.special.erf(z / sqrt2))

    # Standard normal PDF
    inv_sqrt_2pi = 1.0 / jnp.sqrt(2.0 * jnp.pi)
    phi = inv_sqrt_2pi * jnp.exp(-0.5 * z * z)

    # analytic expectation (valid when sigma > 0)
    result_continuous = b + (m - b) * Phi - sigma * phi

    # degenerate case: sigma == 0 -> X == m almost surely => E[min(X,b)] = min(m, b)
    result = jnp.where(sigma > 0.0, result_continuous, jnp.minimum(m, b))

    return result


def log_clipped_lognormal_mean(m: ArrayLike, s2: ArrayLike, b: ArrayLike) -> Array:
    """Compute log E[min(exp(x), exp(b))] for x ~ N(m, s2), JAX-compatible and vectorized.

    The inputs `m`, `s2`, and `b` may be scalars or arrays (broadcastable shapes).
    The returned value is the elementwise natural logarithm of the expectation:

        log E[min(e^x, e^b)].

    Analytic decomposition used:

        E[min(e^x, e^b)] = e^b * P(x >= b) + E[e^x 1_{x < b}]
                          = e^b * (1 - Φ(z1)) + exp(m + s2/2) * Φ(z2),

    where z1 = (b - m) / σ and z2 = (b - m - s2) / σ, with σ = sqrt(s2).
    We compute the logarithm using stable `log` and `logaddexp` operations and
    stable evaluations of log-Φ and log-(1-Φ) in the tails.

    Degenerate case (s2 == 0): x == m deterministically so
        E[min(e^x, e^b)] = exp(min(m, b))  =>  log E = min(m, b).

    Args:
        m: Mean(s) of the normal variable x (scalar or array).
        s2: Variance(s) of the normal variable x (scalar or array).
        b: Log clipping bound(s) (returns log E[min(e^x, e^b)]).

    Returns:
        An array (or scalar) with the same broadcasted shape containing
        log E[min(e^x, e^b)].
    """
    m = jnp.asarray(m)
    s2 = jnp.asarray(s2)
    b = jnp.asarray(b)

    # sigma, but guard zero variance to avoid division by zero
    sigma = jnp.sqrt(s2)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    # z-values
    z1 = (b - m) / safe_sigma
    z2 = (b - m - s2) / safe_sigma

    # log term corresponding to e^b * (1 - Phi(z1))
    log_term1 = b + norm.logsf(z1)

    # log term corresponding to exp(m + s2/2) * Phi(z2)
    log_term2 = m + 0.5 * s2 + norm.logcdf(z2)

    # log-sum-exp of two positive contributions (elementwise)
    log_sum = jnp.logaddexp(log_term1, log_term2)

    # For degenerate case (s2 == 0): log E = min(m, b)
    result = jnp.where(sigma > 0.0, log_sum, jnp.minimum(m, b))

    return result

    
def _sample_batch_gaussian_tril(key: PRNGKey, L: Array, m: ArrayLike = 0.0, n: int = 1):
    """Return n samples from N(m, LL^T), where m and L can be batched.

    L: shape (b, d, d) or (d, d)
    m: shape (d,), (b, d), or scalar
    Returns: shape (n, b, d) [n samples from batch of b d-dimensional Gaussians]
    """
    L = jnp.asarray(L)

    # Promote (d, d) to (1, d, d)
    if L.ndim == 2:
        L = L[None, :, :]
    elif L.ndim != 3:
        raise ValueError('L must have shape (d, d) or (b, d, d)')

    b = L.shape[0]
    d = L.shape[1]
    m = jnp.broadcast_to(jnp.asarray(m), (b, d))

    samp = L @ jr.normal(key, shape=(b, d, n)) # (b, d, n)
    samp = m[:, :, None] + samp

    return jnp.transpose(samp, axes=(2, 0, 1)) # (n, b, d)


def _sample_gaussian_tril(key: PRNGKey, L: Array, m: ArrayLike = 0.0, n: int = 1):
    """Return n samples from N(m, LL^T). Out shape: (n, d)
    
    Notes:
        _sample_batch_gaussian_tril() is intended to replace this function with
        a batched generalization. This function is maintained for backwards 
        compatibility.
    """
    d = L.shape[0]
    samp = L @ jr.normal(key, shape=(d,n))
    return jnp.asarray(m).ravel() + samp.T

def _gaussian_log_density_tril(x: Array, m: Array, L: Array):
    """ Batched evaluation of log Gaussian density

    Evaluates d-dimensional multivariate normal log density 
    log N(x | m, L @ L.T), where x, m, and L can all
    be batched over n values.

    Args:
        x: input points with shape (d,) or (n, d)
        m: Gaussian mean with shape (d,) or (n, d)
        L: Lower Cholesky factors, with shape (d, d) or (n, d, d).

    Returns:
        log density evaluations of shape (n,).
    """
    x = jnp.atleast_2d(x) - m
    d = x.shape[1]
    L = jnp.broadcast_to(L, (x.shape[0], d, d))
    Linv_x = solve_triangular(L, x, lower=True)
    mah2 = jnp.sum(Linv_x ** 2, axis=1)
    log_det_term = _gaussian_log_det_term_tril(L)
    return log_det_term - 0.5 * mah2


def _gaussian_log_det_term_tril(L):
    """
    The log of the determinant term in the Gaussian density. 
    log{det(2*pi*C)^{-1/2}} = -0.5 * d * log(2*pi) - 0.5 * log{det(C)},
    where C = LL^T.

    Vectorized so that L can be (n, d, d), in which case the return value
    is (n,). If argument is (d, d) then returns 0d array.

    Notes:
        This term also represents an upper bound on the log density.
    """
    if L.ndim == 2:
        L = L[None]

    d = L.shape[-1]
    dim_times_two_pi = d * jnp.log(2.0 * jnp.pi)
    diag_vmap = jax.vmap(jnp.diag)
    log_det_cov = 2.0 * jnp.log(diag_vmap(L)).sum(axis=1)
    result = -0.5 * (dim_times_two_pi + log_det_cov)

    return result.squeeze()