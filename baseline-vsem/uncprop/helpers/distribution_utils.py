from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_normal, norm
from typing import Tuple, Union
import math

ArrayLike = Union[np.ndarray, float, int]

class ClippedGaussian:
    """Distribution of Y = min(X, upper) where X ~ N(mean, cov) and min is elementwise.

    The transform is elementwise: y_i = min(x_i, upper_i). This produces a mixed
    distribution (continuous on coordinates where y_i < upper_i and discrete mass
    at y_i == upper_i).

    Attributes
    ----------
    base_mean : np.ndarray
        Mean vector `m` of the underlying Gaussian (shape (d,)).
    base_cov : np.ndarray
        Covariance matrix `C` of the underlying Gaussian (shape (d,d)).
    upper : np.ndarray
        Upper bound vector (shape (d,)).
    rng_seed : int | None
        Seed for Monte Carlo operations.

    Properties
    ----------
    mean -> np.ndarray
        E[Y] computed exactly per-component using the marginal univariate Gaussian.
    cov -> np.ndarray
        Covariance matrix of Y. Computed by Monte Carlo sampling (configurable).
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        upper: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Construct a ClippedGaussian.

        Parameters
        ----------
        mean
            Underlying Gaussian mean (shape (d,))
        cov
            Underlying Gaussian covariance matrix (shape (d,d))
        upper
            Elementwise upper bound (shape (d,))
        rng
            Optional numpy random generator.
        """
        self.base_mean = np.asarray(mean, dtype=float)
        self.base_cov = np.asarray(cov, dtype=float)
        self.upper = np.asarray(upper, dtype=float)

        if self.base_mean.ndim != 1:
            raise ValueError("mean must be 1-D array")
        d = self.base_mean.shape[0]

        if self.base_cov.shape != (d, d):
            raise ValueError("cov must be (d,d) matrix consistent with mean")
        if self.upper.shape != (d,):
            raise ValueError("upper must be (d,) vector consistent with mean")

        self.rng = rng or np.random.default_rng()
        self._cached_cov: np.ndarray | None = None

    def sample(self, n: int) -> np.ndarray:
        """Sample n draws from Y = min(X, upper) (shape (n,d)).

        Parameters
        ----------
        n
            Number of iid samples to draw.

        Returns
        -------
        samples : np.ndarray
            Array of shape (n, d).
        """
        x = self.rng.multivariate_normal(self.base_mean, self.base_cov, size=n)
        y = np.minimum(x, self.upper)
        return y

    @property
    def dim(self) -> int:
        return self.base_cov.shape[0]

    @property
    def mean(self) -> np.ndarray:
        """E[Y] computed per-dimension using univariate marginals (exact).

        For each coordinate i:
            E[min(X_i, u_i)] = ∫_{-∞}^{u_i} x f_{X_i}(x) dx + u_i * P(X_i >= u_i)
                             = mu_i * Phi(alpha) - sigma_i * phi(alpha) + u_i * (1 - Phi(alpha))
        where alpha = (u_i - mu_i)/sigma_i.
        """
        mu = self.base_mean
        C = self.base_cov
        d = mu.shape[0]
        out = np.empty(d, dtype=float)
        for i in range(d):
            mu_i = mu[i]
            sigma = math.sqrt(max(C[i, i], 0.0))
            if sigma <= 0.0:
                # degenerate: X_i is almost constant
                out[i] = min(mu_i, self.upper[i])
                continue
            alpha = (self.upper[i] - mu_i) / sigma
            Phi = norm.cdf(alpha)
            phi = norm.pdf(alpha)
            # ∫_{-∞}^{u} x f(x) dx = mu * Phi - sigma * phi
            integral = mu_i * Phi - sigma * phi
            out[i] = integral + self.upper[i] * (1.0 - Phi)
        return out

    @property
    def cov(self, n_mc=100_000) -> np.ndarray:
        """Covariance matrix of Y. Approximated via Monte Carlo sampling.

        Computing the exact covariance matrix of the mixed distribution is
        somewhat involved (requires many multivariate integrals). Here we
        estimate it by Monte Carlo with `self.mc_cov_samples` samples and cache
        the result. If you want a more accurate estimate, set a larger
        `mc_cov_samples` in the constructor.
        """
        if self._cached_cov is not None:
            return self._cached_cov
        n = max(10_000, n_mc)
        y = self.sample(n)
        # unbiased sample covariance (n-1)
        cov_est = np.cov(y, rowvar=False, ddof=1)
        self._cached_cov = cov_est
        return cov_est

    @property
    def variance(self) -> np.ndarray:
        """Exact marginal variances Var[Y_i] for Y_i = min(X_i, upper_i).

        Uses only the univariate marginal N(mu_i, sigma_i^2).

        For each i:
            E[Y_i]     = mu * Phi - sigma * phi + u * (1-Phi)
            E[Y_i^2]   = (mu^2 + sigma^2)*Phi - sigma*(mu+u)*phi + u^2*(1-Phi)
            Var[Y_i]   = E[Y_i^2] - E[Y_i]^2

        Returns
        -------
        var : np.ndarray
            Vector of marginal variances (shape (d,)).
        """
        mu = self.base_mean
        C = self.base_cov
        upper = self.upper
        d = mu.shape[0]
        out = np.empty(d, dtype=float)

        EY = self.mean  # exact vector

        for i in range(d):
            mu_i = mu[i]
            sigma2 = C[i, i]
            sigma = math.sqrt(max(sigma2, 0.0))
            u = upper[i]

            if sigma <= 0:
                # degenerate X_i -> deterministic Y_i
                out[i] = 0.0
                continue

            alpha = (u - mu_i) / sigma
            Phi = norm.cdf(alpha)
            phi = norm.pdf(alpha)

            EY2 = (mu_i**2 + sigma2) * Phi - sigma * (mu_i + u) * phi + u**2 * (1 - Phi)

            out[i] = EY2 - EY[i] ** 2

        return out


def logscore_clipped_gaussian(
    y: ArrayLike,
    m: ArrayLike,
    s: ArrayLike,
    b: ArrayLike,
    atol_b: float = 0.0
) -> np.ndarray:
    """
    Vectorized log score for X' = min(X, b) with X ~ N(m, s^2).

    Args:
        y: Observations (scalar or array-like).
        m: Forecast mean (scalar or array-like; broadcasts with y).
        s: Forecast standard deviation (scalar or array-like; must be > 0).
        b: Clipping threshold (scalar or array-like; broadcasts with y).
        atol_b: Absolute tolerance for treating y as equal to b (default 0.0).
                If > 0, values with |y - b| <= atol_b are treated as `y == b`.

    Returns:
        An ndarray of the same broadcasted shape as the inputs containing
        the log predictive probabilities (log score). Values with y > b get -inf.

    Raises:
        ValueError: if any s <= 0 after broadcasting.
    """
    y_a, m_a, s_a, b_a = np.broadcast_arrays(np.asarray(y), np.asarray(m),
                                             np.asarray(s), np.asarray(b))
    if np.any(s_a <= 0):
        raise ValueError("All s must be positive.")

    # masks: continuous (y < b), mass (y == b within tolerance), impossible (y > b)
    if atol_b > 0:
        is_mass = np.abs(y_a - b_a) <= atol_b
    else:
        is_mass = y_a == b_a

    is_cont = (y_a < b_a) & (~is_mass)
    is_impossible = y_a > b_a

    out = np.empty_like(y_a, dtype=float)

    # continuous part: log pdf
    out[is_cont] = norm.logpdf(y_a[is_cont], loc=m_a[is_cont], scale=s_a[is_cont])

    # mass at b: log(1 - Phi((b-m)/s)) computed with logsf for numerical stability
    if np.any(is_mass):
        z = (b_a[is_mass] - m_a[is_mass]) / s_a[is_mass]
        out[is_mass] = norm.logsf(z)   # numerically stable

    # impossible: log prob = -inf
    if np.any(is_impossible):
        out[is_impossible] = -np.inf

    return out


def logscore_clipped_lognormal(
    logy: ArrayLike,
    m: ArrayLike,
    s: ArrayLike,
    b: ArrayLike,
    atol_b: float = 0.0
) -> np.ndarray:
    """
    Vectorized log score for the clipped log-normal predictive distribution
    defined by:
        X' = min(exp(X), exp(b)),  where X ~ N(m, s^2).

    This version accepts 'logy' = log(observation), so that the user does not
    need to compute y = exp(logy) explicitly.

    Args:
        logy: Observed log-values (scalar or array-like). If logy <= -inf
              (i.e., invalid), the predictive probability is zero.
        m: Mean of X (scalar or array-like; broadcasts).
        s: Std. dev. of X (must be > 0; broadcasts).
        b: Log-space clipping threshold (scalar or array-like).
           The point mass is located at logy == b (within tolerance).
        atol_b: Absolute tolerance for treating logy as equal to b.

    Returns:
        ndarray of log predictive probabilities (log scores).
    """
    # Broadcast everything
    logy_a, m_a, s_a, b_a = np.broadcast_arrays(
        np.asarray(logy), np.asarray(m), np.asarray(s), np.asarray(b)
    )

    if np.any(s_a <= 0):
        raise ValueError("All s must be positive.")

    # Initialize output with -inf (default: impossible)
    out = np.full_like(logy_a, -np.inf, dtype=float)

    # Valid observations: logy > -inf => y > 0
    valid = np.isfinite(logy_a)
    if not np.any(valid):
        return out

    # Masks for continuous part and mass part
    if atol_b > 0.0:
        is_mass = valid & (np.abs(logy_a - b_a) <= atol_b)
    else:
        is_mass = valid & (logy_a == b_a)

    is_cont = valid & (logy_a < b_a) & (~is_mass)

    # Continuous part: 0 < y < exp(b)
    # p(y) = Normal(logy | m, s) * 1/exp(logy)
    # log p(y) = logpdf(logy) - logy
    if np.any(is_cont):
        idx = is_cont
        lp = norm.logpdf(logy_a[idx], loc=m_a[idx], scale=s_a[idx])
        out[idx] = lp - logy_a[idx]   # subtract log(y)

    # Mass at exp(b): p = P(X >= b)
    if np.any(is_mass):
        idx = is_mass
        z = (b_a[idx] - m_a[idx]) / s_a[idx]
        out[idx] = norm.logsf(z)   # numerically stable

    # Invalid or logy > b remain -inf by default
    return out


# Validation check
if __name__ == "__main__":
    m = np.array([1., 2.])
    C = np.array([[2.0, -0.8], [-0.8, 1.0]])
    upper = np.array([2., 3.])
    n_mc = int(1e6)
    rng = np.random.default_rng(62435)

    x = multivariate_normal(m, C)
    y = ClippedGaussian(m, C, upper=upper, rng=rng)

    x_samp = x.rvs(size=n_mc)
    y_samp = np.minimum(x_samp, upper)
    mc_mean = y_samp.mean(axis=0)
    mc_var = y_samp.var(axis=0)

    print(f"Mean difference: {np.abs(mc_mean - y.mean)}")
    print(f"SD difference: {np.abs(np.sqrt(mc_var) - np.sqrt(y.variance))}")

