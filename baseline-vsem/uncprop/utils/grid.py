# uncprop/utils/grid.py
"""
Utility functions for working with discretized probability densities 
over a uniform set of grid points; i.e., the area of each cell in the 
grid is constant.

Functions use the naming convention that logp may represent (log)  
densities or point masses, while log_prob represents masses. 
logZ is used to denote (log) normalizing constants.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Sequence, Callable, Mapping

from jax.scipy.special import logsumexp
from jax.lax import cumlogsumexp

from ott.geometry import grid as ott_grid
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem

from uncprop.custom_types import Array, ArrayLike, PRNGKey
from uncprop.core.inverse_problem import Distribution
from uncprop.utils.plot import smart_subplots

class Grid:

    def __init__(self,
                 low: ArrayLike,
                 high: ArrayLike,
                 n_points_per_dim: ArrayLike,
                 dim_names: list[str] | None):
        low = jnp.asarray(low).ravel()
        high = jnp.asarray(high).ravel()
        n_points_per_dim = jnp.asarray(n_points_per_dim, dtype=int).ravel()

        n_dims = len(low)
        if (len(high) != n_dims) or (len(n_points_per_dim) != n_dims):
            raise ValueError('low, high, and n_points_per_dim must have equal length.')
        
        if dim_names is None:
            dim_names = [f'x{i}' for i in range(n_dims)]
        else:
            if len(dim_names) != n_dims:
                raise ValueError(f'dim_names must have length equal to n_dims = f{n_dims}')

        # list of coordinates for each dimension
        coords = [
            jnp.linspace(l, h, n) for (l, h, n) in zip(low, high, n_points_per_dim)
        ]

        # two grid representations
        grid_arrays = jnp.meshgrid(*coords, indexing='xy')
        flat_grid = jnp.stack([a.ravel() for a in grid_arrays], axis=-1)

        # grid spacing
        dx = (high - low) / (n_points_per_dim - 1)

        self.low = low
        self.high = high
        self.n_points_per_dim = n_points_per_dim
        self.dx = dx
        self.dim_names = dim_names
        self.coords = coords
        self.grid_arrays = grid_arrays
        self.flat_grid = flat_grid

    @property
    def cell_area(self) -> float:
        return float(jnp.prod(self.dx))
    
    @property
    def n_points(self) -> int:
        return int(jnp.prod(self.n_points_per_dim))
    
    @property
    def n_dims(self):
        return len(self.n_points_per_dim)
    
    @property
    def shape(self):
        return self.n_points_per_dim
    
    def plot(self, 
             f: Callable | None = None,
             z: Array | list[Array] | None = None,
             titles: list[str] | None = None,
             points: ArrayLike | None = None,
             ax: Axes | None = None,
             **kwargs) -> tuple[Figure, Sequence[Axes]]:
        """
        Visualize function values at grid points. Values must either be specified 
        directly via `z`, or a function `f` must be supplied and the values 
        obtained via `f(flat_grid)`.
        """
        
        if not ((f is None) ^ (z is None)):
            raise ValueError('Exactly one of f and z must be provided.')
        if z is None:
            z = f(self.flat_grid)

        # validate that z can be interpreted as multiple sets of values over the grid
        if isinstance(z, list):
            z = [arr.ravel() for arr in z]
            z = jnp.stack(z, axis=1)
        else:
            z = jnp.asarray(z)

        if z.ndim > 2:
            raise ValueError('z cannot have more than 2 dimensions.')
        if z.ndim < 2:
            z = z.reshape(-1, 1)
        if z.shape[0] != self.n_points:
            raise ValueError(f'Plot z values have length {z.shape[0]}; expected {self.n_points}')

        if self.n_dims == 1:
            return self._plot_1d(z, titles=titles, points=points, ax=ax, **kwargs)
        elif self.n_dims == 2:
            return self._plot_2d(z, titles=titles, points=points, ax=ax, **kwargs)
        else:
            raise NotImplementedError(f'No plot() method defined for grid with n_dims = {self.n_dims}')
        
    def plot_kde(self, samples: Array, ax=None, **kwargs):
        if self.n_dims == 2:
            return self._plot_2d_kde(samples, ax=ax, **kwargs)
        else:
            raise NotImplementedError(f'No plot_kde() method defined for grid with n_dims = {self.n_dims}')

    def _plot_2d_kde(self, samples: Array, contours: bool = False, ax=None, **kwargs):
        assert self.n_dims == 2

        # Fit KDE and evaluate on grid
        kde = gaussian_kde(samples.T)
        zz = kde(self.flat_grid.T).reshape(self.shape)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        low, high = self.low, self.high
        im = ax.imshow(
            zz,
            origin='lower',
            extent=(low[0], high[0], low[1], high[1]),
            aspect='auto',
        )

        if contours:
            xx, yy = self.grid_arrays
            ax.contour(xx, yy, zz, levels=10, linewidths=1.0)

        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.colorbar(im, ax=ax, label='Density')

        return fig, ax

    def _plot_1d(self,
                 z: Array,
                 titles: str | list[str] | None = None,
                 points: ArrayLike | None = None,
                 legend: bool = True,
                 colors: Mapping[str, str] | None = None,
                 ax: Axes | None = None,
                 **kwargs) -> tuple[Figure, Sequence[Axes]]:
        assert self.n_dims == 1

        X = self.grid_arrays[0]
        Z = z.reshape(X.shape[0], -1)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        for i in range(Z.shape[1]):
            label = None if titles is None else titles[i]
            color = colors[label] if (colors is not None and label in colors) else None
            ax.plot(X, Z[:,i], linestyle='-', label=label, color=color)

        if points is not None:
            y0,y1 = ax.get_ylim()
            color = colors['aux'] if (colors is not None and 'aux' in colors) else None
            ax.vlines(points, 0, 1, transform=ax.get_xaxis_transform(), linestyles='--', colors=color)

        if legend:
            ax.legend()

        return fig, ax

    def _plot_2d(self,
                 z: Array,
                 titles: str | list[str] | None = None, 
                 points: ArrayLike | None = None,
                 **kwargs) -> tuple[Figure, Sequence[Axes]]:
        """
        Note: for now the same points are plotted on each plot
        Currently does not support plotting on existing axes
        """
        assert self.n_dims == 2

        nplots = z.shape[1]
        fig, axs = smart_subplots(nplots=nplots, **kwargs)
        if titles is None:
            titles = [f'z{i}' for i in range(nplots)]

        for i, nm, ax in zip(range(nplots), titles, axs):
            self._plot_single_2d(z=z[:,i], title=nm, points=points, ax=ax)

        return fig, axs


    def _plot_single_2d(self, z, title=None, points=None, ax=None):
        assert self.n_dims == 2

        X, Y = self.grid_arrays
        Z = z.reshape(X.shape)
        fig, ax = plot_2d_heatmap(X, Y, Z, 
                                  dim_names=self.dim_names,
                                  title=title, points=points, ax=ax)

        return fig, ax


class DensityComparisonGrid:

    def __init__(self, 
                 grid: Grid, 
                 distributions: Mapping[str, Distribution] | None = None,
                 log_dens_grid: Mapping[str, Array] | None = None):
        if not (distributions is None) ^ (log_dens_grid is None):
            raise ValueError('DensityComparison grid requires exactly one of `distributions` and `log_dens_grid` to be provided.')

        self.grid = grid
        self.distributions = distributions
        self.log_dens_grid = log_dens_grid

        # unnormalized log densities over grid
        if self.log_dens_grid is None:
            self.log_dens_grid = {
                nm: dist.log_density(self.grid.flat_grid).squeeze() for nm, dist in self.distributions.items()
            }

        # normalized log densities over grid
        self.log_dens_norm_grid = {
            nm: normalize_density_over_grid(logp=logp, cell_area=self.grid.cell_area)[0].squeeze() 
            for nm, logp in self.log_dens_grid.items()
        }

    @property
    def distribution_names(self):
        return list(self.log_dens_grid.keys())
    
    def calc_coverage(self, 
                      baseline: str, 
                      other: str | list[str] | None = None,
                      probs: Array | None = None):
        """
        `baseline` is the name of the distribution treated as the baseline/exact
        distribution for the coverage caclulations.
        """
        if other is None:
            other = [nm for nm in self.distribution_names if nm != baseline]
        
        log_prob_baseline = self.log_dens_norm_grid[baseline]
        log_prob_other = jnp.vstack([self.log_dens_norm_grid[nm] for nm in other])
        
        log_coverage, probs, mask = coverage_curve_grid(log_prob_true=log_prob_baseline,
                                                        log_prob_approx=log_prob_other,
                                                        probs=probs)
        return log_coverage, probs, mask, other


    def plot_coverage(self, 
                      baseline: str, 
                      other: str | list[str] | None = None,
                      probs: Array | None = None):
        
        log_coverage, probs, _, names = self.calc_coverage(baseline=baseline,
                                                           other=other,
                                                           probs=probs)
        return plot_coverage_curve(log_coverage=log_coverage, 
                                   probs=probs, names=names)


    def plot(self, 
             dist_names: str | list[str] | None = None, 
             normalized: bool = False, 
             log_scale: bool = True,
             points: ArrayLike | None = None,
             ax: Axes | None = None,
             **kwargs) -> tuple[Figure, Sequence[Axes]]:
        
        if isinstance(dist_names, str):
            dist_names = [dist_names]
        elif dist_names is None:
            dist_names = self.distribution_names

        plot_vals = self.log_dens_norm_grid if normalized else self.log_dens_grid
        plot_vals = [plot_vals[nm] for nm in dist_names]

        if not log_scale:
            plot_vals = [jnp.exp(arr) for arr in plot_vals]

        return self.grid.plot(z=plot_vals, titles=dist_names, points=points, ax=ax, **kwargs)

    def calc_wasserstein2(self, dist_name1, dist_name2, epsilon=None, **kwargs):
        """
        Discrete approximation of 2-Wasserstein using Sinkhorn.
        """
        densities = self.log_dens_norm_grid
        p_dist = jnp.exp(densities[dist_name1])
        q_dist = jnp.exp(densities[dist_name2])

        geom = ott_grid.Grid(x=self.grid.coords, epsilon=epsilon)
        prob = linear_problem.LinearProblem(geom, a=p_dist, b=q_dist)

        solver = sinkhorn.Sinkhorn(**kwargs)
        out = solver(prob)

        print(out.converged)

        return jnp.sqrt(out.primal_cost)
    

    def sample_from_grid(self, key: PRNGKey, dist_name: str, num_samples: int):

        probs = jnp.exp(self.log_dens_norm_grid[dist_name])

        idx = jr.choice(
            key,
            a=self.grid.n_points,
            shape=(num_samples,),
            p=probs,
            replace=True,
        )
        return self.grid.flat_grid[idx]


# -----------------------------------------------------------------------------
# Helper grid functions 
# -----------------------------------------------------------------------------

def normalize_density_over_grid(logp: Array, 
                                *,
                                cell_area: float | None = None, 
                                return_log: bool = True) -> tuple[Array, Array]:
    """ Normalize density over equally-spaced grid

    If `cell_area` is `None` then `logp` is interpreted as an array
    of point masses. If `cell_area` is a positive float, then this value
    is multiplied to density values to convert to masses of the grid cells.
    
    Args:
        logp: array containing log densities or masses. Can be 1d if
               representing values of single density, or 2d in which case
               each row will be normalized.
        cell_area: positive float or None, see above. Note that for 1d grids area
                   is simply length.
        return_log: If true, returns log of normalized density.

    Returns:
        tuple, containing:
          - (log) normalized version of `logp`
          - (log) normalizing constants
        Returns values on log scale if `return_log` is True. Tuple values are
        arrays of shape (n,), where n is the number of rows of logp
        (1 if logp is flat array).
    """
    logp = _check_grid_batch(logp)

    # convert from densities to masses
    if cell_area is not None:
        if not (jnp.isscalar(cell_area) and cell_area > 0):
            raise ValueError('cell_area must be a positive scalar or None')
        log_prob = logp + jnp.log(cell_area)
    else:
        log_prob = logp

    # log normalizing constant for each row
    logZ = logsumexp(log_prob, axis=1)

    # normalize
    log_prob_norm = log_prob - logZ[:,jnp.newaxis]
    logZ = logZ.ravel()

    if return_log:
        return (log_prob_norm, logZ)
    else:
        return (jnp.exp(log_prob_norm), jnp.exp(logZ))
    

def get_grid_coverage_mask(log_prob: Array, 
                           *,
                           probs: Array | None = None, 
                           check_normalized: bool = True,
                           normalized_log_tol: float = 1e-10):
    """
    Given normalized log probabilities, return indices of elements that 
    correspond to highest density regions at specified probability levels. 
    These regions are not guaranteed to be congiguous.

    Args:
        log_prob: (d,) or (n,d), representing n sets of log normalized probilities
                  over a finite set of d points.
        probs: array of probability levels. Defaults to ten evenly spaced points in [0.1, 0.99].
        check_normalized: If True, checks if each row of log_prob is normalized.
        normalized_log_tol: tolerance in log space for checking normalization.

    Returns:
        Boolean array of shape (n, n_probs, d). Element [i, j, :] gives the coverage
        mask for the ith density at probability level probs[j].
    """
    log_prob = _check_grid_batch(log_prob)

    if probs is None:
        probs = jnp.linspace(0.1, 0.99, 10)
    else:
        probs = jnp.asarray(probs).ravel()
        if jnp.any(probs < 0) or jnp.any(probs > 1):
            raise ValueError('probs must lie in [0,1]')
        
    if check_normalized:
        _check_normalized(log_prob, normalized_log_tol)

    order = jnp.argsort(log_prob, descending=True, axis=1)
    inv_order = jnp.argsort(order, axis=1)
    log_prob_sorted = jnp.take_along_axis(log_prob, order, axis=1)
    log_cum_prob = cumlogsumexp(log_prob_sorted, axis=1)
    
    m = probs.shape[0]
    n, d = log_prob.shape
    logp_lvls = jnp.log(probs)
    cond = (log_cum_prob[:, None, :] >= logp_lvls[None, :, None]) # shape (n, m, d)

    # For each (n, m) find smallest k s.t. cum_prob >= p). Set to -1 if never satisfied.
    any_true = jnp.any(cond, axis=2)  # shape (n, m)
    first_idx = jnp.argmax(cond, axis=2)
    first_idx = jnp.where(any_true, first_idx, -1)

    # Build sorted masks: for each k include indices 0..k (inclusive) (all False if k=1)
    idx_range = jnp.arange(d)  # shape (d,)
    mask_sorted = (idx_range[None, None, :] <= first_idx[:, :, None]) # shape (n, m, d)
    inv_idx_for_take = jnp.broadcast_to(inv_order[:, None, :], (n, m, d))
    mask = jnp.take_along_axis(mask_sorted, inv_idx_for_take, axis=2)

    return mask


def coverage_curve_grid(log_prob_true: ArrayLike,
                        log_prob_approx: ArrayLike,
                        *,
                        probs: Array | None = None,
                        check_normalized: bool = True,
                        normalized_log_tol: float = 1e-10) -> tuple[Array, Array, Array]:
    """
    Given (n, n_grid) arrays of normalized log probability values (each row 
    is normalized), computes highest probability density (HPD) regions at the
    probability levels `probs` using `log_prob_approx`, then computes the log 
    probability of each region under `log_prob_true`. `log_prob_true` and
    `log_prob_approx` must be broadcastable. A common case is that `log_prob_true`
    is a 1d array (multiple approximating distributions, one true baseline).

    Note that the log probabilities must be normalized. By default a check is done
    to ensure this.

    Returns:
        tuple, with elements:
            - (n, n_prob) array, where the (i,j) element is the log coverage of the nth
              distribution at probability level probs[j].
            - the `probs` array of shape (n_prob,)
            - the (n, n_prob, n_grid) boolean mask returned by get_grid_coverage_mask().
    """

    log_prob_true = _check_grid_batch(log_prob_true)
    log_prob_approx = _check_grid_batch(log_prob_approx)

    if check_normalized:
        _check_normalized(log_prob_true, normalized_log_tol)
        _check_normalized(log_prob_approx, normalized_log_tol)

    if probs is None:
        probs = jnp.linspace(0.1, 0.99, 10)
    else:
        probs = jnp.asarray(probs).ravel()
        if jnp.any(probs < 0) or jnp.any(probs > 1):
            raise ValueError('probs must lie in [0,1]')
        
    # compute coverage masks for approximating distribution
    masks = get_grid_coverage_mask(log_prob=log_prob_approx,
                                   probs=probs, check_normalized=False)

    # compute probability in the approximate HPD regions under the true distribution; return (n, n_prob)
    log_coverage = logsumexp(
        log_prob_true[:, jnp.newaxis, :] + jnp.where(masks, 0.0, -jnp.inf), # (n, n_prob, n_grid)
        axis=-1
    )

    return log_coverage, probs, masks

    
def _is_normalized(log_prob: Array, 
                   log_tol: float = 1e-10,
                   verbose: bool = False) -> Array:
    """
    Check if density is normalized, given log probabilities. Vectorized
    to operate over rows of `log_prob`.
    """
    log_prob = _check_grid_batch(log_prob)
    logZ = logsumexp(log_prob, axis=1)

    n = log_prob.shape[0]
    is_normalized = jnp.tile(False, n)
    is_finite = jnp.isfinite(logZ)
    is_normalized = is_normalized.at[is_finite].set(
        jnp.less_equal(jnp.abs(logZ[is_finite]), log_tol)
    ) 

    if verbose and not jnp.all(is_normalized):
        n_failed = jnp.sum(~is_normalized)
        max_err = jnp.max(jnp.abs(logZ[is_finite]))
        print(f'{n_failed} log_probs not normalized. Maximum absolute log integral = {max_err}')

    return is_normalized


def _check_normalized(log_prob: Array, log_tol: float = 1e-10) -> None:
    is_normalized = jnp.all(_is_normalized(log_prob, log_tol, verbose=True))
    if not is_normalized:
        raise ValueError('log_prob is not normalized.')


def _check_grid_batch(x: ArrayLike) -> Array:
    """ 
    Converts to shape (n,d), where each row represents a discretized 
    density over the same grid points. Converts (d,) input to (1,d).
    """
    x = jnp.asarray(x)
    if x.ndim > 2:
        raise ValueError('Invalid input: x.ndim > 2')
    
    return jnp.atleast_2d(x)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_2d_heatmap(X, Y, Z, dim_names, title=None, points=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # pcolormesh expects X, Y, Z of shape (ny, nx)
    pcm = ax.pcolormesh(X, Y, Z, shading='auto')

    # optionally add points
    if points is not None:
        points = np.atleast_2d(points)
        ax.plot(points[:,0], points[:,1], 'o')

    fig.colorbar(pcm, ax=ax)
    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])
    ax.set_title(title)

    return fig, ax


def plot_2d_mask(mask: Array, 
                 grid_shape: tuple[int, int],
                 prob_idx: ArrayLike | None = None):
    """
    Plots coverage mask, as returned by get_grid_coverage_mask()
    """

    if mask.ndim != 3:
        raise ValueError('mask should be 3d, as returned by `get_grid_coverage_mask()`.')
    if prob_idx is not None:
        prob_idx = jnp.atleast_1d(prob_idx)
        mask = mask[:, prob_idx, :]

    n, m, d = mask.shape
    d1, d2 = grid_shape
    if d != d1 * d2:
        raise ValueError(f'Grid shape {grid_shape} and flat mask length {d} disagree.')
    
    fig, axs = plt.subplots(nrows=n, ncols=m)
    axs = np.atleast_2d(axs)

    for i in range(n):
        for j in range(m):
            ax = axs[i,j]
            mask_grid = mask[i,j,:].reshape(d1, d2).astype(int)
            ax.imshow(mask_grid, origin='lower', interpolation='nearest')
    
    return fig, axs


def plot_2d_kde(samp, flat_grid, low, high, contours=True, gridsize=200):
    """
    Plot a 2D kernel density estimate as a heatmap, with optional contours.

    Parameters
    ----------
    samp : array_like, shape (n, 2)
        Samples from a 2D distribution.
    low : array_like, shape (2,)
        Lower bounds of the support (x_min, y_min).
    high : array_like, shape (2,)
        Upper bounds of the support (x_max, y_max).
    contours : bool, default=True
        Whether to overlay contour lines on the heatmap.
    gridsize : int, default=200
        Number of grid points per dimension used for the KDE evaluation.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The created axes.
    """
    samp = np.asarray(samp)
    low = np.asarray(low)
    high = np.asarray(high)

    if samp.ndim != 2 or samp.shape[1] != 2:
        raise ValueError("samp must have shape (n, 2)")
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError("low and high must have shape (2,)")

    # Create evaluation grid
    x = np.linspace(low[0], high[0], gridsize)
    y = np.linspace(low[1], high[1], gridsize)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    # Fit KDE and evaluate on grid
    kde = gaussian_kde(samp.T)
    zz = kde(grid).reshape(gridsize, gridsize)

    # Plot
    fig, ax = plt.subplots()

    im = ax.imshow(
        zz,
        origin="lower",
        extent=(low[0], high[0], low[1], high[1]),
        aspect="auto",
    )

    if contours:
        ax.contour(
            xx,
            yy,
            zz,
            levels=10,
            linewidths=1.0,
        )

    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(im, ax=ax, label="Density")

    return fig, ax
