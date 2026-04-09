# experiments/vsem/grid_utils.py
"""
Utilities for normalizing densities and computing coverage metrics over 2d grid.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from collections.abc import Sequence

Array = np.typing.NDArray


def logsumexp(x: Array):
    """
    Vectorized: applies logsumexp to each row of `x`.
    Returns -np.inf for rows with inf/nan max or empty.
    Return shape is (n,) for x with n rows.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim > 2:
        raise ValueError('logsumexp() requires x.ndim <= 2.')
    if x.ndim < 2:
        x = x.reshape(1, -1)
    
    num_rows = x.shape[0]
    # If there are any empty rows
    if x.size == 0 or x.shape[1] == 0:
        return np.full(num_rows, -np.inf)
    
    max_by_row = np.max(x, axis=1)
    is_finite = np.isfinite(max_by_row)
    out = np.full(num_rows, -np.inf)

    # only process finite rows
    if np.any(is_finite):
        s = np.sum(np.exp(x[is_finite] - max_by_row[is_finite, np.newaxis]), axis=1)
        out[is_finite] = max_by_row[is_finite] + np.log(s)
    return out


def logcumsumexp_1d(logx):
    logx = np.asarray(logx, dtype=float)
    n = logx.size
    out = np.empty(n, dtype=float)
    if n == 0:
        return out
    out[0] = logx[0]
    for i in range(1, n):
        a = out[i-1]
        b = logx[i]
        if a == -np.inf and b == -np.inf:
            out[i] = -np.inf
        else:
            m = a if a > b else b
            out[i] = m + np.log(np.exp(a - m) + np.exp(b - m))
    return out


def log_probs_are_normalized(logp, tol_log: float = 1e-10) -> tuple[Array, Array]:
    """ Check whether density is (approximately) normalized.

    Given an array of log density values, test whether the exponentiated values
    are normalized; i.e., whether they sum to one. Note that this treats the 
    values as point masses - no correction is made for grid cell size.

    Vectorized to operate over the rows of logp.

    Returns:
        tuple, with values:
            is_normalized: boolean array
            logZ: arrays of log sum of the density values for each density
    """
    logp = np.asarray(logp, dtype=float)
    logZ = logsumexp(logp)

    n = logZ.shape[0]
    is_normalized = np.tile(False, n)
    is_finite = np.isfinite(logZ)
    is_normalized[is_finite] = (abs(logZ[is_finite]) <= tol_log)
    
    return is_normalized, logZ


def normalize_over_grid(log_dens, cell_area: float | None = None, *, 
                        return_log: bool = True) -> tuple[Array, Array]:
    """ Normalize density over equally-spaced 2d grid

    If `cell_area` is `None` then `log_dens` is interpreted as an array
    of point masses. If `cell_area` is a positive float, then this value
    is multiplied to density values to convert to masses of the grid cells.
    
    Args:
        log_dens: array containing log densities or masses. Can be 1d if
            representing values of single density, or 2d in which case
            each row will be normalized.

    Returns:
        tuple, containing:
          - (log) normalized version of `log_dens`
          - (log) normalizing constants
        Returns values on log scale if `return_log` is True. Tuple values are
        arrays of shape (n,), where n is the number of rows of log_dens
        (1 if log_dens is flat array).
    """
    log_dens = np.asarray(log_dens, dtype=float)
    if log_dens.ndim > 2:
        raise ValueError('normalize_over_grid() requires log_dens.ndim <= 2')
    log_dens = np.atleast_2d(log_dens)

    # convert from densities to masses
    if cell_area is not None:
        if not (np.isscalar(cell_area) and cell_area > 0):
            raise ValueError('cell_area must be a positive scalar or None')
        log_prob = log_dens + np.log(cell_area)
    else:
        log_prob = log_dens

    # row-wise logZ
    logZ = logsumexp(log_prob)

    with np.errstate(invalid='ignore'): # may be all -Inf
        log_prob_norm = log_prob - logZ[:, np.newaxis]
    logZ = logZ.ravel()

    if return_log:
        return (log_prob_norm, logZ)
    else:
        return (np.exp(log_prob_norm), np.exp(logZ))


def normalize_if_not(logp: Array, 
                     cell_area: float | None = None, 
                     tol_log: float = 1e-10) -> tuple[Array, Array]:
    """ Check if normalized - normalize if not.

    Wrapper around `normalize_over_grid` that first checks whether `logp` represents
    a normalized set of point masses, up to some numerical tolerance. If not, 
    normalizes them (potentially converting densities to masses using `cell_area`).

    Vectorized to operate over rows of `logp`.
    """
    is_norm, logZ_guess = log_probs_are_normalized(logp, tol_log=tol_log)
    logp = np.atleast_2d(logp)

    logp_norm = logp.copy()
    logZ = logZ_guess.copy()

    logp_norm[~is_norm], logZ[~is_norm] = normalize_over_grid(logp[~is_norm], 
                                                              cell_area=cell_area,
                                                              return_log=True)
    logZ[is_norm] = 0.0

    return logp_norm, logZ


def kl_grid(logp, logq):
    """
    Numerical approximation of the KL divergence over a grid:

    KL(p || q) = int p * (log p - log q) dx.
    
    logp, logq are assumed to be normalized log probability masses.
    """
    p = np.exp(logp)
    integrand = p * (logp - logq)
    kl = integrand.mean()
    return kl


def coverage_curve_grid(
    logp_true: Array,
    logp_approx: Array,
    cell_area: float | None = None,
    probs: Array | None = None,
    *,
    return_masks: bool = True,
    return_weights: bool = True,
    tie_break: str = 'fractional',
    normalized_log_tol: float = 1e-10,
    rng=None
):
    """
    Compute HPD coverage curve on a discrete grid using robust log-space arithmetic.

    This function compares an *approximate* distribution (logp_approx) to a
    *reference/true* distribution (logp_true) defined on the same discrete, evenly-spaced grid.
    All inputs are provided as log-values (unnormalized log-densities). The code
    normalizes in log-space, sorts the approximate distribution to produce HPD sets, 
    and computes the mass of the *true* distribution inside those HPD sets.

    Ties (level sets) are handled robustly: by default a **fractional**
    inclusion of the last level-set is used so the approximating distribution's
    HPD mass equals the probability level exactly (avoiding systematic 
    integer-selection bias). Optionally supports other tie breaking methods
    'random' (shuffle equal values) or 'expand' (include whole level set).

    The densities are by default (`cell_area = None`) treated as log masses. If 
    `cell_area` is a positive float, then `cell_area` is multiplied by the densities
    to convert to cell masses. 

    Args:
        logp_true : array_like
            1D array of log-weights for the *true* distribution (length n = H*W)
        logp_approx : array_like
            1D array of log-weights for the *approximate* distribution (same length)
        cell_area : float or None
            Area of each grid cell. If provided it's applied as +log(cell_area) before normalization.
        probvs : array_like or None
            Array of target HPD masses (in (0,1)). If None, defaults to np.linspace(0.01,0.99,99).
        return_masks : bool
            If True, also return boolean masks (per-prob) of fully included cells.
        return_weights : bool
            If True (default), also return a 2D array `weights` shaped (len(probs), n)
            with fractional inclusion weights in [0,1] for every cell and prob. This
            encodes partial inclusion of the level-set block: 1.0 = fully included,
            0.0 = excluded, intermediate values indicate partial inclusion.
        tie_break : {'fractional','random','expand'}
            - 'fractional' (default): include fractional part of the level set so that
            the approximating mass equals the prob exactly (deterministic).
            - 'random': randomly shuffle equal-valued runs and take integer selection
            (Monte Carlo unbiased if averaged over repeats).
            - 'expand': include entire level-set (may overshoot prob).
        normalized_log_tol : float
            Tolerance in log-space for deciding whether an input is already normalized
            (i.e., |logZ| <= normalized_log_tol implies normalized).
        rng : numpy.random.Generator or None
            Optional RNG to control random tie-breaking.

    Returns:
        tuple (probs, log_coverage, masks, weights)

    Where
    - probs : array of requested probability values (same as input or default)
    - log_coverage : array of log(true_mass(HPD_approx(prob))) for each prob
    - masks : list of boolean arrays (only fully-included cells) if return_masks True. Else None.
    - weights : a float ndarray with shape (len(probs), n) with fractional weights in [0,1]
                if return_weights is True. Else None.

    Notes:
        - `masks` show which cells were fully included deterministically. For fractional
        inclusion, `weights` should be used to compute true coverage or visualization.
        - The function computes true coverage via a weighted sum over the true mass:
            true_mass = sum_i weights[prob_idx, i] * p_true[i],
        where p_true = exp(logp_true_normalized).
    """
    if probs is None:
        probs = np.linspace(0.1, 0.99, 10)
    probs = np.asarray(probs, dtype=float)
    if np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must lie in [0,1]")

    logp_t = np.asarray(logp_true).ravel().astype(float)
    logp_a = np.asarray(logp_approx).ravel().astype(float)
    if logp_t.shape != logp_a.shape:
        raise ValueError("logp_true and logp_approx must have the same number of elements")
    n_grid = logp_t.size

    # normalize log probs, potentially converting densities to cell masses
    logp_t_norm, logZ_t = normalize_if_not(logp_t, cell_area, 
                                           tol_log=normalized_log_tol)
    logp_a_norm, logZ_a = normalize_if_not(logp_a, cell_area, 
                                           tol_log=normalized_log_tol)
    logp_t_norm = logp_t_norm.ravel()
    logp_a_norm = logp_a_norm.ravel()
    logZ_t = logZ_t.item()
    logZ_a = logZ_a.item()

    if not np.isfinite(logZ_t):
        raise ValueError("true log-probabilities are all -inf or non-finite")
    if not np.isfinite(logZ_a):
        raise ValueError("approx log-probabilities are all -inf or non-finite")

    # For weighted sums
    p_t = np.exp(logp_t_norm)

    # sorted descending approximate log-probs
    # for reproducibility with 'random' tie-break, optionally shuffle equal-values
    order = np.argsort(logp_a_norm)[::-1]
    logp_a_sorted = logp_a_norm[order]

    if tie_break == 'random':
        # shuffle equal-value runs to randomize ties
        if rng is None:
            rng = np.random.default_rng()
        # find runs of equal values (within isclose)
        i = 0
        while i < n_grid:
            thr = logp_a_sorted[i]
            # find run end where values close to thr
            j = i + 1
            while j < n_grid and np.isclose(logp_a_sorted[j], thr, atol=0, rtol=1e-12):
                j += 1
            if j - i > 1:
                # random permute indices in this run
                run_idx = np.arange(i, j)
                perm = rng.permutation(run_idx)
                logp_a_sorted[run_idx] = logp_a_sorted[perm]
                order[run_idx] = order[perm]
            i = j

    log_cum = logcumsumexp_1d(logp_a_sorted)  # log of cumulative approx mass in sorted order

    log_coverage = np.empty_like(probs, dtype=float)
    masks = [] if return_masks else None
    weights = np.zeros((probs.size, n_grid), dtype=float) if return_weights else None

    for prob_idx, prob in enumerate(probs):
        if prob <= 0:
            log_coverage[prob_idx] = -np.inf
            if return_masks:
                masks.append(np.zeros(n_grid, dtype=bool))
            if return_weights:
                weights[prob_idx, :] = 0.0
            continue

        if prob >= 1:
            mask_full = np.ones(n_grid, dtype=bool)
            log_cov_full = logsumexp(logp_t_norm)  # should be 0 if normalized
            log_coverage[prob_idx] = log_cov_full
            if return_masks:
                masks.append(mask_full)
            if return_weights:
                weights[prob_idx, :] = 1.0
            continue

        log_prob = np.log(prob)

        # First index where cumulative >= prob
        k = None
        exceeds_prob = (log_cum >= log_prob) & np.isfinite(log_cum)
        if np.any(exceeds_prob):
            k = np.argmax(exceeds_prob)  # first True index
        else:
            k = None

        # shouldn't happen since should sum to one, but possible
        # due to numerical error; fallback to include all
        if k is None:
            mask_full = np.ones(n_grid, dtype=bool)
            log_cov = logsumexp(logp_t_norm[mask_full])
            log_coverage[prob_idx] = log_cov
            if return_masks:
                masks.append(mask_full)
            if return_weights:
                weights[prob_idx, :] = 1.0
            continue

        # find contiguous block of entries equal to threshold
        # (contiguous due to fact that array is sorted)
        threshold = logp_a_sorted[k]
        approx_equal = np.isclose(logp_a_sorted, threshold, rtol=0, atol=1e-12)
        left_idx = k
        while left_idx > 0 and approx_equal[left_idx - 1]:
            left_idx -= 1
        right_idx = k
        while right_idx + 1 < n_grid and approx_equal[right_idx + 1]:
            right_idx += 1

        if tie_break == 'expand':
            sel_sorted_indices = np.arange(0, right_idx + 1)
            mask = np.zeros(n_grid, dtype=bool)
            mask[order[sel_sorted_indices]] = True
            if return_weights:
                weights[prob_idx, :] = 0.0
                weights[prob_idx, order[: (right_idx + 1)]] = 1.0
            log_cov = logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        elif tie_break == 'random':
            sel_sorted_indices = np.arange(0, k + 1)
            mask = np.zeros(n_grid, dtype=bool)
            mask[order[sel_sorted_indices]] = True
            if return_weights:
                weights[prob_idx, :] = 0.0
                weights[prob_idx, order[: (k + 1)]] = 1.0
            log_cov = logsumexp(logp_t_norm[mask]) if np.any(mask) else -np.inf

        else:  # tie_break == 'fractional'
            # compute cumulative mass before the block
            if left_idx == 0:
                cum_before = 0.0
            else:
                log_cum_before = log_cum[left_idx - 1]
                cum_before = float(np.exp(log_cum_before))

            block_log_mass = logsumexp(logp_a_sorted[left_idx: right_idx + 1])
            block_mass = float(np.exp(block_log_mass))

            if block_mass <= 0:
                f = 0.0
            else:
                f = float((prob - cum_before) / block_mass)
                f = np.clip(f, 0.0, 1.0)

            # build weights: 1 for fully included before the block, f for block, 0 otherwise
            if return_weights:
                w = np.zeros(n_grid, dtype=float)
                if left_idx > 0:
                    w[order[:left_idx]] = 1.0
                w[order[left_idx: right_idx + 1]] = f
                weights[prob_idx, :] = w

            # compute true mass via weighted sum in linear space for stability
            if return_weights:
                total_true = float(np.dot(weights[prob_idx, :], p_t))
            else:
                # if weights not requested, fall back to computing true_before and true_block
                if left_idx > 0:
                    idx_before = order[:left_idx]
                    true_before = float(np.sum(p_t[idx_before]))
                else:
                    true_before = 0.0
                idx_block = order[left_idx: right_idx + 1]
                true_block = float(np.sum(p_t[idx_block]))
                total_true = true_before + f * true_block

            if total_true <= 0:
                log_cov = -np.inf
            else:
                log_cov = np.log(total_true)

            # build boolean mask of fully-included cells (before block)
            mask = np.zeros(n_grid, dtype=bool)
            if left_idx > 0:
                mask[order[:left_idx]] = True

        log_coverage[prob_idx] = log_cov
        if return_masks:
            masks.append(mask.copy())

    return probs, log_coverage, masks, weights


def plot_hpd_weights_grid(
    weights: Array,
    grid_shape: tuple[int, int],
    probs: Sequence[float] | None = None,
    *,
    ncols: int | None = None,
    cmap: str | plt.Colormap = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    title_fmt: str = "prob={:.3f}",
    show_colorbar: bool = True,
    suptitle: str | None = None
) -> tuple[Figure, Array]:
    """
    Plot multiple HPD weight maps (one subplot per prob) on a 2D grid.

    This helper expects `weights` to encode fractional inclusion per grid cell,
    values in [0,1]. It supports weights shaped either (m, n) where n == H*W,
    or (m, H, W). Each row (index) corresponds to one prob.

    Args:
        weights: np.ndarray
            Array of fractional weights. Shape should be (m, n) or (m, H, W).
        grid_shape: (H, W)
            Height and width of the 2D grid.
        probs: Optional[Sequence[float]]
            Sequence of length m with the prob values. If None, simple indices
            will be used for titles.
        ncols: Optional[int]
            Number of columns in the figure layout. If None, chosen automatically
            (<= 4 columns).
        cmap: str or Colormap
            Colormap used for plotting fractional weights.
        vmin: float
            Minimum color scale (default 0.0).
        vmax: float
            Maximum color scale (default 1.0).
        figsize_per_panel: (w, h)
            Size of each subplot in inches; figure size = panels * this.
        title_fmt: str
            Python format string used to build subplot titles from prob values.
            Example default: "prob={:.3f}".
        show_colorbar: bool
            Whether to include a single shared colorbar for all panels.
        suptitle: Optional[str]
            Optional overall figure title.

    Returns:
        (fig, axes) :
            - fig: matplotlib.figure.Figure containing the subplots.
            - axes: 2D numpy array of Axes objects with shape (nrows, ncols_used).
    """
    weights = np.asarray(weights, dtype=float)
    H, W = grid_shape
    n_cells = H * W

    # Normalize weights shape -> (m, H, W)
    if weights.ndim == 2:
        m, n = weights.shape
        if n != n_cells:
            raise ValueError("weights shape (m,n) but n != H*W")
        weights_reshaped = weights.reshape((m, H, W))
    elif weights.ndim == 3:
        m, h, w = weights.shape
        if (h, w) != (H, W):
            raise ValueError("weights shape (m,H,W) does not match grid_shape")
        weights_reshaped = weights
    else:
        raise ValueError("weights must have ndim 2 or 3: (m,n) or (m,H,W)")

    # probs labels
    if probs is not None:
        if len(probs) != m:
            raise ValueError("length of probs must equal number of weight maps")
        prob_labels = [title_fmt.format(a) for a in probs]
    else:
        prob_labels = [title_fmt.format(i) for i in range(m)]

    # layout: columns
    if ncols is None:
        ncols = min(4, m) if m > 0 else 1
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(m / ncols))

    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows
    fig, axes_flat = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h),
                                  squeeze=False)

    # flatten to iterate, but keep 2D axes for return
    axes_iter = axes_flat.ravel()
    im = None

    # plot each weights map
    for idx in range(nrows * ncols):
        ax = axes_iter[idx]
        if idx < m:
            arr = weights_reshaped[idx]
            # validate range
            if np.nanmin(arr) < -1e-12 or np.nanmax(arr) > 1.0 + 1e-12:
                raise ValueError(f"Weights at index {idx} not in [0,1]. min={arr.min()}, max={arr.max()}")
            im = ax.imshow(arr, origin='lower', interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(prob_labels[idx])
        else:
            # empty panel
            ax.axis('off')

        ax.set_xticks([])
        ax.set_yticks([])

    # colorbar
    if show_colorbar and im is not None:
        # place a single colorbar on the right side of the entire grid
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # position colorbar next to the last used axes
        last_ax = axes_iter[min(m - 1, len(axes_iter) - 1)]
        divider = make_axes_locatable(last_ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        fig.colorbar(im, cax=cax)

    if suptitle:
        fig.suptitle(suptitle)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if suptitle else None)
    return fig, axes_flat


def plot_coverage(probs, coverage_list, *, labels=None, figsize=(5,4)):
    fig, ax = plt.subplots(figsize=figsize)
    n_curves = len(coverage_list)
    if labels is None:
        labels = [f"Plot {i}" for i in range(n_curves)]

    for j in range(n_curves):
        ax.plot(probs, coverage_list[j], label=labels[j])

    ax.set_title("Coverage")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color="red", linestyle="--")
    ax.legend()

    # Close figure
    plt.close(fig)
    return fig


# TODO: update this
def plot_coverage_distribution(tests: list[VSEMTest], 
                               metrics: list,
                               labels: list,
                               q_min: float = 0.05, 
                               q_max: float = 0.95, 
                               figsize=(12, 4)):
    """
    The first two arguments are those returned by `run_vsem_experiment()`.
    Assumes the same coverage probabilities were used for all replications
    within the experiment.
    """

    # Assumed constrant across all replications
    probs = metrics[0]['coverage']['probs']

    n_reps = len(tests)
    n_probs = len(probs)
    mean_coverage = np.empty((n_reps, n_probs))
    eup_coverage = np.empty((n_reps, n_probs))
    ep_coverage = np.empty((n_reps, n_probs))
    mean_idx = labels.index('mean')
    eup_idx = labels.index('eup')
    ep_idx = labels.index('ep')

    # assemble arrays of coverage stats
    for i, results in enumerate(metrics):
        cover = results['coverage']['cover']
        mean_coverage[i,:] = cover[mean_idx]
        eup_coverage[i,:] = cover[eup_idx]
        ep_coverage[i,:] = cover[ep_idx]

    # summarize distribution over replications
    mean_m = np.median(mean_coverage, axis=0)
    eup_m = np.median(eup_coverage, axis=0)
    ep_m = np.median(ep_coverage, axis=0)
    mean_q = np.quantile(mean_coverage, q=[q_min, q_max], axis=0)
    eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)
    ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)

    meds = [mean_m, eup_m, ep_m]
    qs = [mean_q, eup_q, ep_q]
    labels = ['mean', 'eup', 'ep']
    
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    axs = axs.reshape(-1)
    n_plots = len(axs)

    for i in range(n_plots):
        ax = axs[i]
        q = qs[i]
        med = meds[i]
        label = labels[i]

        ax.fill_between(probs, q[0,:], q[1,:], alpha=0.7)
        ax.plot(probs, med)
        ax.set_title(label)
        ax.set_xlabel('Nominal Coverage')
        ax.set_ylabel('Actual Coverage')

        # Add line y = x
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        y = x
        ax.plot(x, y, color="red", linestyle="--")
        ax.legend()

    plt.close(fig)
    return fig, axs


# TODO: update this
def plot_coverage_same_plot(tests: list[VSEMTest], 
                            metrics: list, 
                            q_min: float = 0.05, 
                            q_max: float = 0.95, 
                            ax=None):
    """
    The first two arguments are those returned by `run_vsem_experiment()`.
    Assumes the same coverage probabilities were used for all replications
    within the experiment.
    """

    # Assumed constrant across all replications
    probs = metrics[0]['coverage']['probs']

    n_reps = len(tests)
    n_probs = len(probs)
    mean_coverage = np.empty((n_reps, n_probs))
    eup_coverage = np.empty((n_reps, n_probs))
    ep_coverage = np.empty((n_reps, n_probs))

    # assemble arrays of coverage stats
    for i, results in enumerate(metrics):
        mean, eup, ep = results['coverage']
        mean_coverage[i,:] = mean
        eup_coverage[i,:] = eup
        ep_coverage[i,:] = ep

    # summarize distribution over replications
    mean_m = np.median(mean_coverage, axis=0)
    eup_m = np.median(eup_coverage, axis=0)
    ep_m = np.median(ep_coverage, axis=0)
    mean_q = np.quantile(mean_coverage, q=[q_min, q_max], axis=0)
    eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)
    ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.fill_between(probs, mean_q[0,:], mean_q[1,:], color='green', alpha=0.3, label='mean')
    ax.fill_between(probs, eup_q[0,:], eup_q[1,:], color='red', alpha=0.3, label='eup')
    ax.fill_between(probs, ep_q[0,:], ep_q[1,:], color='blue', alpha=0.3, label='ep')
    ax.plot(probs, mean_m, color='green', label='mean')
    ax.plot(probs, eup_m, color='red', label='eup')
    ax.plot(probs, ep_m, color='blue', label='ep')
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color="red", linestyle="--")
    ax.legend()
    plt.close(fig)

    return fig, ax