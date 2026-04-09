import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Numerically stable helpers
# ----------------------------
def _logsumexp(vec):
    """Stable log-sum-exp for 1D array -> scalar (returns -inf for empty or all -inf)."""
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return -np.inf
    m = np.max(vec)
    if not np.isfinite(m):
        return -np.inf
    s = np.sum(np.exp(vec - m))
    return m + np.log(s)

def _logcumsumexp_1d(logx):
    """
    Compute cumulative sums in log-space of a 1D array logx:
    out[i] = log(sum_{j=0..i} exp(logx[j])).
    """
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

# ----------------------------
# Normalizer (works with 1D or 2D inputs)
# ----------------------------
def _normalize_over_grid(log_dens, cell_area=None, *, return_log=True):
    """
    Normalize `log_dens` over a grid. If `cell_area` is provided (positive scalar),
    treats inputs as log-densities and adds log(cell_area) before normalization.
    If inputs are already normalized log-masses (sum exp ~= 1), the function
    will detect that and skip re-normalization (so cell_area won't be reapplied).
    Returns (log_masses, logZ) if return_log True, else (masses, Z).
    """
    log_dens = np.asarray(log_dens)
    single = (log_dens.ndim == 1)
    log2d = np.atleast_2d(log_dens)  # shape (n, d) where n=1 if input was 1D

    def _row_is_normalized(row):
        s = np.sum(np.exp(row))
        return np.isfinite(s) and np.allclose(s, 1.0, atol=1e-8)

    # If single row and already normalized, skip area addition and normalization
    if single and _row_is_normalized(log2d[0]):
        logZ = np.array([0.0])
        log_norm = log2d.copy()
    else:
        if cell_area is not None:
            if not (np.isscalar(cell_area) and cell_area > 0):
                raise ValueError("cell_area must be a positive scalar or None")
            lp = log2d + np.log(cell_area)
        else:
            lp = log2d
        # compute row-wise logZ (sum over columns)
        # use _logsumexp on each row
        logZ_list = np.array([_logsumexp(row) for row in lp])
        # normalize (if a row's logZ == -inf it will produce -inf's)
        log_norm = lp - logZ_list[:, np.newaxis]
        logZ = logZ_list

    if single:
        log_norm = log_norm.ravel()
        logZ = logZ.item()

    if return_log:
        return log_norm, logZ
    else:
        dens = np.exp(log_norm)
        Z = np.exp(logZ) if np.isscalar(logZ) else np.exp(logZ)
        return dens, Z

# ----------------------------
# Coverage function (LOG-SCALE)
# ----------------------------
def coverage_curve(
    logp_true,
    logp_approx,
    cell_area=None,
    alphas=None,
    *,
    return_masks=False,
    expand_ties=False,
    normalized_tol=1e-8
):
    """
    Compute coverage curve on a discrete grid — everything done in LOG-space.

    Returns
    -------
    alphas : array
    log_coverage : array
        For each alpha, log_coverage[i] = log( true_mass( HPD_approx(alpha) ) ).
    calib_error_log : array
        log_coverage - log(alpha). For alpha == 0 the entry is np.nan.
    masks (optional) : list of boolean arrays, only if return_masks=True

    Options
    -------
    return_masks : bool
        If True, also return the list of masks used for each alpha.
    expand_ties : bool
        If True, when the alpha threshold falls on a level set, include the full level set
        (all grid cells with logp_approx >= threshold). If False, include the minimal
        number of top-ranked cells that reach cumulative >= alpha.
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.99, 99)
    alphas = np.asarray(alphas, dtype=float)
    if np.any(alphas < 0) or np.any(alphas > 1):
        raise ValueError("alphas must lie in [0,1]")

    # Flatten and check
    logp_t = np.asarray(logp_true).ravel()
    logp_a = np.asarray(logp_approx).ravel()
    if logp_t.shape != logp_a.shape:
        raise ValueError("logp_true and logp_approx must have the same number of elements")
    n = logp_t.size

    # If inputs already normalized (sum exp ~ 1), skip area; else apply normalization
    def _is_normed(logp):
        s = np.sum(np.exp(logp))
        return np.isfinite(s) and abs(s - 1.0) <= normalized_tol

    if _is_normed(logp_t):
        logp_t_norm = logp_t.copy()
        logZ_t = 0.0
    else:
        logp_t_norm, logZ_t = _normalize_over_grid(logp_t, cell_area, return_log=True)

    if _is_normed(logp_a):
        logp_a_norm = logp_a.copy()
        logZ_a = 0.0
    else:
        logp_a_norm, logZ_a = _normalize_over_grid(logp_a, cell_area, return_log=True)

    if not np.isfinite(logZ_t):
        raise ValueError("true log-probabilities are all -inf or non-finite")
    if not np.isfinite(logZ_a):
        raise ValueError("approx log-probabilities are all -inf or non-finite")

    # Precompute descending order and sorted log-approx
    order = np.argsort(logp_a_norm)[::-1]          # indices from highest to lowest logp
    logp_sorted = logp_a_norm[order]
    log_cum = _logcumsumexp_1d(logp_sorted)       # log cumulative sums in sorted order

    log_coverage = np.empty_like(alphas, dtype=float)
    calib_error_log = np.empty_like(alphas, dtype=float)
    masks = [] if return_masks else None

    for i, alpha in enumerate(alphas):
        if alpha <= 0:
            # log coverage is -inf (zero mass); calib error undefined -> nan
            log_coverage[i] = -np.inf
            calib_error_log[i] = np.nan
            if return_masks:
                masks.append(np.zeros(n, dtype=bool))
            continue
        if alpha >= 1:
            # full set
            mask = np.ones(n, dtype=bool)
            log_cov = _logsumexp(logp_t_norm)  # should be 0 if normalized, but compute anyway
            log_coverage[i] = log_cov
            calib_error_log[i] = log_cov - np.log(alpha)
            if return_masks:
                masks.append(mask)
            continue

        log_alpha = np.log(alpha)
        # find first position k where log_cum[k] >= log_alpha
        k = None
        for idx, val in enumerate(log_cum):
            if np.isfinite(val) and val >= log_alpha:
                k = idx
                break
        if k is None:
            # did not reach alpha (numerical issues), include all
            mask = np.ones(n, dtype=bool)
        else:
            if not expand_ties:
                sel = order[: (k + 1) ]
                mask = np.zeros(n, dtype=bool)
                mask[sel] = True
            else:
                thr = logp_sorted[k]
                mask = logp_a_norm >= thr

        # compute log coverage via logsumexp of true log-probs inside mask
        if not np.any(mask):
            log_cov = -np.inf
        else:
            log_cov = _logsumexp(logp_t_norm[mask])
        log_coverage[i] = log_cov
        calib_error_log[i] = log_cov - np.log(alpha)
        if return_masks:
            masks.append(mask.copy())

    if return_masks:
        return alphas, log_coverage, calib_error_log, masks
    return alphas, log_coverage, calib_error_log

# ----------------------------
# Visualization helper
# ----------------------------
def plot_hpd_mask(mask, grid_shape, ax=None, title=None):
    """
    Visualize boolean HPD mask (1D) on 2D grid `grid_shape` (H, W).
    """
    mask2 = np.asarray(mask, dtype=bool).reshape(grid_shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask2, origin='lower', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    return ax

# ----------------------------
# Minimal reproducible example
# ----------------------------
def _example_run():
    # Create a 2D grid
    H, W = 50, 60
    xs = np.arange(W)
    ys = np.arange(H)
    X, Y = np.meshgrid(xs, ys)

    # True distribution: Gaussian centered at (cx_true, cy_true)
    cx_true, cy_true = 28.0, 22.0
    sigma_true = 3.0
    logp_true = -0.5 * (((X - cx_true)**2 + (Y - cy_true)**2) / sigma_true**2)

    # Approx distribution: slightly shifted, wider (misspecified)
    cx_approx, cy_approx = 30.0, 20.0
    sigma_approx = 4.0
    logp_approx = -0.5 * (((X - cx_approx)**2 + (Y - cy_approx)**2) / sigma_approx**2)

    # We treat these as log-densities on a unit-area grid
    cell_area = 1.0

    # Choose a few alphas for which we will inspect masks
    alphas = np.array([0.5, 0.9, 0.99])

    # Compute coverage (log-scale) and request masks
    alphas_out, log_cov, calib_log, masks = coverage_curve(
        logp_true.ravel(),
        logp_approx.ravel(),
        cell_area=cell_area,
        alphas=alphas,
        return_masks=True,
        expand_ties=True
    )

    # Print numeric results
    print("alpha   log_coverage   coverage (linear)   calib_error_log")
    for a, lc, ce in zip(alphas_out, log_cov, calib_log):
        print(f"{a:4.2f}   {lc:12.6f}   {np.exp(lc):12.6e}    {ce:12.6f}")

    # Plot masks
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 4))
    if len(alphas) == 1:
        axes = [axes]
    for ax, a, mask in zip(axes, alphas_out, masks):
        plot_hpd_mask(mask, grid_shape=(H, W), ax=ax, title=f"HPD mask (alpha={a:.2f})")

    plt.show()

# run the example when module executed as script
if __name__ == "__main__":
    _example_run()
