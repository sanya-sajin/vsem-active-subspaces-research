# uncprop/utils/plot.py

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns

import jax.numpy as jnp
import numpy as np
import math
from jax.scipy.stats import norm

from uncprop.custom_types import Array

def set_plot_theme():
    sns.set_theme(style='ticks', palette='colorblind')
    sns.set_context('paper', font_scale=1.5)

    # Specific Paul Tol color scheme when comparing different posteriors
    colors = {
        'exact': "#4477AA",
        'mean': "#EE6677",
        'eup': "#228833",
        'ep': "#CCBB44",
        'aux': "#888888"
    }

    return colors


def smart_subplots(
    nrows=None, ncols=None, figsize=None,
    tight=True, nplots=None, max_cols=None,
    flatten=True, squeeze=True, sharex=False, sharey=False,
    **kwargs
):
    """
    Wrapper for plt.subplots with enhanced features.
    - tight: Use tight_layout if True
    - nrows, ncols, figsize: as in plt.subplots
    - nplots, max_cols: alternative way to choose grid shape (row-major order)
    - flatten: If True (default), always return flat list of axes
    """
 
    # grid specification logic
    if nplots is not None:
        if (nrows is not None) or (ncols is not None):
            raise ValueError("Specify either nplots (with optional max_cols), OR nrows/ncols, not both.")
        if nplots <= 0:
            raise ValueError("nplots must be positive")
        if max_cols is not None and max_cols <= 0:
            raise ValueError("max_cols must be positive if specified")
        
        # row major logic: Fill rows first
        _max_cols = max_cols if max_cols is not None else 1
        ncols = min(nplots, _max_cols)
        nrows = math.ceil(nplots / ncols)
    elif nrows is not None or ncols is not None:
        if nrows is None or ncols is None:
            raise ValueError("If specifying nrows or ncols, both must be provided.")
        if nrows <= 0 or ncols <= 0:
            raise ValueError("nrows and ncols must both be positive")
        nplots = nrows * ncols
    else:
        # Defaults to one plot
        nrows, ncols, nplots = 1, 1, 1

    # Reasonable auto-sizing if not specified
    if figsize is None:
        base_width = 4.0  # inches per subplot
        base_height = 3.0
        width = ncols * base_width
        height = nrows * base_height
        figsize = (width, height)

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=squeeze,
        sharex=sharex,
        sharey=sharey,
        **kwargs
    )

    if tight:
        fig.tight_layout()

    if flatten:
        # Always return as flat list of axes (even if originally an array or single Axes)
        if isinstance(axs, np.ndarray):
            out_axs = axs.flatten().tolist()
        else:
            out_axs = [axs]
        return fig, out_axs
    else:
        return fig, axs
    

def plot_coverage_curve(log_coverage: Array, 
                        probs: Array, 
                        names: list[str] | None = None):
    """
    Args:
        log_coverage: (n, n_prob) array, where the (i,j) element is the log coverage of the ith
                      distribution at probability level probs[j].
        probs: coverage levels in [0, 1], array of shape (n_prob,)
    """
    log_coverage = jnp.atleast_2d(log_coverage)
    n_curves = log_coverage.shape[0]

    if names is None:
        names = [f'dist{i}' for i in range(n_curves)]

    # coverage curves
    fig, ax = plt.subplots()
    for i in range(n_curves):
        ax.plot(probs, jnp.exp(log_coverage[i]), label=names[i])

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, linestyle='--')

    ax.set_xlabel('nominal coverage')
    ax.set_ylabel('actual coverage')
    ax.legend()

    return fig, ax


def plot_coverage_curve_reps(log_coverage: Array, 
                             probs: Array, 
                             names: list[str] | None = None,
                             colors: Mapping[str, str] | None = None,
                             qmin: float = 0.05,
                             qmax: float = 0.95,
                             single_plot: bool = False,
                             alpha: float = 0.3,
                             xlabel: str = 'nominal coverage',
                             ylabel: str = 'actual coverage',
                             set_title: bool = True,
                             ax: Axes | None = None,
                             **kwargs) -> tuple[Figure, Axes]:
    """
    Summarizes an ensemble of coverage curves, typically from multiple replicates
    of an experiment. Arguments are the same as `plot_coverage_curve`
    except that `log_coverage` has a leading batch access; i.e., it is shape
    (n_reps, n_dists, n_probs). `qmin` and `qmax` specify the quantiles defining
    the width of the confidence band.

    Optionally plots lines on single plot, or splits into multiple plots.
    """
    if ax is not None and not single_plot:
        raise ValueError('currently only allows passing `ax` for `single_plot = True`')

    log_coverage = jnp.atleast_3d(log_coverage)
    n_curves = log_coverage.shape[1]
    
    if names is None:
        names = [f'dist{i}' for i in range(n_curves)]

    # compute quantiles for each distribution
    q = jnp.array([qmin, 0.5, qmax])
    quantiles = jnp.quantile(jnp.exp(log_coverage), q=q, axis=0) # (3, n_dists, n_probs)

    if ax is not None:
        fig = ax.figure
        axs = [ax] * n_curves
    elif single_plot:
        fig, ax = plt.subplots()
        axs = [ax] * n_curves
    else:
        fig, axs = smart_subplots(nplots=n_curves, **kwargs)

    for i, ax in enumerate(axs):
        label = names[i]
        color = colors[label] if (colors is not None and label in colors) else None
        ax.fill_between(probs, quantiles[0,i,:], quantiles[2,i,:], 
                        label=label, color=color, alpha=alpha)
        ax.plot(probs, quantiles[1,i,:], color=color)

        if not single_plot and set_title:
            ax.set_title(names[i])

        # Add line y = x
        if i == 0 or not single_plot:
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            y = x
            aux_color = colors['aux'] if (colors is not None and 'aux' in colors) else None
            ax.plot(x, y, linestyle='--', color=aux_color)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
  
    if single_plot:
        axs[0].legend()

    return fig, axs


def plot_marginal_pred_1d(x: Array,
                          mean: Array,
                          lower: Array,
                          upper: Array,
                          colors: Mapping[str, str] | None = None,
                          points: Array | None = None,
                          true_y: Array | None = None,
                          interval_alpha: float = 0.3,
                          ax: Axes | None = None,
                          **kwargs):
    
    if colors is not None:
        mean_color = colors['mean'] if 'mean' in colors else None
        interval_color = colors['interval'] if 'interval' in colors else None
        points_color = colors['points'] if 'points' in colors else None
        true_color = colors['true'] if 'true' in colors else None
    else:
        mean_color = interval_color = points_color = true_color = None

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, mean, linestyle='-', color=mean_color)
    ax.fill_between(x, lower, upper, alpha=interval_alpha, color=interval_color)

    if true_y is not None:
        ax.plot(x, true_y, linestyle='-', color=true_color)
    if points is not None:
        y0,y1 = ax.get_ylim()
        ax.vlines(points, 0, 1, transform=ax.get_xaxis_transform(), linestyles='--', colors=points_color)
    
    return fig, ax


def plot_gp_1d(x: Array,
               mean: Array,
               sd: Array,
               colors: Mapping[str, str] | None = None,
               points: Array | None = None,
               true_y: Array | None = None,
               interval_prob: float = 0.95,
               interval_alpha: float = 0.3,
               ax: Axes | None = None,
               **kwargs):
    """
    If provided, `colors` should have keys 'mean', 'interval',
    'points', 'true'.
    """
    q = (1 - interval_prob) / 2
    shift = jnp.abs(norm.ppf(q))

    return plot_marginal_pred_1d(x=x,
                                 mean=mean,
                                 lower=mean-shift*sd,
                                 upper=mean+shift*sd,
                                 colors=colors,
                                 points=points,
                                 true_y=true_y,
                                 interval_alpha=interval_alpha,
                                 ax=ax)