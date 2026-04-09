'''
vsem_jax.py
JAX implementation of the Very Simple Ecosystem Model (VSEM).

See BayesianTools R package:
    https://search.r-project.org/CRAN/refmans/BayesianTools/html/VSEM.html
'''

from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Any
from collections.abc import Sequence, Callable, Mapping

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util, lax
import matplotlib.pyplot as plt

from uncprop.custom_types import Array, PRNGKey


# ---------------------------------------------------------------------
# Canonical names - defines canonical ordering
# ---------------------------------------------------------------------

canonical_par_names = [
    'kext',      # KEXT - light extinction coefficient
    'lar',       # LAR  - leaf area ratio
    'lue',       # LUE  - light use efficiency
    'gamma',     # GAMMA - autotrophic respiration fraction
    'tauv',      # tauV - veg longevity
    'taus',      # tauS - soil longevity
    'taur',      # tauR - root longevity
    'av',        # Av - fraction of NPP to veg
    'veg_init',  # Cv initial
    'soil_init', # Cs initial
    'root_init'  # Cr initial
]

canonical_output_names = ['veg', 'soil', 'root', 'nee', 'lai']


# ---------------------------------------------------------------------
# Pytree dataclasses for VSEM inputs
# ---------------------------------------------------------------------

@tree_util.register_pytree_node_class
@dataclass
class VSEMParam:
    """Container for VSEM parameters. Registered as a JAX pytree."""
    kext: float | Array
    lar: float | Array
    lue: float | Array
    gamma: float | Array
    tauv: float | Array
    taus: float | Array
    taur: float | Array
    av: float | Array
    veg_init: float | Array
    soil_init: float | Array
    root_init: float | Array

    def tree_flatten(self):
        children = (self.kext, self.lar, self.lue, self.gamma, self.tauv,
                    self.taus, self.taur, self.av, self.veg_init,
                    self.soil_init, self.root_init)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class VSEMInitialCondition:
    """Initial conditions container (veg_init, soil_init, root_init).
    
    TODO: at present the API for running this model does not differentiate
    between trait parameters and initial conditions - they are all considered
    "parameters". This class doesn't have much use beyond the conceptual.
    Eventually, should update so that the interface is defined in terms of 
    VSEMInput, a pytree with subtrees for driver/params/IC. For batch inputs,
    should maintain the pytree structure rather than flattening.
    """
    veg_init: float | Array
    soil_init: float | Array
    root_init: float | Array

    def tree_flatten(self):
        children = (self.veg_init, self.soil_init, self.root_init)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class VSEMInput:
    """Top-level pytree container for inputs required by the solver."""
    param: VSEMParam
    initial_condition: VSEMInitialCondition
    driver: Array

    def tree_flatten(self):
        children = (self.param, self.initial_condition, self.driver)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ---------------------------------------------------------------------
# Small helpers - models for leaf area index / net ecosystem exchange
# ---------------------------------------------------------------------

def compute_lai(xv: Array, lar: float | Array) -> Array:
    """LAI = LAR * xv."""
    return lar * xv


def compute_nee(xv: Array,
                driver: Array,
                gamma: float | Array,
                lue: float | Array,
                kext: float | Array,
                lar: float | Array) -> Array:
    """Compute NEE(t) = (1-gamma) * lue * PAR(t) * (1 - exp(-kext * LAI))."""
    lai = compute_lai(xv, lar)
    return (1.0 - gamma) * lue * driver * (1.0 - jnp.exp(-kext * lai))


# ---------------------------------------------------------------------
# ODE and solver (compatible with JAX transforms)
# ---------------------------------------------------------------------

def vsem_rhs(x: Array, par: VSEMParam, driver_t: float | Array) -> Array:
    """RHS of VSEM ODE at a single time step."""
    xv, xs, xr = x[0], x[1], x[2]

    # Unpack parameters
    k = par.kext
    lar = par.lar
    lue = par.lue
    gamma = par.gamma
    tauv = par.tauv
    taus = par.taus
    taur = par.taur
    av = par.av

    npp = compute_nee(xv, driver_t, gamma, lue, k, lar)

    dxv_dt = av * npp - xv / tauv
    dxs_dt = xr / taur + xv / tauv - xs / taus
    dxr_dt = (1.0 - av) * npp - xr / taur

    return jnp.stack([dxv_dt, dxs_dt, dxr_dt])


def _vsem_step(x_prev: Array,
               driver_t: float | Array,
               par: VSEMParam) -> tuple[Array, Array]:
    """Single forward-Euler step for use with lax.scan."""
    driver_t = jnp.squeeze(driver_t)
    dx = vsem_rhs(x_prev, par, driver_t)
    x_next = x_prev + dx  # forward Euler, dt=1 (daily time step)

    veg = x_next[0]
    soil = x_next[1]
    root = x_next[2]
    nee = compute_nee(xv=veg, driver=driver_t, gamma=par.gamma, lue=par.lue,
                      kext=par.kext, lar=par.lar)
    lai = compute_lai(xv=veg, lar=par.lar)

    output_row = jnp.stack([veg, soil, root, nee, lai])
    return x_next, output_row


@jax.jit
def solve_vsem(vsem_input: VSEMInput) -> Array:
    """Solve VSEM with forward Euler

    Returns:
        model_out: (n_timesteps, 5) with columns [veg, soil, root, nee, lai]
        Note that the output does not include the initial condition (time step 0)
    """
    driver = vsem_input.driver.ravel()
    param = vsem_input.param

    x0 = jnp.array([vsem_input.initial_condition.veg_init,
                    vsem_input.initial_condition.soil_init,
                    vsem_input.initial_condition.root_init])

    def _step(x_prev, driver_t):
        x_next, out_row = _vsem_step(x_prev, driver_t, param)
        return x_next, out_row

    # shape (n_timesteps, 5) [excludes initial condition]
    _, outputs = lax.scan(_step, x0, driver)
    return outputs


# ---------------------------------------------------------------------
# Defaults and conversions between named dicts and canonical vectors
# ---------------------------------------------------------------------

def get_vsem_default_pars_dict() -> dict[str, float]:
    """Return canonical default parameter values as a dict keyed by canonical_par_names."""
    return {
        'kext': 0.5,
        'lar': 1.5,
        'lue': 0.002,
        'gamma': 0.4,
        'tauv': 1440.0,
        'taus': 27370.0,
        'taur': 1440.0,
        'av': 0.5,
        'veg_init': 3.0,
        'soil_init': 15.0,
        'root_init': 3.0
    }


def canonical_defaults_array(par_default: Mapping[str, float] | None = None) -> Array:
    """Return canonical default parameter vector in canonical order (jnp.array)."""
    d = par_default or get_vsem_default_pars_dict()
    return jnp.asarray([d[name] for name in canonical_par_names])


def par_vector_from_named_dict(named: Mapping[str, Any],
                               par_default: Mapping[str, float] | None = None) -> VSEMParam:
    """Build a VSEMParam pytree from a Python mapping name->value (user-facing)."""
    defaults = par_default or get_vsem_default_pars_dict()
    vals = []
    for name in canonical_par_names:
        if name in named:
            vals.append(jnp.asarray(named[name]))
        else:
            vals.append(jnp.asarray(defaults[name]))
    return VSEMParam(*vals)


def make_vsem_input_from_named(par_named: Mapping[str, Any],
                               driver: Sequence[float] | Array,
                               par_default: Mapping[str, float] | None = None) -> VSEMInput:
    """Convenience: construct a VSEMInput from named parameter mapping"""
    param = par_vector_from_named_dict(par_named, par_default)
    initial_condition = VSEMInitialCondition(veg_init=param.veg_init,
                                             soil_init=param.soil_init,
                                             root_init=param.root_init)

    # ensure driver is flat array
    driver_jnp = jnp.asarray(driver).ravel()

    return VSEMInput(param=param, initial_condition=initial_condition, driver=driver_jnp)


def vector_to_vsemparam(full_vector: Array) -> VSEMParam:
    """Convert a full-length canonical-order vector to a VSEMParam pytree.

    This function is JAX-traceable. full_vector must have shape 
    (len(canonical_par_names),).
    """
    return VSEMParam(*tuple(full_vector[i] for i in range(len(canonical_par_names))))


# ---------------------------------------------------------------------
# Forward model builders:
#   Vectorized VSEM runs as function of varying input parameters
# ---------------------------------------------------------------------

def build_batch_forward_model(driver: Array,
                              par_names: list[str],
                              default_param_dict: Mapping[str, float] | None = None):
    """
    Returns forward_batch(batch_of_partial_params) -> batched outputs.
    - driver is treated as fixed (captured in closure) and should be a jnp array.
    - par_names are validated against canonical_par_names.
    - default_param_dict is converted to canonical defaults array.
    """

    # ensure driver is flat array
    driver = jnp.asarray(driver).ravel()

    # Validate parameter names
    for name in par_names:
        if name not in canonical_par_names:
            raise ValueError(f"Unknown parameter name: {name}")

    # indices as a small python tuple (static) OR as a jnp array precomputed:
    indices = tuple(canonical_par_names.index(name) for name in par_names)
    indices_arr = jnp.asarray(indices, dtype=jnp.int32)   # used in-array ops

    # default_param as a DeviceArray (captured in closure)
    default_param = canonical_defaults_array(default_param_dict)

    def _single_run(partial_param: Array):
        param_vec = default_param.at[indices_arr].set(partial_param)
        param = vector_to_vsemparam(param_vec)

        initial_condition = VSEMInitialCondition(veg_init=param.veg_init,
                                                 soil_init=param.soil_init,
                                                 root_init=param.root_init)
        vsem_input = VSEMInput(param=param,
                               initial_condition=initial_condition,
                               driver=driver)
        return solve_vsem(vsem_input)

    # vectorize over the first axis of a batch of partial params, then jit the batched function
    forward_batch = jax.jit(jax.vmap(_single_run, in_axes=0, out_axes=0))

    return forward_batch


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def get_vsem_par_names() -> list[str]:
    """Return canonical parameter names."""
    return list(canonical_par_names)


def get_vsem_output_names() -> list[str]:
    """Return canonical output names."""
    return list(canonical_output_names)


def get_default_prior_bounds() -> dict[str, tuple[float, float]]:
    """Uniform prior bounds for canonical parameters."""
    return {
        "kext": (0.2, 1.0),
        "lar": (0.2, 3.0),
        "lue": (5e-4, 4e-3),
        "gamma": (0.2, 0.6),
        "tauv": (5e2, 3e3),
        "taus": (4e3, 5e4),
        "taur": (5e2, 3e3),
        "av": (2e-1, 1.0),
        "veg_init": (0.0, 10.0),
        "soil_init": (0.0, 30.0),
        "root_init": (0.0, 10.0)
    }


def simulate_vsem_driver(key: PRNGKey, n_days: int) -> tuple[Array, Array]:
    """Generate a synthetic PAR time series (time_steps, driver)."""
    time_steps = jnp.arange(n_days)
    driver = 10 * jnp.abs(jnp.sin(time_steps / 365 * jnp.pi) + 0.25 * jr.normal(key, (n_days,)))
    return time_steps, driver


def plot_vsem_outputs(output: Array,
                      output_names: Sequence[str] | None = None,
                      nrows: int = 1,
                      ncols: int | None = None,
                      figsize: tuple[float, float] = (5, 4),
                      plot_kwargs: dict | None = None):
    """Plot model outputs. `output` expected (n_run, n_time, n_output) or (n_time, n_output)."""
    if output.ndim == 2:
        output = output[jnp.newaxis, ...]

    if output_names is None:
        output_names = get_vsem_output_names()
    if plot_kwargs is None:
        plot_kwargs = {}

    n_plots = len(output_names)
    if ncols is None:
        ncols = int(np.ceil(n_plots / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).flatten()
    time_steps = np.arange(output.shape[1])
    all_output_names = get_vsem_output_names()

    for j in range(n_plots):
        output_name = output_names[j]
        output_idx = all_output_names.index(output_name)
        ax = axs[j]
        ax.plot(time_steps, output[:, :, output_idx].T, **plot_kwargs)
        ax.set_title(output_name)
        ax.set_xlabel('days')
        ax.set_ylabel(output_name)

    for k in range(n_plots, nrows*ncols):
        fig.delaxes(axs[k])
    plt.close(fig)
    return fig, axs