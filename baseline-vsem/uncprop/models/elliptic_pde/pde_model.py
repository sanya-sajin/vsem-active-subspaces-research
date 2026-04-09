# experiments/elliptic_pde/pde_model.py
"""
JAX implementation of 1-D diffusion steady-state finite-difference solver.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from uncprop.custom_types import Array


def solve_pde(xgrid: Array,
              left_flux: float,
              k: Array,
              source: Array | float,
              rightbc: float) -> Array:
    """JAX implementation of finite-difference steady diffusion solver.

    Written so it can be jitted and vmapped over the k-field.

    Args:
        xgrid: 1D array of length N (float).
        left_flux: flux at left-hand boundary, k*du/dx = -F
        k: 1D array of length N of diffusivity values.
        source: either an (N,1) array or scalar.
        rightbc: scalar Dirichlet boundary condition, right boundary.

    Returns:
        usolution: 1D array of length N with discretized solution.
    """
    xgrid = jnp.ravel(xgrid)
    k = jnp.ravel(k)
    N = xgrid.shape[0]
    h = xgrid[-1] - xgrid[-2]

    # initialize A and b
    A = jnp.zeros((N - 1, N - 1), dtype=jnp.float64)
    b = jnp.zeros((N - 1, 1), dtype=jnp.float64)

    if isinstance(source, jnp.ndarray) or hasattr(source, "__len__"):
        f = -source[: (N - 1), :]
    else:
        f = -source * jnp.ones((N - 1, 1), dtype=jnp.float64)

    diag_main = -2.0 * k[: (N - 1)] - k[1:] - jnp.concatenate((jnp.array([k[0]]), k[: (N - 2)]))
    A = A + jnp.diag(diag_main)
    if N - 2 > 0:
        A = A + jnp.diag(k[: N - 2], 1) + jnp.diag(k[1: N - 1], 1)
        A = A + jnp.diag(k[: N - 2], -1) + jnp.diag(k[1: N - 1], -1)
    A = A / (2.0 * (h ** 2))

    # Neumann BC at left
    A = A.at[0, 1].set(A[0, 1] + k[0] / (h ** 2))
    b = b.at[0, 0].set(2.0 * left_flux / h)

    # Dirichlet BC at right
    b = b.at[N - 2, 0].set(rightbc * (k[N - 1] + k[N - 2]) / (2.0 * (h ** 2)))

    rhs = (f - b).reshape((N - 1,))
    uinternal = jnp.linalg.solve(A, rhs)
    usolution = jnp.concatenate((uinternal.reshape(N - 1), jnp.array([rightbc])))
    return usolution


# vmap wrapper: batch over k fields (leading axis of k)
solve_pde_vmap = jax.vmap(solve_pde, in_axes=(None, None, 0, None, None))

def get_discrete_source(X: Array, well_locations: Array, strength: float, width: float) -> Array:
    """Source term for the PDE.

    Args:
        X: array of spatial points, shape (n,1) or (n,).
        well_locations: 1D array of source centers (well locations) in (0,1).
        strength: source strength/amplitude.
        width: source width parameter.

    Returns:
        src: array of shape (n,1) containing source at each spatial location.
    """
    X = jnp.reshape(X, (-1, 1))  # (n,1)
    m = jnp.reshape(well_locations, (1, -1))  # (1, M)
    S = (X - well_locations) ** 2
    S = jnp.exp(S / (-2.0 * width ** 2))
    src = strength * jnp.sum(S, axis=1).reshape((-1, 1)) / (width * jnp.sqrt(2.0 * jnp.pi))
    return src