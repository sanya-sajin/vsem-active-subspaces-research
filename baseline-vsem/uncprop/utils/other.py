# uncprop/utils/other.py

import jax.random as jr
from uncprop.custom_types import PRNGKey


def _numpy_rng_seed_from_jax_key(key: PRNGKey) -> int:
    return int(jr.randint(key, (), 0, 2**63 - 1))