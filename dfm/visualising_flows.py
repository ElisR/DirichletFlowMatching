"""Functions for visualising the Dirichlet noiser for discrete probability distributions."""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats.dirichlet import pdf as dirichlet_pdf
from jaxtyping import Array, Float, Int, Key

SIMPLEX_TOLERANCE = 1e-6


def is_simplex(x: Array) -> Array:
    """Return the indices which of points which are determined to be on the simplex."""
    x_sum = jnp.sum(x, axis=0)
    return jnp.all(x > 0, axis=0) & (abs(x_sum - 1) < SIMPLEX_TOLERANCE)


def dirichlet_density(
    x_1: Int[Array, "1"],
    t: Float,
    num_cats: int,
    *,
    bins: int = 100,
) -> tuple[
    tuple[Float[Array, "L"], Float[Array, "L"], Float[Array, "L"], Float[Array, "L"]],
    Float[Array, "{num_cats} L"],
]:
    r"""Calculate the Dirichlet distribution after a certain amount of time.

    Will include coordinates that are not on the simplex.
    Note that this strategy of keeping too many points is inefficient compared to a principled
    traversal of the simplex in a principled way.

    Args:
        x_1: The observation.
        t: The time elapsed.
        num_cats: The number of categories.
        key: The random key.
        bins: The number of bins for the histogram. (Too many will crash matplotlib.)

    Returns:
        A tuple of (t, l, r, v) where t, l, r are the ternary coordinates and v are values.
    """
    # TODO Swap for the target class
    alpha = jnp.ones((num_cats,)) + t * jax.nn.one_hot(x_1, num_classes=num_cats).squeeze()
    values = jnp.linspace(0.0, 1.0, num=bins)
    tlr = jnp.meshgrid(values, values, values)
    tlr = jax.tree_map(jnp.ravel, tlr)
    coords = jnp.stack(tlr, axis=0)
    probs = dirichlet_pdf(coords, alpha=alpha).ravel()
    return (*tlr, probs), coords


def sample_x(
    x_1: Int[Array, "D"],
    num_cats: int,
    t: float,
    delta_t: float,
    *,
    key: Key,
) -> Float[Array, "steps D {num_cats}"]:
    r"""Produce a stochastic trajectory of noised $\mathbf{x}$ values.

    Args:
        x_1: The observation.
        num_cats: The number of categories.
        t: The time to integrate to.
        delta_t: The time step between each sampling.
        key: The random key.

    Returns:
        A trajectory of categorical distribution parameters.
    """
    alpha_zero = jnp.ones((num_cats,), dtype=jnp.float32)
    oh_x_1 = jax.nn.one_hot(x_1, num_cats, axis=-1)

    def time_step(key: Key, t: Int):  # noqa: ANN202
        alpha = alpha_zero + t * oh_x_1
        key, x_key = jr.split(key, 2)
        x = jr.dirichlet(x_key, alpha, shape=oh_x_1.shape[:-1])
        return key, x

    ts = jnp.arange(0.0, t, step=delta_t)
    _, x_timeline = jax.lax.scan(time_step, key, ts)
    return x_timeline
