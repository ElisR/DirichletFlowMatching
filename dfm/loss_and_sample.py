"""Module containing loss functions and sampling methods for Dirichlet Flow Matching."""
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import betainc
from jaxtyping import Array, Float, Int, Key, PyTree
from optax import softmax_cross_entropy

GRAD_ESTIMATOR = 1e-3


def alpha_t(
    oh_x_infty: Float[Array, "*shape num_cats"],
    t: float,
    *,
    num_cats: int,
) -> Int[Array, "*shape num_cats"]:
    """Get the parameters of the Dirichlet distribution at a given time for given data.

    Args:
        x_infty: Input array of one-hot encoded data.
        t: The time parameter.
        num_cats: The number of categories i.e. vertices on the simplex.

    Returns:
        The biased alpha parameters for all the input data.
    """
    return jnp.ones((num_cats,)) + t * oh_x_infty


def loss(params: PyTree, model: nn.Module, x_infty: Int[Array, "*shape"], t_infty: float, *, key: Key) -> float:
    """Return the Dirichlet Flow Matching cross-entropy loss for a single sample.

    Args:
        params: Parameters of the neural network.
        model: Neural network that transforms parameters of categorical distribution.
        x_infty: Input array of integers, where integers are in the range [0, K).
        t_infty: The maximum time that can be sampled.
        key: The random key to be used for sampling.

    Returns:
        The cross-entropy loss.
    """
    # TODO Make this a keyword argument
    num_cats = model.num_cats

    x_key, t_key = jr.split(key)
    t = jr.uniform(t_key) * t_infty

    oh_x_infty = jax.nn.one_hot(x_infty, num_classes=num_cats, axis=-1)
    alpha = alpha_t(oh_x_infty, t, num_cats=num_cats)
    x = jr.dirichlet(x_key, alpha, shape=oh_x_infty.shape[:-1])

    logits = model.apply({"params": params}, x, t)
    return softmax_cross_entropy(logits, oh_x_infty)


def conditional_flows(
    x: Float[Array, "*shape num_cats"],
    t: float,
) -> Float[Array, "*shape num_cats num_cats"]:
    """Calculate the conditional vector field at a given `x` and `t`.

    Args:
        x: The position on the simplex at which to calculate the vector field.
        t: The time parameter.

    Returns:
        The conditional vector field `u_t(x | x_infty)`.
    """
    num_cats = x.shape[-1]
    ones = jnp.ones((*x.shape, num_cats))
    x_all = jnp.expand_dims(x, axis=-1) * ones

    # Calculate the magnitude
    # NOTE Autodiff not allowed for first argument of regularised incomplete beta function
    # NOTE beta_inc_diff = jax.grad(betainc, argnums=0)(t + 1, num_cats - 1, x_all)
    beta_inc_diff = (
        betainc(t + 1 + GRAD_ESTIMATOR / 2, num_cats - 1, x_all)
        - betainc(t + 1 - GRAD_ESTIMATOR / 2, num_cats - 1, x_all)
    ) / GRAD_ESTIMATOR
    beta = jax.scipy.special.beta(t + 1, num_cats - 1)
    c = -beta_inc_diff * beta / (jax.lax.integer_pow(1 - x_all, num_cats - 1) * jnp.power(x_all, t))

    return c * (ones - x_all)


@partial(jax.jit, static_argnums=(1, 3), static_argnames=("shape",))
def sample(
    params: PyTree,
    model: nn.Module,
    t_infty: float,
    steps: int,
    *,
    shape: tuple[int],
    key: Key,
) -> Float[Array, "*shape"]:
    """Sample by integrating .

    Args:
        params: Parameters of the neural network.
        model: Neural network that transforms parameters of categorical distribution.
        t_infty: The final value of beta at t = 1.
        steps: The number of sampling steps.
        shape: The shape of the data (before one-hot encoding).
        key: The random key to be used for sampling.

    Returns:
        The sampled data.
    """
    num_cats = model.num_cats
    dt = t_infty / steps

    alpha = jnp.ones((num_cats,))
    x_0 = jr.dirichlet(key, alpha, shape=shape)

    def time_step(x: Float[Array, "*shape num_cats"], t: Int):  # noqa: ANN202
        logits = model.apply({"params": params}, x, t)
        probs = nn.softmax(logits, axis=-1)

        u_all = conditional_flows(x, t)
        v = jnp.sum(u_all * jnp.expand_dims(probs, axis=-2), axis=-1)

        return x + v * dt, x

    ts = jnp.linspace(0.0, t_infty, num=steps + 1)
    x_infty, x_timeline = jax.lax.scan(time_step, x_0, ts)
    x_infty = jnp.argmax(x_infty, axis=-1)

    return x_infty, x_timeline
