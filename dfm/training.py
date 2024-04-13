"""Module for training the model."""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Int, Key, PyTree

import dfm.loss_and_sample as las


@partial(jax.jit, static_argnums=(0, 2))
def make_step(  # noqa: PLR0913
    model: nn.Module,
    x_batch: Int[Array, "N *shape"],
    optim,
    opt_state,
    params: PyTree,
    t_infty: float,
    *,
    key: Key,
):
    """Calculate loss & grad for a batch and update model according to optimiser.

    Args:
        model: The DFM model to be trained.
        x_batch: The input data, a JAX array of integers of shape (N, D).
        optim: Optax optimiser.
        opt_state: Optax optimiser state.
        params: The parameters of the model.
        key: The random key to be used during loss calculation.

    Returns:
        The loss, updated parameters, and updated optimiser state.
    """
    batch_size = x_batch.shape[0]

    def loss_for_batch(params: PyTree, key: Key) -> Float:
        keys = jr.split(key, batch_size)
        loss = jnp.mean(jax.vmap(las.loss, in_axes=(None, None, 0, None))(params, model, x_batch, t_infty, key=keys))
        return loss

    loss, grads = jax.value_and_grad(loss_for_batch)(params, key)
    updates, opt_state = optim.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return loss, params, opt_state
