"""Module containing model for transforming between categorical distributions."""
from typing import Self

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


class InnerNetwork(nn.Module):
    """Neural network that acts on categorical distribution."""

    num_cats: int  # Number of categories

    # TODO Consider turning time argument into a JAX array with a single float
    @nn.compact
    def __call__(self: Self, x: Float[Array, "*shape num_cats"], t: Float) -> Float[Array, "*shape num_cats"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        d = x.shape[-2]
        residual = x

        x = nn.Dense(features=self.num_cats, name="category_mixer")(x)  # TODO Be careful with bigger shape
        x = nn.gelu(x)
        x = nn.Dense(features=d, name="position_mixer")(x.T)
        x = nn.gelu(x.T)
        return x + residual, None


class MultipleMLP(nn.Module):
    """Neural network that applies many layers of the inner network."""

    num_cats: int  # Number of categories
    scale: int = 2  # Scale of the inner network

    @nn.compact
    def __call__(self: Self, x: Float[Array, "*shape num_cats"], t: Float) -> Float[Array, "*shape num_cats"]:
        """Return the output distribution of the model, given an underyling NN architecture."""
        scanned_inner = nn.scan(
            InnerNetwork,
            variable_axes={"params": 0},
            variable_broadcast=False,
            in_axes=(nn.broadcast,),
            split_rngs={"params": True},
            length=10,
        )
        x = nn.Dense(features=self.scale * self.num_cats)(x)
        x, _ = scanned_inner(self.scale * self.num_cats)(x, t)
        x = nn.Dense(features=self.num_cats)(x)
        return x
