"""Module holding some neural networks for 2D images."""

from typing import Self

import einops
import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array, Float


class InputMLP(nn.Module):
    """Simple MLP to change from float to categorical logits."""

    hidden_dim: int  # Dimension after embedding

    @nn.compact
    def __call__(self: Self, y: Float[Array, "h w num_cats"]) -> Float[Array, "h w 1"]:
        """Apply an MLP with a single hidden layer."""
        num_cats = y.shape[-1]
        y = nn.Dense(features=num_cats, use_bias=True)(y)
        y = nn.relu(y)
        y = nn.Dense(features=self.hidden_dim, use_bias=True)(y)
        return y


class OutputMLP(nn.Module):
    """Simple MLP to change from float to categorical logits."""

    num_cats: int  # Number of categories for output

    @nn.compact
    def __call__(self: Self, y: Float[Array, "h w 1"]) -> Float[Array, "h w num_cats"]:
        """Apply an MLP with a single hidden layer."""
        y = nn.Dense(features=self.num_cats, use_bias=True)(y)
        y = nn.relu(y)
        y = nn.Dense(features=self.num_cats, use_bias=True)(y)
        return y


class MLP(nn.Module):
    """Basic MLP architecture."""

    hidden_dim: int

    @nn.compact
    def __call__(self: Self, y: Float[Array, "c d"]) -> Float[Array, "c d"]:
        """Apply an MLP with a single hidden layer."""
        in_dim = y.shape[-1]
        y = nn.Dense(features=self.hidden_dim, use_bias=True)(y)
        y = nn.relu(y)
        y = nn.Dense(features=in_dim, use_bias=True)(y)
        return y


class MixerBlock(nn.Module):
    """Basic block of the MLP-Mixer architecture."""

    mix_patch_size: int  # Size of the patch mixing MLP
    mix_hidden_size: int  # Size of channel mixing MLP

    @nn.compact
    def __call__(self: Self, y: Float[Array, "c d"]) -> Float[Array, "c d"]:
        """Apply mixer block to input, with image as one size."""
        y = y + MLP(self.mix_patch_size)(nn.LayerNorm()(y))
        y = y.T
        y = y + MLP(self.mix_hidden_size)(nn.LayerNorm()(y))
        y = y.T
        return y, None


class Mixer2D(nn.Module):
    """Basic MLP-Mixer architecture.

    Reusing architecture from another diffusion demonstration,
    so converting one-hot-encoded variable to single float channel.
    """

    num_cats: int  # Number of categories for discrete variable
    num_blocks: int  # Number of mixer blocks
    patch_size: int  # Size of the patches
    hidden_size: int  # Size of the hidden layers during convolution
    mix_patch_size: int  # Size of the patch mixing MLP
    mix_hidden_size: int  # Size of channel mixing MLP

    @nn.compact
    def __call__(self: Self, y: Float[Array, "h w num_cats"], t: Float) -> Float[Array, "h w num_cats"]:
        """Apply MLP-Mixer to input."""
        height, width, num_cats = y.shape
        assert height % self.patch_size == 0
        assert width % self.patch_size == 0

        # Collapse one-hot encoding into single float channel
        y = InputMLP(hidden_dim=5)(y)

        # Stack time as a channel
        t = einops.repeat(t, "-> h w 1", h=height, w=width)
        y = jnp.concatenate([y, t], axis=-1)

        # Apply convolutional layer
        y = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )(y)
        patch_height, patch_width, _ = y.shape

        # Apply mixer blocks sequentially
        y = einops.rearrange(y, "h w c -> c (h w)")
        y, _ = nn.scan(MixerBlock, variable_axes={"params": 0}, split_rngs={"params": True}, length=self.num_blocks)(
            self.mix_patch_size,
            self.mix_hidden_size,
        )(y)

        # Rearrange and apply final convolutional layer
        y = nn.LayerNorm()(y)
        y = einops.rearrange(y, "c (h w) -> h w c", h=patch_height, w=patch_width)
        y = nn.ConvTranspose(
            features=1,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
        )(y)

        # Turn single float channel into logits
        y = OutputMLP(num_cats=num_cats)(y)
        return y
