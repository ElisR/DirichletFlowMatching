"""Module containing some useful functions for generating example data for training."""

from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    """Dataset of MNIST images, with pixels existing in {0, 1, ..., `num_cats` - 1} based on intensity.

    Usually choose three categories so that flow is easier to visualise on the triangle simplex later.
    """

    def __init__(self: Self, num_cats: int = 3) -> None:
        """Initialise the dataset."""

        def transform(pic: np.array) -> Int[Array, "28 28"]:
            """Function for transforming PIL image to JAX array."""
            normalised = jnp.array(pic, dtype=jnp.float32) / 255.0
            bins = jnp.linspace(0.0, 1.0, num_cats + 1)
            image = jnp.digitize(normalised, bins) - 1
            # Clamp since floating point error can sometimes take it out of range
            image = jax.lax.clamp(0, image, num_cats - 1)
            return image

        self.mnist = MNIST(root="./data", train=False, download=True, transform=transform)
        self.num_cats = num_cats

    def __len__(self: Self) -> int:
        """Return the length of the dataset."""
        return len(self.mnist)

    def __getitem__(self: Self, idx: int) -> Int[Array, "28 28"]:
        """Return the image at the given index."""
        return self.mnist[idx][0]
