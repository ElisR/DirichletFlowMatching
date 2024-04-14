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

    @staticmethod
    def _transform(pic: Int[np.ndarray, "28 28"], *, num_cats: int) -> Int[Array, "28 28"]:
        """Function for transforming PIL image to JAX array."""
        normalised = jnp.array(pic, dtype=jnp.float32) / 255.0
        bins = jnp.linspace(0.0, 1.0, num_cats + 1)
        image = jnp.digitize(normalised, bins) - 1
        # Clamp since floating point error can sometimes take it out of range
        image = jax.lax.clamp(0, image, num_cats - 1)
        return image

    def __init__(self: Self, *, train: bool, num_cats: int = 3, digits: set[int] | None = None) -> None:
        """Initialise the dataset."""
        self.mnist = MNIST(root="./data", train=train, download=True, transform=None)

        # Limit to certain digits if specified
        # Will increase preprocessing time since MNIST internals have to read image
        if digits is not None:
            self.mnist = [(pic, label) for pic, label in self.mnist if label in digits]

        self.num_cats = num_cats

    def __len__(self: Self) -> int:
        """Return the length of the dataset."""
        return len(self.mnist)

    def __getitem__(self: Self, idx: int) -> Int[Array, "28 28"]:
        """Return the image at the given index."""
        pic, _ = self.mnist[idx]
        pic = self._transform(pic, num_cats=self.num_cats)
        return pic
