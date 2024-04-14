"""Testing some basic properties about the dataset."""

import jax.numpy as jnp

from dfm.mnist_data import MNISTDataset


def test_mnist_properties():
    dataset = MNISTDataset(train=False)
    image = dataset[0]

    assert image.shape == (28, 28)

    assert jnp.all(image >= 0)
    assert jnp.all(image < dataset.num_cats)
