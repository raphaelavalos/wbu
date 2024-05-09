import matplotlib.pyplot as plt
from belief_learner.utils.array import array_to_np

import sys
import logging

logger = logging.getLogger(__name__)


def is_debug() -> bool:
    return getattr(sys, 'gettrace', None) is not None


def plot_image_in_line(array):
    array = array_to_np(array)
    assert len(array.shape) == 4
    b = array.shape[0]
    fig, ax = plt.subplots(b, 1, figsize=(15, 15 * b))
    [axi.set_axis_off() for axi in ax.ravel()]
    for i in range(b):
        ax[i].imshow(array[i])
    fig.tight_layout()
    fig.show()


def plot_image_in_pairs(array, next_array):
    array = array_to_np(array)
    next_array = array_to_np(next_array)
    assert len(array.shape) == 4
    assert len(array.shape) == len(next_array.shape)
    b = array.shape[0]
    fig, ax = plt.subplots(b, 2, figsize=(15 * 2, 15 * b))
    [axi.set_axis_off() for axi in ax.ravel()]
    for i in range(b):
        ax[i, 0].imshow(array[i])
        ax[i, 1].imshow(next_array[i])
    fig.tight_layout()
    fig.show()
