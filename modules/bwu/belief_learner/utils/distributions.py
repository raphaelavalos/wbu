from abc import ABC

import numpy as np
import tensorflow as tf

from belief_learner.utils.definitions import Array


def categorical(p: np.ndarray) -> np.ndarray:
    """
    Categorical sampling over the last dimension.
    :param p: Probability weights (should sum to 1.)
    :return: Category / index sampled (shape is the same as p without the last dimension p.shape[:-1])
    """
    assert np.all(np.isclose(p.sum(-1), 1.)), f"p should sum to 1 over the last dimension. p: {p}"
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def softmax(array: Array) -> Array:
    """
    Applies softmax to np.ndarray and tf.Tensors to the last dimension.
    :param array: The input array.
    :return: The output array.
    """
    if isinstance(array, np.ndarray):
        array = array.copy()
        array -= array.max(-1, keepdims=True)
        array = np.exp(array)
        array /= array.sum(-1, keepdims=True)
    elif isinstance(array, tf.Tensor):
        array = tf.math.softmax(array, -1)
    return array


class Distribution(ABC):
    """Distribution abstract class. Can take as input numpy arrays and tf tensors."""

    def __init__(self, input_: Array):
        assert isinstance(input_, (np.ndarray, tf.Tensor))
        self._input = input_
        self._probs = softmax(input_)
        # if isinstance(input_, np.ndarray):
        #     # self._probs = input_ if np.all(np.sum(input_, -1) == 1.) else softmax(input_)
        #     self._probs = softmax(input_)
        # else:
        #     # self._probs = input_ if tf.reduce_all(tf.reduce_sum(input_, -1) == 1.) else softmax(input_)
        #     self._probs = softmax(input_)

    def sample(self):
        if isinstance(self._input, np.ndarray):
            return self._np_sample(False)
        elif isinstance(self._input, tf.Tensor):
            return self._tf_sample(False)
        else:
            raise ValueError("Unknown type.")

    def deterministic_sample(self):
        if isinstance(self._input, np.ndarray):
            return self._np_sample(True)
        elif isinstance(self._input, tf.Tensor):
            return self._tf_sample(True)
        else:
            raise ValueError("Unknown type.")

    def _np_sample(self, deterministic: bool):
        raise NotImplementedError()

    def _tf_sample(self, deterministic: bool):
        raise NotImplementedError()


class DeterministicDistribution(Distribution):
    def _np_sample(self, deterministic: bool):
        return np.argmax(self._probs, -1)

    def _tf_sample(self, deterministic: bool):
        return tf.argmax(self._probs, -1)


class StochasticDistribution(DeterministicDistribution):
    def _np_sample(self, deterministic: bool):
        if deterministic:
            return super(StochasticDistribution, self)._np_sample(deterministic)
        return categorical(self._probs)

    def _tf_sample(self, deterministic: bool):
        if deterministic:
            return super(StochasticDistribution, self)._tf_sample(deterministic)
        return tf.squeeze(tf.random.categorical(self._probs, 1, dtype=tf.int32), -1)
