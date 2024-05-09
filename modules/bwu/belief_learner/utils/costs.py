from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf

from tf_agents.typing.types import Float


def norm2(x: Float, axis: Union[int, List[int], Tuple[int, ...]] = -1, epsilon=1e-12):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + epsilon)


def l2(x: Float, y: Float):
    return norm2(x - y, axis=tf.range(1, tf.rank(x)))


def l22(x: Float, y: Float):
    return tf.reduce_sum(tf.square(x - y), axis=tf.range(1, tf.rank(x)))


def cosine_similarity(x: Float, y: Float):
    # assert tf.reduce_all(tf.shape(x) == tf.shape(y))
    b = tf.shape(x)[0]
    x = tf.reshape(x, (b, -1))
    y = tf.reshape(y, (b, -1))
    return tf.reduce_sum(x * y, axis=1) / norm2(x) / norm2(y)


def m_cosine_similarity(x: Float, y: Float):
    x /= norm2(x,)[:, None]
    y /= norm2(y,)[:, None]
    return x @ tf.transpose(y)


def cosine_distance(x: Float, y: Float):
    return 1 - cosine_similarity(x, y)


def angular_distance(x: Float, y: Float, epsilon=1e-12):
    return tf.math.acos(cosine_similarity(x, y) + epsilon) / np.pi


def binary_cross_entropy(x: Float, y: Float):
    # tricky: we assume that the original vector is x, and the reconstructed is y.
    output = tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=x)
    return tf.reduce_sum(output, axis=tf.range(1, tf.rank(x)))


_registry = {
    "norm2": norm2,
    "l2": l2,
    "l22": l22,
    "cosine_similarity": cosine_similarity,
    "cosine_distance": cosine_distance,
    "angular_distance": angular_distance,
    "binary_cross_entropy": binary_cross_entropy,
}


def get_cost_fn(cost_fn_name: str):
    if cost_fn_name not in _registry:
        raise ValueError(f"Cost function {cost_fn_name} unknown.")
    return _registry[cost_fn_name]
