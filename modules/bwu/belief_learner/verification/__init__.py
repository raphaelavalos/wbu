import tensorflow as tf
from tf_agents.typing.types import Int


def binary_latent_space(latent_state_size: Int, dtype=tf.int32):
    return tf.cast(
        tf.math.mod(
            tf.bitwise.right_shift(
                tf.expand_dims(tf.range(tf.pow(2, latent_state_size)), 1),
                tf.range(latent_state_size)),
            2),
        dtype)
