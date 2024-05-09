from typing import Union, Dict, Any

import numpy as np
import tensorflow as tf

from belief_learner.utils.definitions import Array, DictArray


@tf.function
def diff_clip(array, low, high, epsilon):
    clipped_value = tf.clip_by_value(array, low + epsilon, high - epsilon)
    array = array - tf.stop_gradient(array) + tf.stop_gradient(clipped_value)
    return array

def array_to_np(array: Array):
    """
    Transforms an array to a numpy array.

    :param array: A tensorflow / numpy array.
    :return: A numpy array.
    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, tf.Tensor):
        return array.numpy()
    else:
        raise ValueError('Unknown array type.')


def dict_array_to_dict_np(dict_array: Dict[Any, Union[Array, DictArray]]):
    dict_np = {key: dict_array_to_dict_np(value) if isinstance(value, dict) else array_to_np(value)
               for key, value in dict_array.items()}
    return dict_np


def dict_array_to_dict_python(dict_array: Dict[Any, Union[Array, DictArray]]):
    dict_np = {key: dict_array_to_dict_np(value) if isinstance(value, dict) else array_to_np(value).tolist()
               for key, value in dict_array.items()}
    return dict_np

def dict_array_to_dict_tf(dict_array: Dict[Any, Union[Array, DictArray]]):
    dict_tensor = {key: dict_array_to_dict_tf(value) if isinstance(value, dict) else tf.convert_to_tensor(value, name=key)
                   for key, value in dict_array.items()}
    return dict_tensor


def mask_dict_array(dict_array: DictArray, mask: np.ndarray):
    """
    Filters a dict array based on a mask.

    :param dict_array: The dict array to mask/filter.
    :param mask: A boolean numpy array whose length is equal to the first dimension of all the elements in the dict
    array.
    :return: The masked/filter dict array.
    """
    masked_dict = {key: value[mask] for key, value in dict_array.items()}
    return masked_dict

def is_consecutive(array: np.ndarray):
    array = array.flatten()
    if len(array) == 0:
        return True
    assert len(array.shape) == 1
    return np.all(array[:-1] + 1 == array[1:])

def merge_first_dims(array: Array, n: int = 1):
    if isinstance(array, np.ndarray):
        return np.reshape(array, (-1,) + array.shape[n:])
    elif isinstance(array, tf.Tensor):
        return tf.reshape(array, (-1,) + tuple(array.shape[n:].as_list()))
    else:
        raise ValueError()

def unmerge_first_dims(array, seq):
    if isinstance(array, np.ndarray):
        return np.reshape(array, tuple(seq) + array.shape[1:])
    elif isinstance(array, tf.Tensor):
        return tf.reshape(array, tuple(seq) + tuple(array.shape[1:].as_list()))
    else:
        raise ValueError()