# Adapted from https://stackoverflow.com/a/66406793
from typing import Union

import numpy as np


class NPdequeue:
    def __init__(self, dtype: np.dtype, capacity: int):
        #allocate the memory we need ahead of time
        self._capacity: int = capacity
        self._dtype = dtype
        self._array = np.empty(capacity, dtype=dtype)
        self._tail = 0
        self._full = False

    def empty(self):
        self._tail = 0
        self._full = False

    def __len__(self):
        if self._full:
            return self._capacity
        return self._tail

    def to_array(self) -> np.array:
        if self._full:
            return np.roll(self._array, -self._tail)  # this will force a copy
        return self._array[:self._tail]

    def enqueue(self, item: object) -> None:
        self._array[self._tail] = item
        self._increase(1)

    def _increase(self, n: int):
        self._tail += n
        if self._tail >= self._capacity:
            self._full = True
            self._tail %= self._capacity

    def enqueue_array(self, array: np.ndarray) -> None:
        if len(array) + self._tail > self._capacity:
            i = self._capacity - self._tail
            self.enqueue_array(array[:i])
            self.enqueue_array(array[i:])
        else:
            self._array[self._tail: self._tail + len(array)] = array
            self._increase(len(array))

    def get_n_elements(self, n: int) -> np.ndarray:
        if n > len(self):
            raise ValueError("n is too big")
        i = self._tail - n
        i_ = i % self._capacity
        if i >= 0:
            return self._array[i: self._tail]
        else:
            assert self._full
            return np.concatenate([self._array[i_:], self._array[:self._tail]], 0)

    def get_idx_in_n_elem(self, n: int, idxs: Union[list, np.ndarray]):
        i = self._tail - n
        i_ = i % self._capacity
        if i >= 0:
            return self._array[i: self._tail][idxs].copy()
        else:
            assert self._full
            items = np.empty(len(idxs), dtype=self._dtype)
            items[idxs < -i] = self._array[i_:][idxs[idxs < -i]]
            items[idxs >= -i] = self._array[:self._tail][idxs[idxs >= -i] + i]
            assert np.all(items == self.get_n_elements(n)[idxs])
        return items

    def as_data(self):
        data = {
            "array": self._array.copy(),
            "tail": self._tail,
            "full": self._full,
        }
        return data

    def load_data(self, data):
        self._array = data["array"]
        self._tail = data["tail"]
        self._full = data["full"]
