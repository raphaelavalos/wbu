import copy
import os.path
from typing import Dict, Union, Optional, Tuple, List

import gym
import numpy as np
import tensorflow as tf
from gym import spaces
from gym.core import ObsType

from belief_learner.utils.definitions import SUB_BELIEF, ACTION, OBS, STATE, PREV_ACTION, NEXT_OBS, PREV_SUB_BELIEF, \
    EPISODE_NBR, NEXT_IS_RESET_STATE, \
    TIMESTEP, NEXT_STATE, DONE, DictArray, Array, REWARD, inv_next_keys, IS_RESET_STATE, RAW_INDEXES
from belief_learner.utils import array_to_np, dict_array_to_dict_np, mask_dict_array, is_consecutive, \
    dict_array_to_dict_tf
from belief_learner.utils.np_queue import NPdequeue


from belief_learner.utils import get_logger

logger = get_logger(__name__)

include_in_sample = [STATE, OBS, SUB_BELIEF, ACTION, REWARD, DONE, NEXT_OBS, NEXT_STATE, EPISODE_NBR, TIMESTEP, IS_RESET_STATE, NEXT_IS_RESET_STATE ]


class ReplayBuffer:
    """Replay Buffer

    This replay buffer supports input from multiple environments while still performing sanity checks and keeping
    episode data contiguous. This last part allows to be more memory efficient as previous actions, next observations
    and previous beliefs are not stored twice. Finally, it allows to update the previous belief in O(1) without having
    to rely on a lookup table.
    """

    def __init__(self,
                 elements_scheme: Dict[str, spaces.Space],
                 capacity: int,
                 policy_capacity: int,
                 belief_capacity: int,
                 wae_capacity: int,
                 horizon: int,
                 reset_obs_state: Union[int, float, ObsType],
                 nbr_feeds: int = 1,
                 batch_size: Optional[int] = None,
                 default_prev_action: Optional[Union[np.ndarray, int, float]] = None,
                 name: str = "main"
                 ):
        """
        Replay buffer constructor.

        :param elements_scheme: A dictionary mapping the keys of elements to store with their gym.Space.
        :param capacity: The number of transition to store in the replay buffer.
        :param policy_capacity: The number of transition of the original POMDP to store in the replay buffer.
        :param belief_capacity: The number of transition of the original POMDP augmented with a reset state to store
        in the replay buffer.
        :param wae_capacity: The number of transition of the perturbed POMDP to store in the replay buffer.
        :param horizon: The maximum size of an episode.
        :param reset_obs_state: The padding value used for the reset obs state.
        :param nbr_feeds: The number of environment that will be fed to the replay buffer in parallel.
        :param batch_size: The default batch size when sampling.
        :param default_prev_action: The default previous action (default to 0).
        :param name: Name of the replay buffer.
        """

        self.elements_scheme = elements_scheme
        self.capacity = capacity + 1
        self.policy_capacity = policy_capacity
        self.belief_capacity = belief_capacity
        self.wae_capacity = wae_capacity
        self.horizon = horizon

        if isinstance(reset_obs_state, (int, float)):
            obs = np.full(self.elements_scheme[OBS].sample(), reset_obs_state)
            state = np.zeros_like(self.elements_scheme[STATE].sample(), reset_obs_state)
            self._reset_obs_state = {OBS: obs, STATE: state}
        else:
            self._reset_obs_state = reset_obs_state.copy()

        if default_prev_action is None:
            default_prev_action = np.zeros_like(self.elements_scheme[ACTION].sample())
        elif not isinstance(default_prev_action, np.ndarray):
            default_prev_action = np.array(default_prev_action)
        self.default_prev_action = default_prev_action
        assert isinstance(self.elements_scheme[ACTION], gym.spaces.Discrete), "Action space should be discrete."
        self.nbr_actions = self.elements_scheme[ACTION].n

        self.nbr_feeds = nbr_feeds
        self.batch_size = batch_size
        self.name = name

        self.data = dict()
        self._check()
        self._allocate_data()
        self.last_data: Dict[int, dict] = dict()

        self.feeds: list[ReplayBuffer] = []
        feed_capacity = int(self.horizon + 150)
        self.epid_to_feed = {}
        if self.nbr_feeds > 1:
            self.feeds = [ReplayBuffer(
                elements_scheme=elements_scheme,
                capacity=feed_capacity,
                policy_capacity=feed_capacity,
                belief_capacity=feed_capacity,
                wae_capacity=feed_capacity,
                horizon=horizon,
                reset_obs_state=reset_obs_state,
                nbr_feeds=1,
                default_prev_action=default_prev_action,
                name=f"feed {i}"
            ) for i in range(self.nbr_feeds)]

            self.free_feed = set(range(self.nbr_feeds))

        self.index = 0
        self.episode_nbr = 0
        self.full = False
        self._empty = True
        self._policy_sample_idx = NPdequeue(int, self.policy_capacity)
        self._belief_sample_idx = NPdequeue(int, self.belief_capacity)
        self._wae_sample_idx = NPdequeue(int, self.wae_capacity)

    def reset_feeds(self):
        self.epid_to_feed = {}
        for feed in self.feeds:
            feed.reset()
        self.free_feed = set(range(self.nbr_feeds))

    def reset(self) -> None:
        """Reset the replay buffer"""
        t = max(self.capacity * self.full, self.index + 1)
        self.index = 0
        self._empty = True
        self.episode_nbr = 0
        self.last_data = dict()
        self._policy_sample_idx.empty()
        self._belief_sample_idx.empty()
        self._wae_sample_idx.empty()
        for key in self.data:
            self.data[key][:t] = 0
        self.full = False

    def _check(self) -> None:
        """Perform checks on the elements scheme"""
        if NEXT_OBS in self.elements_scheme and OBS not in self.elements_scheme:
            raise ValueError('Next observation requires current observation.')
        if STATE in self.elements_scheme and STATE not in self.elements_scheme:
            raise ValueError('Next state requires current state.')
        if PREV_ACTION in self.elements_scheme and PREV_ACTION not in self.elements_scheme:
            raise ValueError('Previous action requires current action.')
        if PREV_SUB_BELIEF in self.elements_scheme and SUB_BELIEF not in self.elements_scheme:
            raise ValueError('Previous belief requires current belief.')

    def _allocate_data(self) -> None:
        """Allocate the data."""
        if EPISODE_NBR not in self.elements_scheme.keys():
            self.elements_scheme[EPISODE_NBR] = spaces.Box(0, float('inf'), (), dtype=int)
        if TIMESTEP not in self.elements_scheme.keys():
            self.elements_scheme[TIMESTEP] = spaces.Box(0, float('inf'), (), dtype=int)
        if DONE not in self.elements_scheme.keys():
            self.elements_scheme[DONE] = spaces.Box(False, True, (), dtype=bool)
        if IS_RESET_STATE not in self.elements_scheme.keys():
            self.elements_scheme[IS_RESET_STATE] = spaces.Box(False, True, (), dtype=bool)
        if REWARD not in self.elements_scheme.keys():
            self.elements_scheme[REWARD] = spaces.Box(float('-inf'), float('inf'), (), dtype=float)
        for name, space in self.elements_scheme.items():
            if name in [NEXT_OBS, NEXT_STATE, PREV_ACTION, PREV_SUB_BELIEF]:
                continue
            shape = space.shape
            shape = (self.capacity,) + shape
            self.data[name] = np.zeros(shape=shape, dtype=space.dtype)

    def new_episode(self, reset_state_count: int = 0) -> int:
        """
        Get a new episode id and allocate a feed to it if needed.

        :param reset_state_count: The number of reset state count to add.
        :return: The new episode index.
        """
        self.episode_nbr += 1
        if self.nbr_feeds > 1:
            assert len(self.free_feed) > 0
            feed_nbr = self.free_feed.pop()
            self.epid_to_feed[self.episode_nbr] = feed_nbr
            self.feeds[feed_nbr].episode_nbr = self.episode_nbr
        self._add_reset_state(self.episode_nbr, reset_state_count)
        return self.episode_nbr

    def _add_reset_state(self, episode_nbr: int, n: int) -> None:
        """
        Utility method to add the reset states at the beginning of the episode.

        :param episode_nbr: The episode id.
        :param n: The number of reset states to add.
        """
        if n > 0:
            if self.nbr_feeds > 1:
                feed_nbr = self.epid_to_feed[episode_nbr]
                self.feeds[feed_nbr]._add_reset_state(episode_nbr, n)
            else:
                data = {
                    OBS: self._reset_obs_state[OBS][None].repeat(n, 0),
                    STATE: self._reset_obs_state[STATE][None].repeat(n, 0),
                    ACTION: self.default_prev_action[None].repeat(n, 0),
                    IS_RESET_STATE: np.ones((n,), bool),
                }
                timestep = np.arange(0, n)
                self._write_to_data(episode_nbr, timestep, data, remove=True)
                self._wae_sample_idx.enqueue_array(timestep[:-1])
                self._increase_index(n - 1)

    def episode_end(self, episode_nbr: int) -> None:
        """Mark an episode as finished, commit and release the feed if needed.

        :param episode_nbr: The episode id to mark as finished.
        """
        if self.nbr_feeds > 1:
            assert episode_nbr in self.epid_to_feed
            feed_nbr = self.epid_to_feed.pop(episode_nbr)
            feed = self.feeds[feed_nbr]
            feed._check_episode_ended(episode_nbr)
            episode_data = feed.get_data_dict(True)
            assert len(episode_data[OBS]) > 0
            is_first = self.index == 0 and not self.full
            if not is_first:
                # add index to all the indexes
                self._wae_sample_idx.enqueue(self.index)
                self._increase_index(1)

            batch_size = self._write_to_data(episode_nbr, episode_data[TIMESTEP], episode_data, remove=True)
            self._wae_sample_idx.enqueue_array((self.index + feed._wae_sample_idx.to_array()) % self.capacity)
            self._belief_sample_idx.enqueue_array((self.index + feed._belief_sample_idx.to_array()) % self.capacity)
            self._policy_sample_idx.enqueue_array((self.index + feed._policy_sample_idx.to_array()) % self.capacity)
            self._increase_index(batch_size - 1)
            self.free_feed.add(feed_nbr)
            feed.reset()

    def _check_episode_ended(self, episode_nbr: int) -> None:
        """Utility that check if an episode ended and issues warning if it did not.

        :param episode_nbr: The episode id to check.
        """
        assert self.data[EPISODE_NBR][self.index] == episode_nbr
        assert self.data[EPISODE_NBR][self.index - 1] == episode_nbr
        if not self.data[DONE][self.index - 1]:
            logger.warning(
                f"Ending episode {episode_nbr} while data does not contain done=True for that episode.")

    def _remove(self, index: Union[int, slice]) -> None:
        """Remove indexes from the data storage and last data if episode is no longer present in data storage.

        :param index: Index or slice of element(s) to remove.
        """
        if isinstance(index, int):
            index = slice(index, index + 1)
        for key, value in self.data.items():
            self.data[key][index] = 0

    def add(self, data: DictArray, last: bool = False) -> None:
        """Add elements to the replay buffer.

        :param data: The dict array to add can contain data from multiple episodes but only one timestep per episode.
        :param last: If true this data is the next obs/state/belief of the timestep after reaching done.
        """
        data = dict_array_to_dict_np(data)
        data = copy.deepcopy(data)
        self._check_data(data)
        episode_nbrs = set(np.unique(data[EPISODE_NBR]))
        assert episode_nbrs.issubset({self.episode_nbr, *self.epid_to_feed.keys()}), "Got unexpected episode numbers."
        if self.nbr_feeds > 1:
            for episode_nbr in episode_nbrs:
                episode_mask = data[EPISODE_NBR] == episode_nbr
                feed_nbr = self.epid_to_feed[episode_nbr]
                self.feeds[feed_nbr].add(mask_dict_array(data, episode_mask), last)
        else:
            self._add(data, last=last)

    def _add(self, data: DictArray, last=False) -> None:
        """Helper function to add data to the replay buffer.

        :param data: The dict array to add can contain data from only a single episode and one timestep.
        :param last: If true this data is the next obs/state of the timestep after reaching done.
        """
        assert len(np.unique(data[EPISODE_NBR])) == 1, "_add should not be called with data of different episodes."
        episode_nbr: int = data[EPISODE_NBR][0]
        assert episode_nbr == self.episode_nbr
        assert len(data[TIMESTEP]) == 1, "_add should not be called with data from multiple timesteps."
        timestep = data[TIMESTEP][0]
        remove = True
        if self.data[EPISODE_NBR][self.index] == episode_nbr:
            if self.data[TIMESTEP][self.index] == timestep:
                remove = False
            elif self.data[TIMESTEP][self.index] == timestep - 1:
                if True:  # not last:
                    if not self.data[IS_RESET_STATE][self.index]:
                        self._policy_sample_idx.enqueue(self.index)
                    self._belief_sample_idx.enqueue(self.index)
                self._wae_sample_idx.enqueue(self.index)
                self._increase_index()
            else:
                raise ValueError("Timestep mismatch.")
        else:
            assert not last
            if self._empty:
                self._empty = False
            else:
                self._increase_index()

        self._write_to_data(episode_nbr, data[TIMESTEP], data, remove=remove)

    def _write_to_data(self, episode_nbr: int, timestep: np.ndarray, data: DictArray, index: Optional[int] = None,
                       remove: bool = False) -> int:
        """Helper function to write to the data storage.
        Warning: does not update index !

        :param episode_nbr: The episode id associated with the data.
        :param timestep: Consecutive array containing the timestep for which to write the data.
        :param data: The data to write.
        :param index: Optional index from which to start writing the data. Default to own index.
        :param remove: If true, the current data of the indexes will be removed before writing.
        :return: The number of timesteps written.

        """
        assert is_consecutive(timestep), (episode_nbr, timestep, data)
        batch_size = self._check_data(data)
        assert len(timestep) == batch_size

        if index is None:
            index = self.index

        if index + batch_size <= self.capacity:
            if remove:
                self._remove(slice(index, index + batch_size))
            for key, value in data.items():
                if key in [EPISODE_NBR, TIMESTEP]:
                    continue
                self.data[key][index: index + batch_size] = array_to_np(value)
            self.data[EPISODE_NBR][index: index + batch_size] = episode_nbr
            self.data[TIMESTEP][index: index + batch_size] = timestep
        else:
            x = self.capacity - index
            data_split_1 = {key: value[:x] for key, value in data.items()}
            data_split_2 = {key: value[x:] for key, value in data.items()}
            self._write_to_data(episode_nbr, timestep[:x], data_split_1, index, remove)
            self._write_to_data(episode_nbr, timestep[x:], data_split_2, 0, remove)

        return batch_size

    def _increase_index(self, n: int = 1) -> None:
        """Increase the index by n

        :param n: The number to increase the index by.
        """
        self.index += n
        if self.index >= self.capacity:
            self.full = True
        self.index %= self.capacity

    def update_next_sub_belief(self, raw_indexes: np.ndarray, next_sub_beliefs: Array):
        next_sub_beliefs = array_to_np(next_sub_beliefs)
        (own_mask, feeds_masks), (own_index, feeds_indexes) = self._convert_raw_indexes('belief', raw_indexes)
        self.data[SUB_BELIEF][(own_index + 1) % self.capacity] = next_sub_beliefs[own_mask]
        for feed, feed_mask, feed_index in zip(self.feeds, feeds_masks, feeds_indexes):
            if feed_index is None:
                continue
            feed.data[SUB_BELIEF][feed_index + 1] = next_sub_beliefs[feed_mask]

    def get_capacity(self, worker: str) -> int:
        """Get the capacity of a worker.

        :param worker: The worker string identifier.
        :return: The capacity.
        """
        if worker == 'rl':
            capacity = self.policy_capacity
        elif worker == 'belief':
            capacity = self.belief_capacity
        elif worker == 'wae':
            capacity = self.wae_capacity
        else:
            raise ValueError(f"Unknown worker {worker}")
        return capacity

    def get_indexer(self, worker: str) -> NPdequeue:
        """Get the indexer of a worker.

        :param worker: The worker string identifier.
        :return: The indexer.
        """
        if worker == 'rl':
            indexer = self._policy_sample_idx
        elif worker == 'belief':
            indexer = self._belief_sample_idx
        elif worker == 'wae':
            indexer = self._wae_sample_idx
        else:
            raise ValueError(f"Unknown worker {worker}")
        return indexer

    def sample(self, worker: str, batch_size: Optional[int] = None) -> DictArray:
        """
        Sample a batch of transitions from the replay buffer.

        :param worker: The worker string identifier.
        :param batch_size: Optional batch size (default to batch_size attribute).
        :return: Batch of transitions.
        """
        raw_indexes = self._sample_raw_indexes(worker, batch_size)
        (own_mask, feeds_masks), (own_index, feeds_indexes) = self._convert_raw_indexes(worker, raw_indexes)

        data = {}
        for key in include_in_sample: # [STATE, OBS, SUB_BELIEF, ACTION, REWARD, DONE, NEXT_OBS, NEXT_STATE, ]:
            is_next = key in (NEXT_OBS, NEXT_STATE, NEXT_IS_RESET_STATE)
            key_ = inv_next_keys[key] if is_next else key
            if self.nbr_feeds == 1:
                data[key] = self._get_data_by_indexes(indexes=own_index, key=key_, query_next=is_next)
            else:
                data[key] = np.empty((batch_size,) + self.elements_scheme[key_].shape, self.elements_scheme[key_].dtype)
                for feed, f_mask, f_index in zip(self.feeds, feeds_masks, feeds_indexes):
                    if f_index is not None:
                        data[key][f_mask] = feed._get_data_by_indexes(f_index, key_, is_next)
                data[key][own_mask] = self._get_data_by_indexes(own_index, key_, is_next)

        data[RAW_INDEXES] = raw_indexes
        return data

    def _sample_raw_indexes(self, worker: str, batch_size: Optional[int]) -> np.ndarray:
        """
        Sample raw indexes.
        :param worker: The worker string identifier.
        :param batch_size: Optional batch size (default to batch_size attribute).
        :return: The raw indexes.
        """
        assert worker in {'rl', 'belief', 'wae'}

        capacity = self.get_capacity(worker)
        indexer = self.get_indexer(worker)

        if batch_size is None:
            assert self.batch_size is not None
            batch_size = self.batch_size
        assert isinstance(batch_size, int)

        count = len(indexer)
        feeds_indexer = [feed.get_indexer(worker) for feed in self.feeds]
        feeds_count = [len(indexer_) for indexer_ in feeds_indexer]
        total = count + sum(feeds_count)
        n = min(capacity, total)

        if total < batch_size:
            raise ValueError(f"Not enough data to sample (asking for {batch_size} while having {total} samples).")

        raw_indexes = np.random.choice(n, batch_size, replace=False)
        return raw_indexes

    def _convert_raw_indexes(self, worker: str, raw_indexes: np.ndarray) \
            -> Tuple[Tuple[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[Optional[np.ndarray]]]]:
        """
        Convert raw indexes into own and feed indexes with masks.

        :param worker: The worker string identifier.
        :param raw_indexes: Array of raw indexes.
        :return: (own_mask, feeds_masks), (own_index, feeds_indexes)
        """
        capacity = self.get_capacity(worker)
        indexer = self.get_indexer(worker)
        count = len(indexer)
        feeds_indexer = [feed.get_indexer(worker) for feed in self.feeds]
        feeds_count = [len(indexer_) for indexer_ in feeds_indexer]
        sum_feeds_count = sum(feeds_count)
        total = count + sum_feeds_count
        n = min(capacity, total)

        own_mask = raw_indexes >= sum_feeds_count
        own_index = raw_indexes[own_mask] - sum_feeds_count
        own_index = indexer.get_idx_in_n_elem(n - sum(feeds_count), own_index)

        feeds_masks = []
        feeds_indexes = []
        i = 0
        for feed, feed_count, feed_indexer in zip(self.feeds, feeds_count, feeds_indexer):
            mask = (i <= raw_indexes) & (raw_indexes < feed_count + i)
            if any(mask):
                f_index = feed_indexer.get_idx_in_n_elem(feed_count, raw_indexes[mask] - i)
            else:
                f_index = None
            feeds_masks.append(mask)
            feeds_indexes.append(f_index)
            i += feed_count

        assert np.all(own_mask + sum(feeds_masks) == 1)

        return (own_mask, feeds_masks), (own_index, feeds_indexes)

    def _get_data_by_indexes(self, indexes: np.ndarray, key: str, query_next: bool = False) -> np.ndarray:
        """
        Get the data by key and indexes.
        :param indexes: Array of indexes to query.
        :param key: The key, should not contain 'next' or 'prev'.
        :param query_next: If true returns the 'next' of the key.
        :return: The data.
        """
        batch = len(indexes)
        dones = self.data[DONE][indexes]

        if not query_next:
            return self.data[key][indexes]

        next_indexes = (indexes + 1) % self.capacity

        return self.data[key][next_indexes]

    def _check_data(self, data: DictArray) -> int:
        """
        Check that the dict array have a shape dimension and is the same for all the elements.

        :param data: The dict array to check.
        :return: The batch size.
        :raise ValueError: if the shape of an element does not match the element scheme or if they have different batch
         sizes.
        """
        batch_dim = None
        for key, value in data.items():
            if value.shape[1:] != self.elements_scheme[key].shape:
                raise ValueError(f"shape of {key} in data does not match elements_scheme.")
            batch_dim_ = value.shape[0]
            if batch_dim is None:
                batch_dim = batch_dim_
            if batch_dim_ != batch_dim:
                raise ValueError(f"Not all the elements have the same shape.")
        return batch_dim

    def _add_batch_dim(self, data: DictArray) -> DictArray:
        """
        Add a batch dimension if not present.

        :param data: Input dict array.
        :return: Output dict array.
        """
        key, value = list(data.items())[0]
        value = array_to_np(value)
        if self.data[key].shape[1:] == value.shape:
            for key, value in data.items:
                data[key] = data[key][None]
        return data

    def get_data_dict(self, copy=False) -> DictArray:
        """
        Get the data from the replay buffer in a dict array. This does not include data marked as `last`.

        :param copy: Whether to copy to data.
        :return: The dict array.
        """
        if self.full:
            if copy:
                return self.data.copy()
            return self.data
        data = {key: value[:self.index + 1] for key, value in self.data.items()}
        if copy:
            data = data.copy()
        return data

    def as_dataset(self, batch_size: Optional[int], num_parallel_calls: Optional[int] = None,
                   worker: str = 'wae') -> tf.data.Dataset:
        """
        Build a tensorflow dataset.

        :param batch_size: Optional batch size.
        :param num_parallel_calls: Optional number of parallel calls.
        :param worker: The target worker to use for the sampling (default to 'wae').
        :return: The Tensorflow dataset.
        """

        def get_next():
            while True:
                batch = self.sample(worker, batch_size)
                # no_zero_rank = lambda x: x if tf.rank(x) > 1 else tf.expand_dims(x, axis=-1)
                batch_tf = dict_array_to_dict_tf(batch)
                yield (batch_tf[STATE], batch_tf[OBS]), \
                       tf.cast(batch_tf[IS_RESET_STATE], tf.float32)[..., None], \
                       tf.one_hot(batch_tf[ACTION], self.nbr_actions), \
                       tf.cast(batch_tf[REWARD], tf.float32)[..., None], \
                       (batch_tf[NEXT_STATE], batch_tf[NEXT_OBS]), \
                       tf.cast(batch_tf[NEXT_IS_RESET_STATE], tf.float32)[..., None],

        output_signature = (
            (
                tf.TensorSpec((batch_size,) + self.elements_scheme[STATE].shape, dtype=tf.float32),
                tf.TensorSpec((batch_size,) + self.elements_scheme[OBS].shape, dtype=tf.float32),
            ),
            tf.TensorSpec((batch_size, 1), dtype=tf.float32),
            tf.TensorSpec((batch_size, self.nbr_actions), dtype=tf.float32),
            tf.TensorSpec((batch_size, 1), dtype=tf.float32),
            (
                tf.TensorSpec((batch_size,) + self.elements_scheme[STATE].shape, dtype=tf.float32),
                tf.TensorSpec((batch_size,) + self.elements_scheme[OBS].shape, dtype=tf.float32),
            ),
            tf.TensorSpec((batch_size, 1), dtype=tf.float32),
        )

        return tf.data.Dataset.from_generator(get_next, output_signature=output_signature)

    def as_belief_dataset(self, batch_size: Optional[int], num_parallel_calls: Optional[int] = None,
                          worker: str = 'belief') -> tf.data.Dataset:
        """
        Build a tensorflow dataset.

        :param batch_size: Optional batch size.
        :param num_parallel_calls: Optional number of parallel calls.
        :param worker: The target worker to use for the sampling (default to 'wae').
        :return: The Tensorflow dataset.
        """

        def get_next():
            while True:
                batch = self.sample(worker, batch_size)
                # no_zero_rank = lambda x: x if tf.rank(x) > 1 else tf.expand_dims(x, axis=-1)
                batch_tf = dict_array_to_dict_tf(batch)
                yield batch_tf[STATE], \
                    batch_tf[OBS], \
                    tf.cast(batch_tf[IS_RESET_STATE], tf.float32)[..., None], \
                    tf.one_hot(batch_tf[ACTION], self.nbr_actions), \
                    batch_tf[SUB_BELIEF]

        output_signature = (
            tf.TensorSpec((batch_size,) + self.elements_scheme[STATE].shape, dtype=tf.float32),
            tf.TensorSpec((batch_size,) + self.elements_scheme[OBS].shape, dtype=tf.float32),
            tf.TensorSpec((batch_size, 1), dtype=tf.float32),
            tf.TensorSpec((batch_size, self.nbr_actions), dtype=tf.float32),
            tf.TensorSpec((batch_size,) + self.elements_scheme[SUB_BELIEF].shape, dtype=tf.float32),
        )

        return tf.data.Dataset.from_generator(get_next, output_signature=output_signature)

    def __repr__(self) -> str:
        repr_ = f"Replay Buffer ({self.name}): {len(self._wae_sample_idx)}/{self.wae_capacity} WAE samples " \
               f"- {len(self._belief_sample_idx)}/{self.belief_capacity} Belief samples " \
               f"- {len(self._policy_sample_idx)}/{self.policy_capacity} Policy samples."
        return repr_

    def checkpoint(self, checkpoint_dir):
        data = {
            "data": self.data,
            "index": self.index,
            "full": self.full,
            "belief_sample_idx": self._belief_sample_idx.as_data(),
            "policy_sample_idx": self._policy_sample_idx.as_data(),
            "wae_sample_idx": self._wae_sample_idx.as_data(),
        }
        np.save(os.path.join(checkpoint_dir, "replay_buffer.npy"), data)

    def restore(self, checkpoint_dir):
        data = np.load(os.path.join(checkpoint_dir, "replay_buffer.npy"), allow_pickle=True)
        data = data[()]
        self.data = data["data"]
        self.index = data["index"]
        self.full = data["full"]
        self._belief_sample_idx.load_data(data["belief_sample_idx"])
        self._policy_sample_idx.load_data(data["policy_sample_idx"])
        self._wae_sample_idx.load_data(data["wae_sample_idx"])

