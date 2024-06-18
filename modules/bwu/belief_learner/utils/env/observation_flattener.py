from typing import Callable, Optional, List

import gym
import numpy as np
import popgym
import tensorflow as tf
import tree
from gym import Env, spaces
from gym.core import ObsType
from popgym.core.env import POPGymEnv

from belief_learner.utils.costs import get_cost_fn


class POPGymWrapper(gym.Wrapper, POPGymEnv):

    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(env, POPGymEnv)
        self._state_space: Optional[spaces.Space] = None

    @property
    def state_space(self) -> spaces.Space:
        """Returns the observation space of the environment."""
        if self._state_space is None:
            return self.env.state_space
        return self._state_space

    @state_space.setter
    def state_space(self, space: spaces.Space):
        self._state_space = space


class POPGymObservationWrapper(POPGymWrapper):
    def __init__(self, env: POPGymEnv):
        super().__init__(env)
        assert isinstance(env.unwrapped, POPGymEnv)
        self.observation_space = self.map_space(self.observation_space)
        self.preprocess_fn = self.get_obs_mapper(self.env.observation_space)
        self.state_space = self.map_space(self.state_space)
        self.state_preprocess_fn = self.get_obs_mapper(self.env.state_space)
        if hasattr(self, 'reset_state_obs'):
            self.reset_state_obs = self.preprocess_fn(self.env.reset_state_obs)
        if hasattr(self, 'reset_state_state'):
            self.reset_state_state = self.state_preprocess_fn(self.env.reset_state_state)

    def observation(self, observation):
        return self.preprocess_fn(observation)

    def get_state(self):
        state = self.env.get_state()
        return self.state_preprocess_fn(state)

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
            return self.observation(obs), info
        else:
            return self.observation(self.env.reset(**kwargs))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    @staticmethod
    def map_space(space: gym.Space) -> gym.Space:
        raise NotImplementedError

    @staticmethod
    def get_obs_mapper(space: gym.Space) -> Callable[[ObsType], ObsType]:
        raise NotImplementedError


class ResetObsWrapper(POPGymObservationWrapper):
    def __init__(self, env: POPGymEnv):
        super(ResetObsWrapper, self).__init__(env)
        # self.observation_space = gym.spaces.Tuple((gym.spaces.MultiBinary(1), self.env.observation_space))
        # self.state_space = gym.spaces.Tuple((gym.spaces.MultiBinary(1), self.env.state_space))
        self.reset_state_obs = (np.array(1, dtype=np.int8), space_to_zero_state(self.env.observation_space))
        self.reset_state_state = (np.array(1, dtype=np.int8), space_to_zero_state(self.env.state_space))

    def get_state(self):
        return np.array(0, dtype=np.int8), self.env.get_state()

    @staticmethod
    def map_space(space: gym.Space) -> gym.Space:
        return gym.spaces.Tuple((gym.spaces.MultiBinary(1), space))

    @staticmethod
    def get_obs_mapper(space: gym.Space) -> Callable[[ObsType], ObsType]:
        def add_reset_state(obs: ObsType):
            return np.array(0, dtype=np.int8), obs

        return add_reset_state


class ResetObsWrapperPOMinAtar(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(ResetObsWrapperPOMinAtar, self).__init__(env)
        self.reset_state_obs = space_to_zero_state(self.env.observation_space)
        self.reset_state_state = space_to_zero_state(self.env.state_space)


class FlattenObservationSpaceWrapper(POPGymObservationWrapper):
    @staticmethod
    def map_space(space: gym.Space) -> gym.Space:
        if isinstance(space, gym.spaces.Box):
            space = gym.spaces.Box(space.low.flatten(), space.high.flatten(), (np.prod(space.shape).item(),))
        elif isinstance(space, gym.spaces.Discrete):
            pass
        elif isinstance(space, gym.spaces.MultiDiscrete):
            space = gym.spaces.MultiDiscrete(nvec=space.nvec.flatten())
        elif isinstance(space, gym.spaces.MultiBinary):
            space = gym.spaces.MultiBinary(n=np.prod(space.shape))
        elif isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)):
            space = gym.spaces.Tuple(tuple(map(FlattenObservationSpaceWrapper.map_space, tree.flatten(space))))
        else:
            raise ValueError
        return space

    @staticmethod
    def get_obs_mapper(space: gym.Space):
        if isinstance(space, (gym.spaces.Box, gym.spaces.MultiBinary, gym.spaces.MultiDiscrete)):
            def flatten(obs: ObsType):
                return obs.flatten()

            return flatten
        elif isinstance(space, gym.spaces.Discrete):
            def identity(obs: ObsType):
                return obs

            return identity
        elif isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict)):
            sub_fun = list(map(FlattenObservationSpaceWrapper.get_obs_mapper, tree.flatten(space)))

            def flatten(obs: ObsType):
                return tuple(f(o) for f, o in zip(sub_fun, tree.flatten(obs)))

            return flatten


class BoxifyWrapper(POPGymObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.pre_boxify_obs_space = self.env.observation_space
        self.pre_boxify_state_space = self.env.state_space
        if not isinstance(self.pre_boxify_obs_space, gym.spaces.Tuple):
            self.pre_boxify_obs_space_shapes = [BoxifyWrapper.space_box_shape(self.pre_boxify_obs_space)]
        else:
            self.pre_boxify_obs_space_shapes = list(map(BoxifyWrapper.space_box_shape, self.pre_boxify_obs_space))
        if not isinstance(self.pre_boxify_state_space, gym.spaces.Tuple):
            self.pre_boxify_state_space_shapes = [BoxifyWrapper.space_box_shape(self.pre_boxify_state_space)]
        else:
            self.pre_boxify_state_space_shapes = list(map(BoxifyWrapper.space_box_shape, self.pre_boxify_state_space))
        if not hasattr(self, 'reset_state_obs'):
            self.reset_state_obs = np.concatenate([space_to_reset_state(space) for space in self.pre_boxify_obs_space])
            self.reset_state_state = np.concatenate(
                [space_to_reset_state(space) for space in self.pre_boxify_state_space])
            self.reset_state = self.reset_state_obs.copy()

    def boxify_info(self):
        info = {}
        info[popgym.OBS] = {
            "pre_boxify_space": self.pre_boxify_obs_space,
            "pre_boxify_shapes": self.pre_boxify_obs_space_shapes,
        }
        info[popgym.STATE] = {
            "pre_boxify_space": self.pre_boxify_state_space,
            "pre_boxify_shapes": self.pre_boxify_state_space_shapes,
        }

    @staticmethod
    def space_box_shape(space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Box):
            n = np.prod(space.shape).item()
        elif isinstance(space, gym.spaces.Discrete):
            n = space.n
        elif isinstance(space, gym.spaces.MultiDiscrete):
            n = sum(space.nvec)
        elif isinstance(space, gym.spaces.MultiBinary):
            n = np.prod(space.shape).item()
        else:
            raise ValueError
        return n

    @staticmethod
    def map_space(space: gym.Space) -> gym.Space:
        if isinstance(space, gym.spaces.Box):
            space = gym.spaces.Box(space.low.flatten(), space.high.flatten(), (BoxifyWrapper.space_box_shape(space),))
        elif isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete, gym.spaces.MultiBinary)):
            space = gym.spaces.Box(0, 1, (BoxifyWrapper.space_box_shape(space),))
        elif isinstance(space, gym.spaces.Tuple):
            spaces = tuple(map(BoxifyWrapper.map_space, space))
            dim = sum(map(lambda x: x.shape[0], spaces))
            low = np.concatenate([space.low for space in spaces], 0)
            high = np.concatenate([space.high for space in spaces], 0)
            space = gym.spaces.Box(low, high, (dim,))
        else:
            raise ValueError
        return space

    @staticmethod
    def get_obs_mapper(space: gym.Space):
        def flatten(obs: ObsType):
            return obs.flatten().astype(np.float32)

        def get_one_hot(n):
            array = np.eye(n, dtype=np.float32)

            def one_hot(i):
                return array[i]

            return one_hot

        def multi_discrete_one_hot(nvec):
            one_hot_fns = [get_one_hot(n) for n in nvec]

            def multi_one_hot(array):
                array = [one_hot(int(a)) for one_hot, a in zip(one_hot_fns, array)]
                return np.concatenate(array, axis=0)

            return multi_one_hot

        def concat_tuple(space_tulpe):
            fns = tuple(map(BoxifyWrapper.get_obs_mapper, space_tulpe.spaces))

            def apply_and_concat(obs_tuple):
                obs_list = [fn(obs) for fn, obs in zip(fns, obs_tuple)]
                return np.concatenate(obs_list, axis=0)

            return apply_and_concat

        if isinstance(space, gym.spaces.Box):
            return flatten
        elif isinstance(space, gym.spaces.Discrete):
            return get_one_hot(space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return multi_discrete_one_hot(space.nvec)
        elif isinstance(space, gym.spaces.MultiBinary):
            return flatten
        elif isinstance(space, gym.spaces.Tuple):
            return concat_tuple(space)
        else:
            raise ValueError


def split_boxify_np(array: np.ndarray, pre_boxify_obs_space_shapes: List[int]):
    splits = np.cumsum(pre_boxify_obs_space_shapes)[:-1]
    split_arrays = np.split(array, splits, -1)
    return split_arrays


def split_boxify_tf(tensor: tf.Tensor, pre_boxify_space_shapes: List[int]):
    return tf.split(tensor, pre_boxify_space_shapes, axis=-1)


def space_to_tf_loss(space, box_loss: Optional[str] = None) -> Callable:
    if box_loss is None:
        box_loss = 'l22'
    if isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        return get_cost_fn(box_loss)
    elif isinstance(space, gym.spaces.Discrete):
        return tf.nn.softmax_cross_entropy_with_logits
    elif isinstance(space, gym.spaces.MultiDiscrete):
        def multi_discrete_loss(x, y):
            return sum(tf.nn.softmax_cross_entropy_with_logits(x_, y_)
                       for x_, y_ in zip(tf.split(x, space.nvec, axis=-1),
                                         tf.split(y, space.nvec, axis=-1)))

        return multi_discrete_loss
    elif isinstance(space, gym.spaces.MultiBinary):
        return lambda *x: tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(*x), axis=-1)
    else:
        raise NotImplementedError


def space_to_tf_recover(space) -> Callable:
    if isinstance(space, gym.spaces.Box):
        return tf.identity
    elif isinstance(space, gym.spaces.Discrete):
        return tf.nn.softmax
    elif isinstance(space, gym.spaces.MultiDiscrete):
        def multi_softmax(x):
            return tf.concat([tf.nn.softmax(x_) for x_ in tf.split(x, space.nvec, axis=-1)], axis=-1)

        return multi_softmax
    elif isinstance(space, gym.spaces.MultiBinary):
        return tf.math.sigmoid
    else:
        raise NotImplementedError


def space_to_tf_ml(space) -> Callable:
    def ml_discrete(x):
        return tf.one_hot(tf.argmax(x, axis=-1), x.shape[-1])

    if isinstance(space, gym.spaces.Box):
        return tf.identity
    elif isinstance(space, gym.spaces.Discrete):
        return ml_discrete
    elif isinstance(space, gym.spaces.MultiDiscrete):
        def multi_discrete(x):
            return tf.concat([ml_discrete(x_) for x_ in tf.split(x, space.nvec, axis=-1)], axis=-1)

        return multi_discrete
    elif isinstance(space, gym.spaces.MultiBinary):
        def binary_ml(x):
            return tf.cast(x > 0, dtype=tf.float32)

        return binary_ml
    else:
        raise NotImplementedError


def space_to_weight_loss(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        return space.shape[0]
    elif isinstance(space, gym.spaces.Discrete):
        return 1
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return len(space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return space.n
    else:
        raise NotImplementedError


def space_to_reset_state(space: gym.Space) -> np.ndarray:
    if isinstance(space, gym.spaces.Box):
        assert len(space.shape) == 1
        return np.full(space.shape, -1)
    elif isinstance(space, gym.spaces.Discrete):
        return np.full((space.n,), 1 / space.n)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.concatenate([np.full((n,), 1 / n) for n in space.nvec])
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.array([.5, ])
    else:
        raise NotImplementedError


def space_to_zero_state(space: gym.Space) -> ObsType:
    if isinstance(space, gym.spaces.Box):
        # assert len(space.shape) == 1
        # commenting might break things
        zero = np.zeros(space.shape, dtype=np.float32)
        if zero not in space:
            zero = space.low.copy()
            assert float('-inf') not in zero
        return zero
    elif isinstance(space, gym.spaces.Discrete):
        return np.array(0, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros((len(space.nvec),), dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.zeros((space.n,), dtype=space.dtype)
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(space_to_zero_state(space_) for space_ in space)
    elif isinstance(space, gym.spaces.Dict):
        return {key: space_to_zero_state(space_) for key, space_ in space.items()}
    else:
        raise NotImplementedError


def boxify_to_loss(pre_boxify_space, pre_boxify_space_shapes, box_loss: Optional[str] = None):
    losses = [space_to_tf_loss(space, box_loss) for space in pre_boxify_space]
    weights = sum([space_to_weight_loss(space) for space in pre_boxify_space])

    def loss(x, y):
        x = split_boxify_tf(x, pre_boxify_space_shapes)
        y = split_boxify_tf(y, pre_boxify_space_shapes)
        return sum(loss_(x_, y_) for loss_, x_, y_ in zip(losses, x, y)) / weights

    return loss


def boxify_to_recover(pre_boxify_space, pre_boxify_space_shapes):
    recover_f = [space_to_tf_recover(space) for space in pre_boxify_space]

    def recover(x):
        x = split_boxify_tf(x, pre_boxify_space_shapes)
        return tf.concat([recover_f_(x_) for recover_f_, x_ in zip(recover_f, x)], axis=-1)

    return recover


def boxify_to_ml(pre_boxify_space, pre_boxify_space_shapes):
    ml_f = [space_to_tf_ml(space) for space in pre_boxify_space]

    def ml(x):
        x = split_boxify_tf(x, pre_boxify_space_shapes)
        return tf.concat([recover_f_(x_) for recover_f_, x_ in zip(ml_f, x)], axis=-1)

    return ml
