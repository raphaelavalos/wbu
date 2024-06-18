from typing import Union, Dict, Optional

import gym
import numpy as np
# from gym.core import ObsType
from gym.wrappers import TransformObservation

from belief_learner.utils.definitions import OBS, STATE

from belief_learner.utils.logging import get_logger

logger = get_logger(__name__)

def get_reset_state(space: gym.Space, pad_value: float = 0.):
    if isinstance(space, gym.spaces.Dict):
        reset_state = {key: get_reset_state(value, pad_value) for key, value in space.items()}
    elif isinstance(space, gym.spaces.Discrete):
        assert isinstance(pad_value, int)
        reset_state = pad_value
    elif isinstance(space, gym.Space):
        reset_state = space.sample()
        reset_state.fill(pad_value)
        assert np.isclose(reset_state.flat[0], pad_value)
    else:
        raise ValueError("space is not a gym.Space.")
    return reset_state


def are_close(element_1: Union[int, float, Dict, np.ndarray], element_2: Union[int, float, Dict, np.ndarray]):
    if issubclass(type(element_1), dict):
        assert issubclass(type(element_2), dict)
        assert set(element_1.keys()) == set(element_2.keys())
        for key in element_1.keys():
            if are_close(element_1[key], element_2[key]):
                return True
    elif isinstance(element_1, (int, float, np.ndarray)):
        assert element_1.__class__ == element_2.__class__
        if np.all(np.isclose(element_1, element_2)):
            return True
    return False


def env_creator(env_config: dict):
    env_config = env_config.copy()
    env_name = env_config.pop('env_name')
    env = gym.make(env_name)

    def f(obs):
        return {OBS: obs, STATE: obs}

    if not isinstance(env.observation_space, gym.spaces.Dict):
        logger.warning("The environment observation is not a dict, transforming it by passing the "
                       "observation as {'obs': obs, 'state': state}.")
        env = TransformObservation(env, f)
        env.observation_space = gym.spaces.Dict({OBS: env.observation_space, STATE: env.observation_space})

    assert set(env.observation_space.keys()) == {'state', 'obs'}
    env = PerturbedGymWrapper(env, **env_config)
    return env


class PerturbedGymWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            perturbation: float,
            recursive_perturbation: bool = False,
            enforce_reset_to_null_once: bool = True,
            reset_state_pad: float = 0.,
            init_in_reset_state: bool = False,
            reset_state_in_info: bool = True,
            reset_state: Optional = None,
    ) -> None:
        super(PerturbedGymWrapper, self).__init__(env)
        self.reset_state_pad = reset_state_pad
        self.perturbation = np.clip(perturbation, a_min=1e-12, a_max=1. - 1e-12)
        self.recursive_perturbation = recursive_perturbation
        self._in_reset_state = False
        if reset_state is None:
            self._reset_state = get_reset_state(self.observation_space, self.reset_state_pad)
        else:
            self._reset_state = reset_state.copy()
        self.init_in_reset_state = init_in_reset_state
        self._initialized = init_in_reset_state
        if enforce_reset_to_null_once and not recursive_perturbation:
            logger.warning("Trying to enforce reset to null once while recursive perturbation "
                           "is disabled. Setting enforce reset to null once to false.")
            enforce_reset_to_null_once = False
        self._enforce_reset_to_null_once = enforce_reset_to_null_once
        self.reset_state_in_info = reset_state_in_info
        if self.init_in_reset_state and self.reset_state_in_info:
            raise ValueError("Can not put the reset state in info and initialize with reset state.")

    def reset(self):
        if not self.reset_state_in_info:
            # Deal with reset state here
            self._reset_state()
        if self._in_reset_state:
            return self._reset_state
        else:
            obs = super(PerturbedGymWrapper, self).reset()
            if not self._initialized:
                self._initialized = True
            return obs

    def _loop_reset_state(self):
        in_reset_state = ((not self._in_reset_state or self.recursive_perturbation)
                                and self._initialized and np.random.uniform() <= self.perturbation) \
                               or (self._enforce_reset_to_null_once and self._initialized and not self._in_reset_state)
        self._in_reset_state = in_reset_state

    def step(self, action):
        if self._in_reset_state:
            assert not self.reset_state_in_info
            return self.reset(), 0., False, {'reset': True}
        else:
            state, reward, done, info = super(PerturbedGymWrapper, self).step(action)
            if not done and are_close(self._reset_state, state):
                raise ValueError(f"The environment returned a state too close to the reset_state. "
                                 f"{state}, {self._reset_state}")
            # Idea: do the reset_state thing here if done=True and put it in info?
            if self.reset_state_in_info and done:
                self._loop_reset_state()
                i = 0
                while self._in_reset_state:
                    i += 1
                    self._loop_reset_state()
                info["reset_state_count"] = i
            return state, reward, done, info
