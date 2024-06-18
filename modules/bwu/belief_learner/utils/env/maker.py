from typing import Optional

import gym
import numpy as np
from gym.wrappers import OrderEnforcing

from belief_learner.utils.env.atari_loader import FlickeringObs
from belief_learner.utils.env.observation_flattener import FlattenObservationSpaceWrapper, BoxifyWrapper, \
    ResetObsWrapper, ResetObsWrapperPOMinAtar
from belief_learner.utils.env.perturbed_env_wrapper import PerturbedGymWrapper


class StateObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert not isinstance(self.env.observation_space, gym.spaces.Dict)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": self.env.observation_space,
                "state": self.env.observation_space,
            }
        )

    def observation(self, observation):
        state_obs = {'obs': observation, 'state': observation}
        return state_obs

class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, max_abs_reward: Optional[float] = None):
        self._max_abs_reward = max_abs_reward
        super().__init__(env)

    def reward(self, reward):
        if self._max_abs_reward is not None:
            reward = reward / self._max_abs_reward
        return reward


def env_maker(env_name: str,
              p_blank: float = .5,
              perturbation: float = 0.75,
              recursive_perturbation: bool = True,
              enforce_reset_to_null_once: bool = True,
              reset_state_pad: float = 0.,
              init_in_reset_state: bool = False,
              reset_state_in_info: bool = True,
              max_abs_reward: Optional[float] = None,
              ):
    if 'popgym' in env_name:
        import popgym
        # from popgym.wrappers.observability_wrapper import ObservabilityWrapper, Observability
        from popgym.wrappers.markovian import Markovian, Observability
        env = gym.make(env_name) #, with_state=True)
        if isinstance(env, OrderEnforcing):
            env = env.env
        env = ResetObsWrapper(env)
        env = FlattenObservationSpaceWrapper(env)
        env = BoxifyWrapper(env)
        # env = ObservabilityWrapper(env, observability_level=Observability.FULL_AND_PARTIAL)
        env = Markovian(env, observability=Observability.FULL_AND_PARTIAL)
    elif 'POMinAtar' in env_name:
        from popgym.wrappers.markovian import Markovian, Observability
        env = gym.make(env_name) #, with_state=True)
        if isinstance(env, OrderEnforcing):
            env = env.env
        env = ResetObsWrapperPOMinAtar(env)
        env = Markovian(env, observability=Observability.FULL_AND_PARTIAL)
    else:
        env = gym.make(env_name)
    horizon = getattr(env, 'episode_length', None)
    horizon = getattr(env, 'max_episode_length', horizon)
    if horizon is None and hasattr(env, 'spec'):
        horizon = getattr(env.spec, 'max_episode_steps', None)
    elif horizon is None:
        horizon = getattr(env, '_horizon')
    env.horizon = horizon
    if not isinstance(env.observation_space, gym.spaces.Dict):
        env = StateObsWrapper(env)
    if max_abs_reward is not None:
        env = RewardScaleWrapper(env, max_abs_reward)
    if p_blank > 0.:
        env = FlickeringObs(env, p_blank)

    reset_state = {"obs": env.reset_state_obs, "state": env.reset_state_state}

    env = PerturbedGymWrapper(env,
                              perturbation,
                              recursive_perturbation,
                              enforce_reset_to_null_once,
                              reset_state_pad,
                              init_in_reset_state,
                              reset_state_in_info,
                              reset_state)
    return env
