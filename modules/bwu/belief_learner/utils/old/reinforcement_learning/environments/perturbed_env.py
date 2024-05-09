from typing import Any, Optional, Text
import tensorflow as tf
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.typing import types
from tf_agents.trajectories import time_step as ts
import numpy as np


class NoisyEnvironment(PyEnvironmentBaseWrapper):

    def __init__(self, env: Any, state_noise: Optional[float] = 0., action_noise: Optional[float] = 0):
        super().__init__(env)
        self.state_noise = state_noise
        self.action_noise = action_noise

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self.action_noise > 0:
            _action = np.random.multivariate_normal(
                mean=action,
                cov=np.diag(self.action_noise ** 2 * np.ones(shape=np.shape(action))))
        else:
            _action = action

        time_step = self.wrapped_env().step(_action)

        if self.state_noise > 0:
            _observation = np.random.multivariate_normal(
                mean=time_step.observation,
                cov=np.diag(self.state_noise ** 2 * np.ones(shape=np.shape(time_step.observation)))
            ).astype(time_step.observation.dtype)
        else:
            _observation = time_step.observation

        return time_step._replace(observation=_observation)


class PerturbedEnvironment(PyEnvironmentBaseWrapper):
    """
    Implementation of the perturbed environment presented in
    Huang et al. 2020: Steady State Analysis of Episodic Reinforcement Learning,
    allowing to enforce an ergodic episodic RL process.
    """

    def __init__(
            self,
            env: Any,
            perturbation: float,
            recursive_perturbation: bool = False,
            rendering=False,
            enforce_reset_to_null_once: bool = True,
    ):
        super(PerturbedEnvironment, self).__init__(env)
        self.perturbation = np.clip(perturbation, a_min=1e-12, a_max=1. - 1e-12)
        self.recursive_perturbation = recursive_perturbation
        self._in_null_state = False
        self._initialized = False
        self._last_rendering = None
        self._handle_auto_reset = True
        self._enforce_reset_to_null_once = enforce_reset_to_null_once and recursive_perturbation
        self.rendering = rendering

    def _reset(self):
        self._in_null_state = (
                (not self._in_null_state or self.recursive_perturbation) and
                self._initialized and
                np.random.uniform() <= self.perturbation) or (
                self._enforce_reset_to_null_once and self._initialized and not self._in_null_state)
        if self._in_null_state:
            return ts.transition(
                observation=np.zeros(
                    shape=self.observation_spec().shape,
                    dtype=self.observation_spec().dtype),
                reward=np.zeros(
                    shape=self.reward_spec().shape,
                    dtype=self.reward_spec().dtype))
        else:
            time_step = super(PerturbedEnvironment, self)._reset()
            if not self._initialized:
                self._initialized = True
                if self.rendering:
                    try:
                        self.render(mode='rgb_array')
                    except Exception:
                        pass
            return time_step

    def _step(self, action):
        if self._in_null_state:
            return self._reset()
        else:
            return super(PerturbedEnvironment, self)._step(action)

    def render(self, mode: Text = 'rgb_array') -> types.NestedArray:
        if not self._in_null_state or self._last_rendering is None:
            self._last_rendering = super(PerturbedEnvironment, self).render(mode)
        return self._last_rendering
