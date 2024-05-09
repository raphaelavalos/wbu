from typing import Callable, Optional, Union
import tensorflow as tf
from tf_agents import specs
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.policies import TFPolicy
from tf_agents.typing import types
from tf_agents.trajectories import time_step as ts, policy_step

from belief_learner.utils.old.io import dataset_generator


class DiscreteActionTFEnvironmentWrapper(TFEnvironment):
    def __init__(self,
                 tf_env: TFEnvironment,
                 action_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 number_of_discrete_actions: int,
                 reward_scaling: Optional[float] = 1.):
        super(DiscreteActionTFEnvironmentWrapper, self).__init__(
            time_step_spec=tf_env.time_step_spec(),
            action_spec=specs.BoundedTensorSpec(
                shape=(),
                dtype=tf.int64,
                minimum=0,
                maximum=number_of_discrete_actions - 1,
                name='latent_action'),
            batch_size=tf_env.batch_size)
        self.wrapped_env: TFEnvironment = tf_env
        self.action_embedding_function = action_embedding_function
        self.reward_scaling = reward_scaling

    def _current_time_step(self):
        time_step = self.wrapped_env.current_time_step()
        return time_step._replace(reward=self.reward_scaling * time_step.reward)

    def _reset(self):
        time_step = self.wrapped_env.reset()
        return time_step._replace(reward=self.reward_scaling * time_step.reward)

    def _step(self, latent_action):
        real_action = self.action_embedding_function(self.current_time_step().observation, latent_action)
        time_step = self.wrapped_env.step(real_action)
        return time_step._replace(reward=self.reward_scaling * time_step.reward)

    def render(self):
        return self.wrapped_env.render()


class LatentEmbeddingTFEnvironmentWrapper(TFEnvironment):
    def __init__(
            self,
            tf_env: TFEnvironment,
            state_embedding_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
            action_embedding_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
            labeling_fn: Callable[[tf.Tensor], tf.Tensor],
            latent_state_size: int,
            number_of_discrete_actions: int,
            reward_scaling: Optional[float] = 1.,
    ):
        super(LatentEmbeddingTFEnvironmentWrapper, self).__init__(
            time_step_spec=ts.time_step_spec(
                observation_spec={
                    'state': tf_env.observation_spec(),
                    'latent_state': specs.BoundedTensorSpec(
                        shape=(latent_state_size,),
                        dtype=tf.int32,
                        minimum=0,
                        maximum=1,
                        name='latent_state'
                    )}),
            action_spec=specs.BoundedTensorSpec(
                shape=(),
                dtype=tf.int64,
                minimum=0,
                maximum=number_of_discrete_actions - 1,
                name='latent_action'),
            batch_size=tf_env.batch_size)
        self._wrapped_env: TFEnvironment = tf_env
        self.state_embedding_fn = state_embedding_fn
        self.action_embedding_fn = action_embedding_fn
        self.reward_scaling = reward_scaling
        self.labeling_fn = dataset_generator.ergodic_batched_labeling_function(labeling_fn)
        self._current_latent_state = None

    def __getattr__(self, name):
        return getattr(self._wrapped_env, name)

    def _current_time_step(self):
        if self._current_latent_state is None:
            return self.reset()

        time_step = self._wrapped_env.current_time_step()
        return time_step._replace(
            observation={
                'state': time_step.observation,
                'latent_state': self._current_latent_state},
            reward=self.reward_scaling * time_step.reward)

    def _reset(self):
        time_step = self._wrapped_env.reset()
        label = self.labeling_fn(time_step.observation)
        self._current_latent_state = self.state_embedding_fn(tf.cast(time_step.observation, tf.float32), label)
        return self._current_time_step()

    def _step(self, latent_action):
        latent_action = self.action_embedding_fn(self._current_latent_state, latent_action)
        next_time_step = self._wrapped_env.step(latent_action)
        next_state_label = self.labeling_fn(next_time_step.observation)
        next_latent_state = self.state_embedding_fn(next_time_step.observation, next_state_label)
        self._current_latent_state = next_latent_state
        return next_time_step._replace(
            observation={
                'state': next_time_step.observation,
                'latent_state': next_latent_state},
            reward=self.reward_scaling * next_time_step.reward)

    def render(self):
        return self._wrapped_env.render()

    def wrap_latent_policy(self, latent_policy: TFPolicy):

        class LatentPolicyWrapper(TFPolicy):

            _latent_policy = latent_policy

            def _distribution(
                    self,
                    time_step: ts.TimeStep,
                    policy_state: types.NestedTensorSpec
            ) -> policy_step.PolicyStep:
                return self._latent_policy._distribution(
                    time_step=time_step._replace(observation=time_step.observation['latent_state']),
                    policy_state=policy_state)

        return LatentPolicyWrapper(
            time_step_spec=self.time_step_spec(),
            action_spec=self.action_spec())
