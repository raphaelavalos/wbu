from typing import Callable

import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.typing import types
import tensorflow_probability as tfp
from tf_agents.trajectories.policy_step import PolicyStep
# from util.io import dataset_generator

tfd = tfp.distributions


class LatentPolicyOverRealStateSpace(tf_policy.TFPolicy):

    def __init__(self,
                 time_step_spec,
                 labeling_function: Callable[[tf.Tensor], tf.Tensor],
                 latent_policy: tf_policy.TFPolicy,
                 state_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=latent_policy.action_spec,
            info_spec=latent_policy.info_spec,
            policy_state_spec=latent_policy.policy_state_spec)
        self._labeling_function = labeling_function
        self.wrapped_policy = latent_policy
        self.state_embedding_function = state_embedding_function
        self.labeling_function = dataset_generator.ergodic_batched_labeling_function(labeling_function)

    def _distribution(self, time_step, policy_state):
        latent_state = self.state_embedding_function(
            time_step.observation, self.labeling_function(time_step.observation))
        return self.wrapped_policy._distribution(time_step._replace(observation=latent_state), policy_state)


class LatentPolicyOverRealStateAndActionSpaces(tf_policy.TFPolicy):

    def __init__(self,
                 time_step_spec,
                 action_spec: types.NestedTensorSpec,
                 labeling_function: Callable[[tf.Tensor], tf.Tensor],
                 latent_policy: tf_policy.TFPolicy,
                 state_embedding_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 action_embedding_function: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]):
        super().__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=latent_policy.info_spec,
            policy_state_spec=latent_policy.policy_state_spec)
        self._labeling_function = labeling_function
        self.wrapped_policy = latent_policy
        self.state_embedding_function = state_embedding_function
        self.labeling_function = dataset_generator.ergodic_batched_labeling_function(labeling_function)
        self.action_embedding_function = action_embedding_function

    def _distribution(self, time_step, policy_state):
        label = self.labeling_function(time_step.observation)
        latent_state = self.state_embedding_function(time_step.observation, label)
        latent_action = self.wrapped_policy._distribution(
            time_step._replace(observation=latent_state),
            policy_state
        ).action.sample()
        action = tf.cast(
            self.action_embedding_function(latent_state, latent_action),
            dtype=self.action_spec.dtype)
        return PolicyStep(action=tfd.Deterministic(action), state=policy_state, info=())
