from tf_agents import specs
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types


class TimeStackedStatesPolicyWrapper(TFPolicy):

    def __init__(self, tf_policy: TFPolicy, history_length: int):
        observation_spec = specs.BoundedTensorSpec(
            shape=(history_length,) + tf_policy.time_step_spec.observation.shape,
            dtype=tf_policy.time_step_spec.observation.dtype,
            name=tf_policy.time_step_spec.observation.name,
            minimum=tf_policy.time_step_spec.observation.minimum,
            maximum=tf_policy.time_step_spec.observation.maximum)
        time_step_spec = tf_policy.time_step_spec._replace(observation=observation_spec)
        super().__init__(time_step_spec, tf_policy.action_spec)
        self.wrapped_tf_policy = tf_policy

    def _distribution(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        _time_step = time_step._replace(observation=time_step.observation[:, -1, ...])
        return self.wrapped_tf_policy._distribution(_time_step, policy_state)

    def _get_initial_state(self, batch_size: int) -> types.NestedTensor:
        return self.wrapped_tf_policy._get_initial_state(batch_size)

    def _variables(self):
        return self.wrapped_tf_policy._variables()