import tensorflow as tf
from tf_agents.metrics.tf_metrics import AverageReturnMetric
from tf_agents.typing.types import Float
from tf_agents.utils import common


class AverageDiscountedReturnMetric(AverageReturnMetric):
    """
    Metric to compute the average discounted return.
    """

    def __init__(
            self,
            gamma: Float = 0.99,  # discount factor
            name="AverageDiscountedReturn",
            dtype=tf.float32,
            batch_size=1,
            buffer_size=10,
            reward_scale: float = 1.,
    ):
        super(AverageDiscountedReturnMetric, self).__init__(name=name, dtype=dtype, batch_size=batch_size,
                                                            buffer_size=buffer_size)
        self.gamma = gamma
        self._step_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='StepAccumulator')
        self.reward_scale = reward_scale

    @common.function(autograph=True)
    def call(self, trajectory):
        # Zero out batch indices where a new episode is starting.
        self._return_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                     self._return_accumulator))
        self._step_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._step_accumulator),
                     self._step_accumulator))

        # Update accumulator with received rewards.
        self._return_accumulator.assign_add(
            tf.pow(self.gamma, self._step_accumulator) * trajectory.reward * self.reward_scale)
        self._step_accumulator.assign_add(tf.ones_like(self._step_accumulator))

        # Add final returns to buffer.
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
        for indx in last_episode_indices:
            self._buffer.add(self._return_accumulator[indx])

        return trajectory
