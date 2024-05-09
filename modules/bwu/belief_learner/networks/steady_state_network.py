from typing import Optional, Callable, Tuple

import tensorflow as tf
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from belief_learner.networks.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork


class SteadyStateNetwork(AutoRegressiveBernoulliNetwork):

    def __init__(
            self,
            atomic_prop_dims: int,
            latent_state_size: int,
            activation: Callable[[Float], Float],
            hidden_units: Tuple[int, ...],
            trainable_prior: bool = False,
            temperature: Optional[Float] = 1e-5,
            output_softclip: Optional[Callable[[Float], Float]] = tfb.Identity(),
            dtype: tf.dtypes = tf.float32,
            name: Optional[str] = None,
            made_name: Optional[str] = None,
    ):
        super(SteadyStateNetwork, self).__init__(
            event_shape=(latent_state_size,) if trainable_prior else (atomic_prop_dims,),
            activation=activation,
            hidden_units=hidden_units,
            temperature=temperature,
            output_softclip=output_softclip,
            dtype=dtype,
            name=name,
            made_name=made_name)

        self.trainable_prior = trainable_prior
        if self.trainable_prior:
            logits = tf.zeros(shape=(0, ), dtype=self.dtype)
        else:
            logits = tf.zeros(shape=(latent_state_size - atomic_prop_dims,), dtype=self.dtype)
        self.prior_variables = tf.Variable(
            initial_value=logits,
            trainable=False,
            name='prior_logits',
            dtype=self.dtype)

    def relaxed_distribution(
            self,
            *args,  **kwargs
    ) -> tfd.Distribution:
        d1 = super(SteadyStateNetwork, self).relaxed_distribution(*args, **kwargs)
        if self.trainable_prior:
            return d1
        else:
            d2 = tfd.Independent(
                tfd.TransformedDistribution(
                    distribution=tfd.Logistic(
                        loc=self.prior_variables / self._temperature,
                        scale=tf.pow(self._temperature, -1.), ),
                    bijector=tfb.Sigmoid()),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, d2])

    def discrete_distribution(
            self,
            *args, **kwargs
    ) -> tfd.Distribution:
        d1 = super(SteadyStateNetwork, self).discrete_distribution(*args, **kwargs)
        if self.trainable_prior:
            return d1
        else:
            d2 = tfd.Independent(
                tfd.Bernoulli(logits=self.prior_variables, dtype=self.dtype),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, d2])

    def get_config(self):
        config = super(SteadyStateNetwork, self).get_config()
        config.update({
            "trainable_prior": self.trainable_prior
        })
