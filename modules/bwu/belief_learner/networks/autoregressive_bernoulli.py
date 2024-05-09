from typing import Union, Tuple, Callable, Optional

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.layers as tfpl
from tf_agents.typing.types import Float

from belief_learner.utils.decorators import log_usage
from belief_learner.networks.base_models import DiscreteDistributionModel

from belief_learner.utils import get_logger
logger = get_logger(__name__)


class UnstableSigmoid(tfb.Sigmoid):
    def _forward(self, x):
        if self._is_standard_sigmoid:
            return tf.math.sigmoid(x)
        else:
            return super(UnstableSigmoid, self)._forward(x)

def relaxed_distribution(
        made: tfb.AutoregressiveNetwork,
        output_softclip: Callable[[Float], Float],
        temperature: Float,
) -> tfd.TransformedDistribution:
    event_shape = made._event_shape

    def bijector_fn(x, conditional_input: Optional[Float] = None) -> tfb.Bijector:
        shift = output_softclip(
            made(x, conditional_input=conditional_input)[..., 0]
        ) / temperature
        return tfb.Chain([UnstableSigmoid(), tfb.Shift(shift)])

    maf = tfb.MaskedAutoregressiveFlow(
        bijector_fn=bijector_fn,
        is_constant_jacobian=True)
    maf._made_variables = made.variables

    return tfd.TransformedDistribution(
        distribution=tfd.Independent(
            tfd.Logistic(
                loc=tf.zeros(event_shape),
                scale=tf.pow(temperature, -1)),
            reinterpreted_batch_ndims=1),
        bijector=maf)


def discrete_distribution(
        made: tfb.AutoregressiveNetwork,
        output_softclip: Callable[[Float], Float],
        conditional_input: Optional[Float] = None,
        dtype=tf.float32,
) -> tfd.Autoregressive:
    event_shape = made._event_shape

    if conditional_input is None:
        sample0 = tf.zeros(shape=event_shape, dtype=dtype)
    else:
        conditional_input = tf.cast(conditional_input, dtype)
        sample0 = tf.zeros(tf.concat([tf.shape(conditional_input)[:-1], event_shape], axis=0), dtype=dtype)

    def distribution_fn(x: Optional[Float] = None):
        if x is None:
            distribution = tfd.Independent(
                tfd.Deterministic(loc=sample0),
                reinterpreted_batch_ndims=1)
        else:
            logits = output_softclip(
                tf.unstack(made(x, conditional_input=conditional_input), axis=-1)[0])
            distribution = tfd.Independent(
                distribution=tfd.Bernoulli(logits=logits, dtype=dtype),
                reinterpreted_batch_ndims=1)

        return distribution

    return tfd.Autoregressive(distribution_fn, sample0=sample0)


class AutoregressiveTransform(tfpl.DistributionLambda):
    def __init__(
            self,
            made: tfb.AutoregressiveNetwork,
            temperature: Optional[Float] = None,
            output_softclip: Optional[Callable[[Float], Float]] = None,
            **kwargs
    ):
        if temperature is None:
            temperature = 1e-5
        if output_softclip is None:
            output_softclip = tfb.Identity()

        super(AutoregressiveTransform, self).__init__(self._transform, **kwargs)
        self._made = made
        self._temperature = temperature
        self._output_softclip = output_softclip
        self._maf = tfb.MaskedAutoregressiveFlow(
            bijector_fn=self.bijector_fn,
            is_constant_jacobian=True, )
        self._maf._made_variables = self._made.variables

    def bijector_fn(self, x, conditional_input: Optional[Float] = None) -> tfb.Bijector:
        shift = self._output_softclip(
            self._made(x, conditional_input=conditional_input)[..., 0]
        ) / self._temperature
        return tfb.Chain([UnstableSigmoid(), tfb.Shift(shift)])

    def build(self, input_shape):
        if self._made._conditional:
            inputs = tfk.Input(input_shape[0][1:], dtype=self.dtype)
            conditional_input = tfk.Input(input_shape[1][1:], dtype=self.dtype)
            outputs = self._made(inputs, conditional_input=conditional_input)[..., 0]
            outputs = tfkl.Lambda(lambda x: self._output_softclip(x) / self._temperature)(outputs)
            tfk.Model(inputs=[inputs, conditional_input], outputs=outputs)
        else:
            tfk.Sequential([
                tfkl.InputLayer(
                    input_shape=input_shape[1:], dtype=self.dtype),
                self._made,
                tfkl.Lambda(lambda x: self._output_softclip(x[..., 0]) / self._temperature)
            ])
        super(AutoregressiveTransform, self).build(input_shape)

    def _transform(self, previous_outputs):
        if self._made._conditional:
            distribution, conditional_input = previous_outputs
            return ConditionalTransformedDistribution(
                tfd.TransformedDistribution(
                    bijector=self._maf,
                    distribution=distribution),
                conditional=conditional_input)
        else:
            distribution, conditional_input = previous_outputs, None
            return tfd.TransformedDistribution(
                bijector=self._maf,
                distribution=distribution)

    def get_config(self):
        config = super(AutoregressiveTransform, self).get_config()
        config.update({
            "bijector_fn": self.bijector_fn,
            "_temperature": self._temperature,
            "_output_softclip": self._output_softclip,
            "_maf": self._maf,
        })
        return config


class AutoRegressiveBernoulliNetwork(DiscreteDistributionModel):

    @log_usage
    def __init__(
            self,
            event_shape: Union[tf.TensorShape, Tuple[int, ...]],
            activation: Callable[[Float], Float],
            hidden_units: Tuple[int, ...],
            conditional_event_shape: Optional[Union[tf.TensorShape, Tuple[int, ...]]] = None,
            temperature: Optional[Float] = 1e-5,
            output_softclip: Optional[Callable[[Float], Float]] = tfb.Identity(),
            dtype: tf.dtypes = tf.float32,
            name: Optional[str] = None,
            made_name: Optional[str] = None,
            time_stacked_input: bool = False,
            time_stacked_lstm_units: int = 128,
            pre_proc_net: Optional[tfk.Model] = None,
    ):
        conditional = conditional_event_shape is not None
        inputs = None
        x = None
        self._preprocess_fn = None

        if not conditional:
            conditional_event_shape = (0,)
        elif time_stacked_input:
            x = tfk.Input(shape=conditional_event_shape)
            inputs = [x]
            if pre_proc_net is not None:
                x = tfkl.TimeDistributed(pre_proc_net)(x)
            x = tfkl.LSTM(units=time_stacked_lstm_units)(x)
            conditional_event_shape = (time_stacked_lstm_units,)
            self._preprocess_fn = tfk.Model(inputs=inputs, outputs=x, name="autoregressive_input_preprocessor")

        logistic_distribution_layer = tfk.Sequential([
            tfkl.InputLayer(input_shape=conditional_event_shape, dtype=dtype, name="logistic_layer_input"),
            tfpl.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Logistic(
                        loc=tf.zeros(tf.concat([tf.shape(t)[:-1], event_shape], axis=0)),
                        scale=tf.pow(temperature, -1), ),
                    reinterpreted_batch_ndims=1, )),
        ], name="sequential_logistic_distribution_layer")

        if inputs == x is None:
            inputs = logistic_distribution_layer.inputs
            x = inputs

        made = tfb.AutoregressiveNetwork(
            params=1,
            hidden_units=list(hidden_units),
            event_shape=event_shape if conditional else None,
            conditional=conditional,
            conditional_event_shape=conditional_event_shape if conditional else None,
            activation=activation,
            name=made_name)

        x = logistic_distribution_layer(x)
        x = AutoregressiveTransform(
            made=made,
            temperature=temperature,
            output_softclip=output_softclip,
            name="autoregressive_transform",
        )([x] + inputs if conditional else x)

        super(AutoRegressiveBernoulliNetwork, self).__init__(
            inputs=inputs,
            outputs=x,
            name=name)

        self.event_shape = event_shape
        self._temperature = temperature
        self._output_softclip = output_softclip
        self._logistic_distribution_network = logistic_distribution_layer
        self._made = made

    @property
    def conditional(self) -> bool:
        return self._made._conditional

    @property
    def pre_process_input(self) -> bool:
        return self._preprocess_fn is not None

    def relaxed_distribution(
            self,
            conditional_input: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        """
        Construct a distribution whose parameters are produced by a Masked Autoregressive Flow.
        More specifically, the Flow uses the internal masked autoregressive network to infer a location of a logistic
        distribution at each event step. This allows (via a chain of reparameterization) to generate logistic samples
        followed by a sigmoid at each time step, in order to generate (dependent) samples of relaxed Bernoulli.
        """
        if self.conditional:
            if conditional_input is None:
                raise ValueError("You must provide a conditional event.")
            distribution = self(conditional_input, *args, **kwargs)
        else:
            distribution = self(tf.zeros((0,)), *args, **kwargs)

        def prob(value, name='prob', **kwargs):
            log_prob = distribution.log_prob(value, name=name, **kwargs)
            prob = tf.exp(log_prob)
            if tf.reduce_any(tf.math.is_inf(prob)):
                tf.print(f"{self.__class__.name} relaxed distribution prob got inf -- log prob was {log_prob}")
            if tf.reduce_any(tf.math.is_nan(prob)):
                tf.print(f"{self.__class__.name} relaxed distribution prob got NaN -- log prob was {log_prob}")
            return prob

        distribution.prob = prob

        return distribution

    def discrete_distribution(
            self,
            conditional_input: Optional[Float] = None,
    ) -> tfd.Distribution:
        return discrete_distribution(
            self._made,
            self._output_softclip,
            conditional_input,
            dtype=self.dtype, )

    def get_config(self):
        config = super(AutoRegressiveBernoulliNetwork, self).get_config()
        config.update({
            "event_shape": self.event_shape,
            "_temperature": self._temperature,
            "_output_softclip": self._output_softclip,
            "conditional": self.conditional,
            "_preprocess_fn": self._preprocess_fn,
            "preprocess_input": self.pre_process_input})
        return config


class ConditionalTransformedDistribution(tfd.Distribution):

    def __init__(
            self,
            masked_autoregressive_flow_transformed_distribution: tfd.TransformedDistribution,
            conditional: Float,
            conditional_kwarg='conditional_input'
    ):
        super().__init__(
            masked_autoregressive_flow_transformed_distribution.dtype,
            masked_autoregressive_flow_transformed_distribution.reparameterization_type,
            masked_autoregressive_flow_transformed_distribution.validate_args,
            masked_autoregressive_flow_transformed_distribution.allow_nan_stats)
        self._wrapped_distribution: tfd.TransformedDistribution = masked_autoregressive_flow_transformed_distribution
        self._conditional = conditional
        self._conditional_kwarg = conditional_kwarg

    @property
    def distribution(self):
        """Base distribution, p(x)."""
        return self._wrapped_distribution.distribution

    @property
    def bijector(self):
        """Function transforming x => y."""
        return self._wrapped_distribution.bijector

    def _event_shape_tensor(self):
        return self._wrapped_distribution._event_shape_tensor()

    def _event_shape(self):
        # Since the `bijector` may change the `event_shape`, we then forward what we
        # know to the bijector. This allows the `bijector` to have final say in the
        # `event_shape`.
        return self._wrapped_distribution._event_shape()

    def _batch_shape_tensor(self):
        return self._wrapped_distribution._batch_shape_tensor()

    def _batch_shape(self):
        return self._wrapped_distribution._batch_shape()

    def _maybe_broadcast_distribution_batch_shape(self):
        return self._wrapped_distribution._maybe_broadcast_distribution_batch_shape()

    def _call_sample_n(self, sample_shape, seed, **kwargs):
        return self._wrapped_distribution._call_sample_n(
            sample_shape=sample_shape,
            seed=seed,
            bijector_kwargs={self._conditional_kwarg: self._conditional},
            **kwargs)

    def _sample_and_log_prob(self, sample_shape, seed, **kwargs):
        return self._wrapped_distribution._sample_and_log_prob(
            sample_shape, seed, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _log_prob(self, y, **kwargs):
        return self._wrapped_distribution._log_prob(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _prob(self, y, **kwargs):
        return self._wrapped_distribution._prob(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _log_cdf(self, y, **kwargs):
        return self._wrapped_distribution._log_cdf(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _cdf(self, y, **kwargs):
        return self._wrapped_distribution._cdf(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _log_survival_function(self, y, **kwargs):
        return self._wrapped_distribution._log_survival_function(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _survival_function(self, y, **kwargs):
        return self._wrapped_distribution._survival_function(
            y, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _quantile(self, value, **kwargs):
        return self._wrapped_distribution._quantile(
            value, bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _mode(self, **kwargs):
        return self._wrapped_distribution._mode(bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _mean(self, **kwargs):
        return self._wrapped_distribution._mean(bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _stddev(self, **kwargs):
        return self._wrapped_distribution._stddev(
            bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    def _entropy(self, **kwargs):
        return self._wrapped_distribution._entropy(
            bijector_kwargs={self._conditional_kwarg: self._conditional}, **kwargs)

    # pylint: disable=not-callable
    def _default_event_space_bijector(self):
        return self._wrapped_distribution._default_event_space_bijector()
