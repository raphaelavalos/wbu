from typing import Optional, Callable, Union, Tuple
import enum

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from belief_learner.utils import get_logger
from belief_learner.utils.decorators import log_usage
from belief_learner.networks.autoregressive_bernoulli import AutoRegressiveBernoulliNetwork
from belief_learner.networks.base_models import DiscreteDistributionModel
from belief_learner.networks.get_model import get_model
from belief_learner.networks import tools
from belief_learner.networks.conv import Convolutional
from belief_learner.networks.fully_connected import FullyConnected
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.tools import _get_elem, rep_layers

logger = get_logger(__name__)


class EncodingType(enum.Enum):
    INDEPENDENT = enum.auto()
    AUTOREGRESSIVE = enum.auto()
    LSTM = enum.auto()
    DETERMINISTIC = enum.auto()


class ObservationEncoder(FullyConnected):
    def __init__(
            self,
            fc_hidden_params: Tuple[int, ...],
            activation_fn: Union[Callable, str],
            output_shape: Tuple[int, ...],
            raw_last: bool,
            name: Optional[str] = None,
            **kwargs
    ):
        super().__init__(
            ModelArchitecture(
                hidden_units=fc_hidden_params,
                activation=activation_fn,
                output_dim=output_shape,
                raw_last=raw_last,
                name=name,
                **kwargs
            ))

    @staticmethod
    def build_from_architecture(architecture: ModelArchitecture):
        assert architecture.output_dim is not None, "output_dim should be provided"
        return ObservationEncoder(
            fc_hidden_params=architecture.hidden_units,
            activation_fn=tools.get_activation_fn(architecture.activation),
            output_shape=architecture.output_dim,
            raw_last=architecture.raw_last,
            name=architecture.name,
        )


class CNNObservationEncoder(Convolutional):
    def __init__(
            self,
            filters: Tuple[int, ...],
            fc_hidden_params: Tuple[int, ...],
            activation_fn: Union[Callable, str],
            output_shape: Tuple[int, ...],
            raw_last: bool,
            kernel_size: Optional[Union[Tuple[int, ...], int]] = None,
            strides: Optional[Union[Tuple[int, ...], int]] = None,
            padding: Optional[Union[Tuple[str, ...], str]] = None,
            name: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(
            ModelArchitecture(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=activation_fn,
                name=name,
                raw_last=False, )
        )
        assert fc_hidden_params is None or fc_hidden_params == ()
        self._fc_hidden_layers = ObservationEncoder(
            fc_hidden_params=(),
            activation_fn=activation_fn,
            output_shape=output_shape,
            raw_last=raw_last,
        )

    def call(self, inputs, *args, **kwargs):
        x = super(CNNObservationEncoder, self).call(inputs, *args, **kwargs)
        x = self._fc_hidden_layers(x)
        return x

    def __repr__(self):
        return f'{getattr(self, "name", self.__class__.__name__)} : {rep_layers(self._layers + self._fc_hidden_layers._layers)}'

    @staticmethod
    def build_from_architecture(architecture: ModelArchitecture):
        assert architecture.is_cnn, "The provided architecture does not describe a CNN."
        assert architecture.output_dim is not None, "output_dim should be provided"
        return CNNObservationEncoder(
            filters=architecture.filters,
            fc_hidden_params=architecture.hidden_units,
            activation_fn=tools.get_activation_fn(architecture.activation),
            output_shape=architecture.output_dim,
            kernel_size=architecture.kernel_size,
            strides=architecture.strides,
            padding=architecture.padding,
            raw_last=architecture.raw_last,
            name=architecture.name,
        )


PreprocessingNetwork = Union[Tuple[Union[tfk.Model, tfkl.Layer], ...], Union[tfk.Model, tfkl.Layer]]


class LatentStateEncoderNetwork(DiscreteDistributionModel):

    @log_usage
    def __init__(self,
                 inputs: Union[tfkl.Input, Tuple[tfkl.Input, ...]],
                 latent_embedding_arch: ModelArchitecture,
                 latent_state_size: int,
                 atomic_prop_dims: int,
                 output_softclip: Callable[[Float], Float] = tfb.Identity(),
                 pre_proc_net: Optional[PreprocessingNetwork] = None,
                 deterministic_reset: bool = True,
                 ):
        n_logits = (latent_state_size - atomic_prop_dims)
        self.deterministic_reset = deterministic_reset
        if not isinstance(inputs, (list, tuple)): 
            inputs = [inputs]
        assert len(inputs) == len(pre_proc_net)
        x = [pre_proc_net_(input_)
             if pre_proc_net_ is not None
             else input_
             for input_, pre_proc_net_ in zip(inputs, pre_proc_net)]
        if len(x) > 1:
            x = tfkl.Concatenate()(x)
        else:
            x = x[0]

        if latent_embedding_arch is not None:
            latent_embedding_arch = latent_embedding_arch.replace(output_dim=(n_logits,))
            latent_embedding = get_model(latent_embedding_arch)
            x = latent_embedding(x)
        else:
            latent_embedding = tfb.Identity()
        assert x.shape[1] == n_logits
        x = tfkl.Lambda(output_softclip)(x)

        super(LatentStateEncoderNetwork, self).__init__(
            inputs=inputs[0] if len(inputs) == 1 else inputs,
            outputs=x,
            name='state_encoder')
        self._pre_proc_net = pre_proc_net
        self._latent_embedding = latent_embedding
        self._output_softclip = output_softclip

    def _deterministic_distribution(
            self,
            state: Float,
            step_fn: Callable[[Float], Float],
            label: Optional[Float] = None,
            training: bool = False
    ):
        loc = step_fn(self(state, training=training))
        if label is not None:
            loc = tf.concat([label, loc], axis=-1)
        return tfd.Independent(
            tfd.Deterministic(loc=loc),
            reinterpreted_batch_ndims=1)

    def relaxed_distribution(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            training: bool = False,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self._deterministic_distribution(
            state=state,
            # smooth heaviside
            step_fn=lambda x: tf.sigmoid(2. * x / temperature),
            label=label,
            training=training,
        )


    def discrete_distribution(
            self,
            state: Float,
            label: Optional[Float] = None,
            deterministic_reset: bool = True,
            dtype=tf.float32
    ) -> tfd.Distribution:
        return self._deterministic_distribution(
            state=state,
            step_fn=lambda x: tf.cast(x > 0., dtype=self.dtype),
            label=label)

    def get_logits(self, state: Float, *args, **kwargs):
        return (self.relaxed_distribution(state, temperature=1e-1).sample() - .5) * 20.


class StateEncoderNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            state: Union[tfkl.Input, Tuple[tfkl.Input, ...]],
            hidden_units: Tuple[int, ...],
            activation: Callable[[tf.Tensor], tf.Tensor],
            latent_state_size: int,
            atomic_prop_dims: int,
            time_stacked_states: bool = False,
            time_stacked_lstm_units: int = 128,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            pre_proc_net: Optional[PreprocessingNetwork] = None,
            lstm_output: bool = False,
            deterministic_reset: bool = True,
    ):
        n_logits = (latent_state_size - atomic_prop_dims)
        self.deterministic_reset = deterministic_reset

        last_layer_units = hidden_units[-1]
        if time_stacked_states:
            last_layer_units = hidden_units[-1] // n_logits * n_logits
        if last_layer_units != hidden_units[-1]:
            logger.warning(f"{self.__class__.__name__}: replacing last layer unit from {hidden_units[-1]} to "
                           f"{last_layer_units}.")
        hidden_units = hidden_units[:-1]
        state = tf.nest.flatten(state)
        self.no_inputs = len(state)

        if pre_proc_net is not None:
            pre_proc_net = tf.nest.flatten(pre_proc_net)
            assert len(pre_proc_net) == self.no_inputs or len(pre_proc_net) == 1, \
                "the number of pre-processing networks should be one or have the same size as the number of inputs"

        encoders = []
        state_encoder_networks = []
        for i, _input in enumerate(state):
            state_encoder_network = tfk.Sequential(name=f"state_encoder_body_for_input_{i:d}")
            state_encoder_network.add(tfkl.Flatten())
            for units in hidden_units:
                state_encoder_network.add(tfkl.Dense(units, activation))

            if time_stacked_states:
                logger.warning('this was not tested')
                if pre_proc_net is not None:
                    encoder = tfkl.TimeDistributed(_get_elem(pre_proc_net, i))(_input)
                else:
                    encoder = _input
                encoder = tfkl.LSTM(units=time_stacked_lstm_units)(encoder)
                encoders.append(state_encoder_network(encoder))
            else:
                if pre_proc_net is not None:
                    _state = _get_elem(pre_proc_net, i)(_input)
                else:
                    _state = _input
                # _state = tfkl.Flatten()(_state)
                encoders.append(state_encoder_network(_state))
            state_encoder_networks.append(state_encoder_network)

        encoder = tfkl.Concatenate()(encoders)
        encoder = tfkl.Dense(units=last_layer_units, activation=activation)(encoder)

        if lstm_output:
            encoder = tfkl.Reshape(target_shape=(n_logits, last_layer_units // n_logits))(encoder)
            encoder = tfkl.LSTM(units=1, activation=output_softclip, return_sequences=True)(encoder)
            encoder = tfkl.Reshape(target_shape=(latent_state_size - atomic_prop_dims,))(encoder)
        else:
            encoder = tfkl.Dense(
                units=latent_state_size - atomic_prop_dims,
                activation=output_softclip
            )(encoder)

        super(StateEncoderNetwork, self).__init__(
            inputs=state[0] if self.no_inputs == 1 else state,
            outputs=encoder,
            name='state_encoder')
        self.pre_proc_net = pre_proc_net
        self.state_encoder_networks = state_encoder_networks

    def __repr__(self):
        return f"Concatenate([\n" \
               f"\t{rep_layers([self.inputs[0]])} --> {self.pre_proc_net[0]} --> " \
               f"{rep_layers([self.state_encoder_networks[0]])},\n" \
               f"\t{rep_layers([self.inputs[1]])} --> {self.pre_proc_net[1]} --> " \
               f"{rep_layers([self.state_encoder_networks[1]])}\n" \
               f"]) --> {rep_layers(self.layers[-2:])}"

    def relaxed_distribution(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            logistic: bool = True,
    ) -> tfd.Distribution:
        logits = self(state)
        if label is not None and self.deterministic_reset:
            # if the "reset state" flag is set, then enforce mapping the reset state to a single latent state
            logits = tf.pow(logits, 1. - label[..., -1:]) * tf.pow(-10., label[..., -1:])
        if logistic:
            distribution = tfd.TransformedDistribution(
                distribution=tfd.Independent(
                    tfd.Logistic(
                        loc=logits / temperature,
                        scale=tf.pow(temperature, -1.)),
                    reinterpreted_batch_ndims=1, ),
                bijector=tfb.Sigmoid())
        else:
            distribution = tfd.Independent(
                tfd.RelaxedBernoulli(
                    logits=logits,
                    temperature=temperature,
                    allow_nan_stats=False),
                reinterpreted_batch_ndims=1)
        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=label),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, distribution])
        else:
            return distribution

    def discrete_distribution(
            self,
            state: Float,
            label: Optional[Float] = None,
            deterministic_reset: bool = True
    ) -> tfd.Distribution:
        logits = self(state)
        if label is not None and deterministic_reset:
            # if the "reset state" flag is set, then enforce mapping the reset state to a single latent state
            logits = tf.pow(logits, 1. - label[..., -1:]) * tf.pow(-10., label[..., -1:])
        d2 = tfd.Independent(
            tfd.Bernoulli(
                logits=logits,
                dtype=self.dtype),
            reinterpreted_batch_ndims=1)

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=tf.cast(label, dtype=self.dtype)),
                reinterpreted_batch_ndims=1)

            def mode(name='mode', **kwargs):
                return tf.concat([
                    d1.mode(name='label_' + name, **kwargs),
                    d2.mode(name='latent_state_' + name, **kwargs)],
                    axis=-1)

            distribution = tfd.Blockwise([d1, d2])
            distribution.mode = mode
            return distribution
        else:
            return d2

    def get_logits(self, state: Float, *args, **kwargs):
        return self(state)

    def get_config(self):
        config = super(AutoRegressiveStateEncoderNetwork, self).get_config()
        config.update({
            "get_logits": self.get_logits,
        })
        return config


class DeterministicStateEncoderNetwork(StateEncoderNetwork):

    @log_usage
    def __init__(
            self,
            state: tfkl.Input,
            hidden_units: Tuple[int, ...],
            activation: Callable[[tf.Tensor], tf.Tensor],
            latent_state_size: int,
            atomic_prop_dims: int,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            pre_proc_net: Optional[tfk.Model] = None,
    ):
        super().__init__(
            state=state,
            activation=activation,
            hidden_units=hidden_units,
            latent_state_size=latent_state_size,
            atomic_prop_dims=atomic_prop_dims,
            time_stacked_states=time_stacked_states,
            lstm_output=False,
            output_softclip=output_softclip,
            pre_proc_net=pre_proc_net)

    def _deterministic_distribution(
            self,
            state: Float,
            step_fn: Callable[[Float], Float],
            label: Optional[Float] = None
    ):
        loc = step_fn(self(state))
        if label is not None:
            loc = tf.concat([label, loc], axis=-1)
        return tfd.Independent(
            tfd.Deterministic(loc=loc),
            reinterpreted_batch_ndims=1)

    def relaxed_distribution(
            self,
            state: Float,
            temperature: Float,
            label: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        return self._deterministic_distribution(
            state=state,
            # smooth heaviside
            step_fn=lambda x: tf.sigmoid(2. * x / temperature),
            label=label)

    def discrete_distribution(
            self,
            state: Float,
            label: Optional[Float] = None,
            deterministic_reset: bool = True,
            dtype=tf.float32
    ) -> tfd.Distribution:
        return self._deterministic_distribution(
            state=state,
            step_fn=lambda x: tf.cast(x > 0., dtype=self.dtype),
            label=label)

    def get_logits(self, state: Float, *args, **kwargs):
        return (self.relaxed_distribution(state, temperature=1e-1).sample() - .5) * 20.


class AutoRegressiveStateEncoderNetwork(AutoRegressiveBernoulliNetwork):
    def __init__(
            self,
            state_shape: Union[tf.TensorShape, Tuple[int, ...]],
            activation: Union[str, Callable[[Float], Float]],
            hidden_units: Tuple[int, ...],
            latent_state_size: int,
            atomic_prop_dims: int,
            temperature: Float,
            time_stacked_states: bool = False,
            time_stacked_lstm_units: int = 128,
            output_softclip: Callable[[Float], Float] = tfb.Identity(),
            pre_proc_net: Optional[tfk.Model] = None,
            deterministic_reset: bool = True,
    ):
        super(AutoRegressiveStateEncoderNetwork, self).__init__(
            event_shape=(latent_state_size - atomic_prop_dims,),
            activation=activation,
            hidden_units=hidden_units,
            conditional_event_shape=state_shape,
            temperature=temperature,
            output_softclip=output_softclip,
            time_stacked_input=time_stacked_states,
            time_stacked_lstm_units=time_stacked_lstm_units,
            pre_proc_net=pre_proc_net,
            name='autoregressive_state_encoder')
        self._atomic_prop_dims = atomic_prop_dims
        self.deterministic_reset = deterministic_reset

    def relaxed_distribution(
            self,
            state: Optional[Float] = None,
            label: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if state is None:
            raise ValueError("a state to encode should be provided.")

        distribution = super(
            AutoRegressiveStateEncoderNetwork, self
        ).relaxed_distribution(conditional_input=state)

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=label),
                reinterpreted_batch_ndims=1)
            return tfd.Blockwise([d1, distribution])
        else:
            return distribution

    def discrete_distribution(
            self,
            state: Optional[Float] = None,
            label: Optional[Float] = None,
            *args, **kwargs
    ) -> tfd.Distribution:
        if state is None:
            raise ValueError("a state to encode should be provided.")

        d2 = super(
            AutoRegressiveStateEncoderNetwork, self
        ).discrete_distribution(conditional_input=state)

        def mode(name='mode', **kwargs):
            def d2_distribution_fn_mode(x: Optional[Float] = None):
                d = d2.distribution_fn(x)

                def call_mode_n(*args, **kwargs):
                    mode = d.mode(**kwargs)
                    return mode

                d._call_sample_n = call_mode_n
                return d

            return tfd.Autoregressive(
                distribution_fn=d2_distribution_fn_mode,
            ).sample(sample_shape=tf.shape(state)[:-1], name=name, **kwargs)

        d2.mode = mode

        if label is not None:
            d1 = tfd.Independent(
                tfd.Deterministic(loc=tf.cast(label, dtype=self.dtype)),
                reinterpreted_batch_ndims=1)

            def mode(name='mode', **kwargs):
                return tf.concat([
                    d1.mode(name='label_' + name, **kwargs),
                    d2.mode(name='latent_state_' + name, **kwargs)],
                    axis=-1)

            def sample(sample_shape=(), seed=None, name='sample', **kwargs):
                return tf.concat([
                    d1.sample(sample_shape, seed=seed, name='label_' + name, **kwargs),
                    d2.sample(sample_shape, seed=seed, name='latent_state_' + name, **kwargs)],
                    axis=-1)

            def prob(latent_state, name='prob', **kwargs):
                return tfd.Blockwise([d1, d2]).prob(latent_state, name=name, **kwargs)

            # dirty Blockwise; do not trigger any warning
            distribution = tfd.TransformedDistribution(d1, bijector=tfb.Identity())
            distribution.mode = mode
            distribution.sample = sample
            distribution.prob = prob
            return distribution
        else:
            return d2

    def get_logits(
            self,
            state: Float,
            latent_state: Float,
            include_label: bool = True,
            *args, **kwargs
    ) -> Float:
        if include_label:
            latent_state = latent_state[..., self._atomic_prop_dims:]
        if self.pre_process_input:
            state = self._preprocess_fn(state)
        return self._output_softclip(self._made(latent_state, conditional_input=state)[..., 0])

    def get_config(self):
        config = super(AutoRegressiveStateEncoderNetwork, self).get_config()
        config.update({
            "_atomic_prop_dims": self._atomic_prop_dims,
            "get_logits": self.get_logits,
        })
        return config


class ActionEncoderNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            latent_state: tfk.Input,
            action: tfk.Input,
            number_of_discrete_actions: int,
            action_encoder_network: tfk.Model,
    ):
        action_encoder = tfkl.Concatenate(name='action_encoder_input')(
            [latent_state, action])
        action_encoder = action_encoder_network(action_encoder)
        action_encoder = tfkl.Dense(
            units=number_of_discrete_actions,
            activation=None,
            name='action_encoder_categorical_logits'
        )(action_encoder)

        super(ActionEncoderNetwork, self).__init__(
            inputs=[latent_state, action],
            outputs=action_encoder,
            name="action_encoder")

    def relaxed_distribution(
            self,
            latent_state: Float,
            action: Float,
            temperature: Float,
    ) -> tfd.Distribution:
        return tfd.RelaxedOneHotCategorical(
            logits=self([latent_state, action]),
            temperature=temperature,
            allow_nan_stats=False)

    def discrete_distribution(
            self,
            latent_state: Float,
            action: Float,
    ) -> tfd.Distribution:
        return tfd.OneHotCategorical(logits=self([latent_state, action]), allow_nan_stats=False)

    def get_config(self):
        config = super(ActionEncoderNetwork, self).get_config()
        return config
