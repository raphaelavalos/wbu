from typing import Optional, Union, Tuple, Callable

import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tf_agents.typing.types import Float

from belief_learner.networks.base_models import DistributionModel
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.get_model import get_model
from belief_learner.networks.tools import rep_layers, rep_concat_layers, _get_elem


class StateReconstructionNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            decoder_network: tfk.Model,
            state_shape: Union[Tuple[int, ...], tf.TensorShape, Tuple[Tuple[int, ...]], Tuple[tf.TensorShape, ...]],
            time_stacked_states: bool = False,
            post_processing_net: Optional[Tuple[Union[tfk.Model, tfkl.Layer], ...]] = None,
            time_stacked_lstm_units: Optional[int] = None,
            flatten_output: bool = False
    ):

        decoder = decoder_network(latent_state)

        try:
            # output with multiple components
            self.n_dim = len(state_shape[0])
            if flatten_output:
                # enforce flattening the output
                state_shape = [np.sum([np.prod(shape_i) for shape_i in state_shape])]
                self.n_dim = 1
        except TypeError:
            self.n_dim = 1
            state_shape = [state_shape]

        post_processing_net = tf.nest.flatten(post_processing_net)

        outputs = []
        for i, _state_shape in enumerate(state_shape):
            _decoder = _get_elem(post_processing_net, i, default=None)
            if _decoder is None:
                _decoder = decoder
            else:
                post_process_input_shape = _decoder.inputs[0].shape[1:]
                if len(post_process_input_shape) >= 3:
                    _2d_shape = post_process_input_shape[:-1]
                    x = tfkl.Dense(
                        units=np.prod(_2d_shape),
                        name=f'dense_state_{i:d}_unit_normalizer'
                    )(decoder)
                    x = tfkl.Reshape(target_shape=tuple(_2d_shape) + (1,))(x)
                    x = tfkl.Conv2D(
                        filters=post_process_input_shape[-1],
                        kernel_size=3,
                        padding='same',
                        name=f'cnn_state_{i:d}_cnn_unit_normalizer'
                    )(x)
                else:
                    x = tfkl.Dense(
                        units=np.prod(post_process_input_shape),
                    )(decoder)
                    x = tfkl.Reshape(target_shape=post_process_input_shape)(x)
                _decoder = _decoder(x)

            if time_stacked_states and time_stacked_lstm_units is not None:
                decoder_output = tfkl.Flatten()(_decoder)
                time_dimension = _state_shape[0]
                _state_shape = _state_shape[1:]

                if decoder_output.shape[-1] % time_dimension != 0:
                    decoder_output = tfkl.Dense(
                        units=decoder_output.shape[-1] + time_dimension - decoder_output.shape[-1] % time_dimension
                    )(decoder_output)

                decoder_output = tfkl.Reshape(
                    target_shape=(time_dimension, decoder_output.shape[-1] // time_dimension)
                )(decoder_output)
                decoder_output = tfkl.LSTM(
                    units=time_stacked_lstm_units, return_sequences=True
                )(decoder_output)
            else:
                decoder_output = _decoder

            if np.prod(decoder_output.shape[1:]) != np.prod(_state_shape):
                if len(decoder_output.shape[1:]) > 1:
                    decoder_output = tfkl.Flatten()(decoder_output)
                decoder_output = tfkl.Dense(
                    units=np.prod(_state_shape),
                    activation=None,
                    name=f'state_{i:d}_decoder_unit_normalizer'
                )(decoder_output)
            if not flatten_output and decoder_output.shape[1:] != _state_shape:
                decoder_output = tfkl.Reshape(
                    target_shape=_state_shape,
                    name=f'state_{i:d}_decoder_raw_output_reshape'
                )(decoder_output)

            if time_stacked_states:
                decoder_output = tfkl.TimeDistributed(decoder_output)(_decoder)

            outputs.append(decoder_output)

        super(StateReconstructionNetwork, self).__init__(
            inputs=latent_state,
            outputs=outputs,
            name='state_reconstruction_network')
        self.time_stacked_states = time_stacked_states

    def distribution(self, latent_state: Float) -> tfd.Distribution:
        outputs = self(latent_state)
        outputs = tf.nest.flatten(outputs)
        distributions = [
            tfd.Independent(tfd.Deterministic(loc=output))
            if self.time_stacked_states
            else tfd.Deterministic(loc=output)
            for output in outputs
        ]
        if self.n_dim == 1:
            return distributions[0]
        else:
            return tfd.JointDistributionSequential(distributions)

    def get_config(self):
        config = super(StateReconstructionNetwork, self).get_config()
        config.update({"time_stacked_states": self.time_stacked_states})
        return config


class Decoder(tfk.Model):

    def __init__(
            self,
            latent_variable: tfkl.Input,
            decoder_fc_arch: Optional[ModelArchitecture] = None,
            decoder_tcnn_arch: Optional[ModelArchitecture] = None,
            name: Optional[str] = None,
    ):
        decoder_fc = get_model(decoder_fc_arch)
        x = decoder_fc(latent_variable)
        decoder_tcnn = None
        if decoder_tcnn_arch is not None:
            decoder_tcnn = get_model(decoder_tcnn_arch)
            x = decoder_tcnn(x)
        super(Decoder, self).__init__(
            inputs=latent_variable,
            outputs=x,
            name=name)
        self.decoder_fc = decoder_fc
        self.decoder_tcnn = decoder_tcnn


class Encoder(tfk.Model):

    def __init__(
            self,
            latent_variable: tfkl.Input,
            encoder_fc_arch: Optional[ModelArchitecture] = None,
            encoder_cnn_arch: Optional[ModelArchitecture] = None,
            name: Optional[str] = None,
            output_softclip: Optional[Callable[[Float], Float]] = None
    ):
        x = latent_variable
        encoder_cnn = None
        if encoder_cnn_arch is not None:
            encoder_cnn = get_model(encoder_cnn_arch)
            x = encoder_cnn(x)
        encoder_fc = get_model(encoder_fc_arch)
        x = encoder_fc(x)
        if output_softclip is not None:
            x = tfkl.Lambda(output_softclip)(x)
        super(Encoder, self).__init__(
            inputs=latent_variable,
            outputs=x,
            name=name)
        self.encoder_fc = encoder_fc
        self.encoder_cnn = encoder_cnn



class StateObservationNormalDecoderN(DistributionModel):

    def __init__(
            self,
            latent_state_size: int,
            observation_decoder_fc_arch: Optional[ModelArchitecture],
            observation_decoder_tcnn_arch: Optional[ModelArchitecture],
            latent_deembedding_state_arch: ModelArchitecture,
            latent_deembedding_observation_arch: ModelArchitecture,
            state_decoder_fc_arch: ModelArchitecture,
            state_shape: Union[tf.TensorShape, Tuple[int, ...]],
            observation_shape: Union[tf.TensorShape, Tuple[int, ...]],
            atomic_prop_dims: int,
            emb_state_size: int,
            emb_observation_size: int,
            random_decoder: bool = False,
            min_variance: Float = 10. / 255,
            name: Optional[str] = None,
            state_decoder_tcnn_arch: Optional[ModelArchitecture] = None,
    ):
        """
        z is latent_state;
        o is the flattened observation;
        s is the flattened state;
        z -> observation_decoder (possibly a transposed cnn) -> mean(o), covar(o)
        z, mean(o), covar(o) -> state_decoder_network -> mean(s), mean(s')
        Notice that o and s are flattened at the network output;
        """
        # super().__init__(name="State Observation Normal Decoder")
        # self._self_setattr_tracking = False
        latent_state = tfkl.Input((latent_state_size,))
        assert not random_decoder, "Not implemented"
        self._random_decoder = random_decoder
        self._observation_shape = observation_shape
        self._state_shape = state_shape

        if latent_deembedding_state_arch is not None:
            latent_deembedding_state_net = get_model(latent_deembedding_state_arch)
        else:
            latent_deembedding_state_net = tfkl.Lambda(
                lambda x: x[:, atomic_prop_dims: atomic_prop_dims + emb_state_size])

        state_decoder_fc_net = get_model(state_decoder_fc_arch)
        if state_decoder_tcnn_arch is not None:
            state_decoder_tcnn_net = get_model(state_decoder_tcnn_arch)
        else:
            state_decoder_tcnn_net = tfkl.Lambda(lambda x: x)
        embedding_state = latent_deembedding_state_net(latent_state)
        state_decoded_raw = state_decoder_fc_net(embedding_state)
        state_decoded = state_decoder_tcnn_net(state_decoded_raw)

        if latent_deembedding_observation_arch is not None:
            latent_deembedding_observation_net = get_model(latent_deembedding_observation_arch)
        else:
            latent_deembedding_observation_net = tfkl.Lambda(lambda x: x[:,:-emb_observation_size])
        embedding_observation = latent_deembedding_observation_net(latent_state)
        observation_decoder_fc_net = None
        if observation_decoder_fc_arch is not None:
            observation_decoder_fc_net = get_model(observation_decoder_fc_arch)
            embedding_observation = observation_decoder_fc_net(embedding_observation)
        observation_decoder_tcnn_net = None
        if observation_decoder_tcnn_arch is not None:
            observation_decoder_tcnn_net = get_model(observation_decoder_tcnn_arch)
            embedding_observation = observation_decoder_tcnn_net(embedding_observation)
        observation_decoded = embedding_observation

        super(StateObservationNormalDecoderN, self).__init__(
            inputs=latent_state,
            outputs=[state_decoded, observation_decoded],
            name=name)
        self.latent_deembedding_state_net = latent_deembedding_state_net
        self.state_decoder_fc_net = state_decoder_fc_net
        self.state_decoder_tcnn_net = state_decoder_tcnn_net
        self.latent_deembedding_observation_net = latent_deembedding_observation_net
        self.observation_decoder_fc_net = observation_decoder_fc_net
        self.observation_decoder_tcnn_net = observation_decoder_tcnn_net
        self._state_shape = state_shape
        self._observation_shape = observation_shape

    def distribution(self, latent_state: Float, training: bool = False) -> tfd.Distribution:
        decoded_state, decoded_obs = self(latent_state, training)
        flat_observation = tf.reshape(decoded_obs, (decoded_obs.shape[0], -1))
        flat_state = tf.reshape(decoded_state, (decoded_state.shape[0], -1))

        state_distribution = tfd.Independent(
            tfd.Deterministic(loc=flat_state),
            reinterpreted_batch_ndims=1)
        observation_distribution = tfd.Independent(
            tfd.Deterministic(loc=flat_observation),
            reinterpreted_batch_ndims=1)

        return tfd.JointDistributionSequential([
            tfd.TransformedDistribution(
                distribution=state_distribution,
                bijector=tfb.Reshape(self._state_shape)),
            tfd.TransformedDistribution(
                distribution=observation_distribution,
                bijector=tfb.Reshape(self._observation_shape)),
        ])

    def observation_distribution(self, latent_state: Float, variance: Optional[Float] = 1e-1, training: bool = False) -> tfd.Distribution:
        observation_embedding = self.latent_deembedding_observation_net(latent_state, training)
        if self.observation_decoder_fc_net is not None:
            observation_embedding = self.observation_decoder_fc_net(observation_embedding, training)
        if self.observation_decoder_tcnn_net is not None:
            observation_embedding = self.observation_decoder_tcnn_net(observation_embedding, training)
        decoded_obs = observation_embedding
        flat_observation_mean = tf.reshape(decoded_obs, (decoded_obs.shape[0], -1))
        flat_observation_scale = tf.ones_like(flat_observation_mean) * tf.sqrt(variance)

        return tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(
                loc=flat_observation_mean,
                scale_diag=flat_observation_scale, ),
            bijector=tfb.Reshape(self._observation_shape))


class ObservationNormalDecoder(StateObservationNormalDecoderN):

    def __init__(
            self,
            latent_state_size: int,
            observation_decoder_fc_arch: Optional[ModelArchitecture],
            observation_decoder_tcnn_arch: Optional[ModelArchitecture],
            observation_shape: Union[tf.TensorShape, Tuple[int, ...]],
            latent_deembedding_observation_arch: Optional[ModelArchitecture],
            emb_state_size: int,
            emb_observation_size: int,
            name: Optional[str] = None,
    ):
        super(ObservationNormalDecoder, self).__init__(
            latent_state_size=latent_state_size,
            observation_decoder_fc_arch=observation_decoder_fc_arch,
            observation_decoder_tcnn_arch=observation_decoder_tcnn_arch,
            observation_shape=observation_shape,
            latent_deembedding_observation_arch=latent_deembedding_observation_arch.replace(raw_last=False),
            latent_deembedding_state_arch=ModelArchitecture(hidden_units=(0, ), output_dim=(0, ), activation='linear', name='obs_net_state_part_removal_0'),
            state_decoder_fc_arch=ModelArchitecture(hidden_units=(0, ), output_dim=(0, ), activation='linear', name='obs_net_state_part_removal_1'),
            state_shape=(0, ),
            name=name,
            atomic_prop_dims=0,
            emb_state_size=emb_state_size,
            emb_observation_size=emb_observation_size)

    def call(self, inputs, training=None, mask=None, *args, **kwargs):
        _, observation_decoded = super(ObservationNormalDecoder, self).call(
            inputs, training=training, mask=mask, *args, **kwargs)
        return observation_decoded


class ActionReconstructionNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            latent_action: tfkl.Input,
            action_decoder_network: tfk.Model,
            action_shape: Union[Tuple[int, ...], tf.TensorShape],
    ):
        action_reconstruction_network = tfkl.Concatenate(name='action_reconstruction_input')([
            latent_state, latent_action])
        action_reconstruction_network = action_decoder_network(action_reconstruction_network)
        action_reconstruction_network = tfkl.Dense(
            units=np.prod(action_shape),
            activation=None,
            name='action_reconstruction_network_raw_output'
        )(action_reconstruction_network)
        action_reconstruction_network = tfkl.Reshape(
            target_shape=action_shape,
            name='action_reconstruction_network_output'
        )(action_reconstruction_network)

        super(ActionReconstructionNetwork, self).__init__(
            inputs=[latent_state, latent_action],
            outputs=action_reconstruction_network,
            name='action_reconstruction_network')

    def distribution(
            self,
            latent_state: tf.Tensor,
            latent_action: tf.Tensor,
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self([latent_state, latent_action]))


class RewardNetwork(DistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input = None,
            latent_action: tfkl.Input = None,
            next_latent_state: tfkl.Input = None,
            reward_network: tfk.Model = None,
            reward_shape: Union[Tuple[int, ...], tf.TensorShape] = None,
    ):
        _reward_network = tfkl.Concatenate(name='reward_function_input')(
            [latent_state, latent_action, next_latent_state])
        _reward_network = reward_network(_reward_network)
        _reward_network = tfkl.Dense(
            units=np.prod(reward_shape),
            activation=None,
            name='reward_network_raw_output'
        )(_reward_network)
        _reward_network = tfkl.Reshape(reward_shape, name='reward')(_reward_network)
        super(RewardNetwork, self).__init__(
            inputs=[latent_state, latent_action, next_latent_state],
            outputs=_reward_network,
            name='reward_network')

    def distribution(
            self,
            latent_state: Float,
            latent_action: Float,
            next_latent_state: Float,
            training: bool = False,
    ) -> tfd.Distribution:
        return tfd.Deterministic(loc=self([latent_state, latent_action, next_latent_state], training=training))

    def __repr__(self):
        name = getattr(self, 'name', self.__class__.__name__)
        layers = rep_concat_layers(self.inputs) + ' --> ' + rep_layers(self.layers[len(self.inputs) + 1:])
        return name + ' : ' + layers
