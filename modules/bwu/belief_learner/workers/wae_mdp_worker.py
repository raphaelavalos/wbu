import os.path
from typing import Tuple, Callable, Optional, Any

import gym.spaces
import numpy as np
from tf_agents.typing.types import Float

from belief_learner import STATE, OBS
from belief_learner.utils.decorators import log_usage
from belief_learner.wae.mdp.wae_bmdp import WassersteinBeliefMDP
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.get_model import get_model
from belief_learner.workers.worker import Worker
from belief_learner.wae.mdp.wasserstein_mdp import WassersteinRegularizerScaleFactor
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

from belief_learner.utils import get_logger, dict_array_to_dict_np

logger = get_logger(__name__)


class WasserteinMDPWorker(Worker):
    __name__ = "WAE Worker"

    @log_usage
    def __init__(
            self,
            state_shape: Tuple[int, ...],
            observation_shape: Tuple[int, ...],
            action_shape: Tuple[int, ...],
            reward_shape: Tuple[int, ...],
            latent_state_size: int,
            emb_observation_size: int,
            emb_state_size: int,
            n_updates: int,
            state_encoder_fc_arch: ModelArchitecture,
            state_decoder_fc_arch: ModelArchitecture,
            observation_encoder_fc_arch: ModelArchitecture,
            observation_decoder_fc_arch: ModelArchitecture,
            latent_embedding_arch: ModelArchitecture,
            transition_arch: ModelArchitecture,
            reward_arch: ModelArchitecture,
            steady_state_lipschitz_arch: ModelArchitecture,
            transition_loss_lipschitz_arch: ModelArchitecture,
            optimizer_name: str,
            minimizer_lr: float,
            maximizer_lr: float,
            # replay_buffer: ReplayBuffer,
            observation_encoder_cnn_arch: Optional[ModelArchitecture] = None,
            state_encoder_cnn_arch: Optional[ModelArchitecture] = None,
            observation_decoder_tcnn_arch: Optional[ModelArchitecture] = None,
            state_decoder_tcnn_arch: Optional[ModelArchitecture] = None,
            clip_by_global_norm: Optional[float] = None,
            latent_policy_arch: Optional[ModelArchitecture] = None,
            latent_stationary_arch: Optional[ModelArchitecture] = None,
            wae_mdp_step: tf.Variable = tf.Variable(
                0, trainable=False, dtype=tf.int64, name='wae_mdp_step'),
            state_encoder_temperature: float = 2. / 3,
            state_prior_temperature: float = 1. / 2,
            wasserstein_regularizer_scale_factor: WassersteinRegularizerScaleFactor = WassersteinRegularizerScaleFactor(
                global_scaling=1.,
                global_gradient_penalty_multiplier=1.),
            n_critic: int = 5,
            use_wae_gan: bool = False,
            wae_gan_regularizer_scale_factor: Optional[float] = 1.,
            wae_gan_discriminator_arch: Optional[ModelArchitecture] = None,
            latent_deembedding_observation_arch: ModelArchitecture = ModelArchitecture(
                name="latent_deembedding_observation_arch"),
            latent_deembedding_state_arch: ModelArchitecture = ModelArchitecture(name="latent_deembedding_state_arch"),
            wae_gan_stop_grad=True,
            wae_gan_clip_by_global_norm: Optional[float] = None,
            wae_gan_pretrain_steps: Optional[int] = None,
            cost_fn: str = 'l22',
            cost_fn_obs: Optional[Callable] = None,
            cost_fn_state: Optional[Callable] = None,
            cost_weights=None,
            recover_fn_obs: Optional[Callable] = None,
            use_running_variance: bool = False,
            observation_wae_minimizer_lr: Optional[float] = None,
            observation_wae_maximizer_lr: Optional[float] = None,
            n_observation_wae_gan: int = 10,
            wae_gan_fixed_checkpoint_path: Optional[str] = None,
            null_action: Optional[Any] = None,
            ml_fn_obs: Optional[Callable] = None,
            ml_fn_state: Optional[Callable] = None,
            disable_common_embedding: bool = False,
            weight_decay: Optional[float] = 0.,
            optimizer_epsilon: float = 1e-7,
            lr_decay_power: Optional[float] = 0.,
            amsgrad: bool = False,
            *args, **kwargs
    ):
        # self.replay_buffer = replay_buffer
        self.wae_gan_pretrain_steps = wae_gan_pretrain_steps
        state_encoder_fc_arch, state_decoder_fc_arch, observation_encoder_fc_arch, observation_decoder_fc_arch, \
        observation_encoder_cnn_arch, observation_decoder_tcnn_arch, latent_embedding_arch, \
        latent_deembedding_observation_arch, latent_deembedding_state_arch, wae_gan_discriminator_arch, \
        latent_state_size, state_encoder_cnn_arch, state_decoder_tcnn_arch = \
            verify_wae_mdp_config(state_shape, observation_shape, action_shape, reward_shape, latent_state_size,
                                  emb_observation_size, emb_state_size, state_encoder_fc_arch, state_decoder_fc_arch,
                                  observation_encoder_fc_arch, observation_decoder_fc_arch,
                                  observation_encoder_cnn_arch, observation_decoder_tcnn_arch, latent_embedding_arch,
                                  latent_deembedding_observation_arch, latent_deembedding_state_arch,
                                  use_wae_gan, wae_gan_discriminator_arch,
                                  disable_common_embedding=disable_common_embedding,
                                  state_encoder_cnn_arch=state_encoder_cnn_arch,
                                  state_decoder_tcnn_arch=state_decoder_tcnn_arch,
                                  )
        if lr_decay_power > 0.:
            minimizer_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=minimizer_lr, decay_steps=n_updates // n_critic, end_learning_rate=1e-6,
                power=lr_decay_power)
            maximizer_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=maximizer_lr, decay_steps=n_updates, end_learning_rate=1e-6, power=lr_decay_power)
            if observation_wae_minimizer_lr is not None:
                observation_wae_minimizer_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=observation_wae_minimizer_lr, decay_steps=n_updates // n_critic,
                    end_learning_rate=1e-6, power=lr_decay_power)
            if observation_wae_maximizer_lr is not None:
                observation_wae_maximizer_lr = tf.keras.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=observation_wae_maximizer_lr, decay_steps=n_updates, end_learning_rate=1e-6,
                    power=lr_decay_power)

        if weight_decay == 0.:
            weight_decay = None

        minimizer = tfk.optimizers.get(
            {'class_name': optimizer_name,
             'config': {'learning_rate': minimizer_lr,
                        # 'weight_decay': weight_decay,
                        'epsilon': optimizer_epsilon,
                        'amsgrad': amsgrad}})
        maximizer = tfk.optimizers.get(
            {'class_name': optimizer_name,
             'config': {'learning_rate': maximizer_lr,
                        # 'weight_decay': weight_decay,
                        'epsilon': optimizer_epsilon,
                        'amsgrad': amsgrad}})
        self._wae_mdp = WassersteinBeliefMDP(
            state_shape=state_shape,
            observation_shape=observation_shape,
            action_shape=action_shape,
            reward_shape=reward_shape,
            label_shape=(0,),
            discretize_action_space=False,
            latent_embedding_network=latent_embedding_arch,
            state_encoder_fc_arch=state_encoder_fc_arch,
            state_encoder_cnn_arch=state_encoder_cnn_arch,
            state_decoder_fc_arch=state_decoder_fc_arch,
            state_decoder_tcnn_arch=state_decoder_tcnn_arch,
            observation_encoder_fc_arch=observation_encoder_fc_arch,
            observation_encoder_cnn_arch=observation_encoder_cnn_arch,
            observation_decoder_fc_arch=observation_decoder_fc_arch,
            observation_decoder_tcnn_arch=observation_decoder_tcnn_arch,
            latent_deembedding_state_arch=latent_deembedding_state_arch,
            latent_deembedding_observation_arch=latent_deembedding_observation_arch,
            action_decoder_network=None,
            transition_network=transition_arch,
            reward_network=reward_arch,
            decoder_network=None,
            latent_policy_network=latent_policy_arch,
            latent_stationary_network=latent_stationary_arch,
            steady_state_lipschitz_network=steady_state_lipschitz_arch,
            transition_loss_lipschitz_network=transition_loss_lipschitz_arch,
            latent_state_size=latent_state_size,
            state_encoder_temperature=state_encoder_temperature,
            state_prior_temperature=state_prior_temperature,
            wasserstein_regularizer_scale_factor=wasserstein_regularizer_scale_factor,
            autoencoder_optimizer=minimizer,
            wasserstein_regularizer_optimizer=maximizer,
            clip_by_global_norm=clip_by_global_norm,
            n_critic=n_critic,
            entropy_regularizer_scale_factor=0.,
            action_entropy_regularizer_scaling=0.,
            trainable_prior=False,
            use_wae_gan=use_wae_gan,
            emb_observation_size=emb_observation_size,
            emb_state_size=emb_state_size,
            wae_gan_regularizer_scale_factor=wae_gan_regularizer_scale_factor,
            wae_gan_discriminator_arch=wae_gan_discriminator_arch,
            wae_gan_stop_grad=wae_gan_stop_grad,
            wae_gan_clip_by_global_norm=wae_gan_clip_by_global_norm,
            cost_fn=cost_fn,
            cost_fn_obs=cost_fn_obs,
            cost_fn_state=cost_fn_state,
            cost_weights=cost_weights,
            recover_fn_obs=recover_fn_obs,
            observation_wae_minimizer_lr=observation_wae_minimizer_lr,
            observation_wae_maximizer_lr=observation_wae_maximizer_lr,
            wae_gan_fixed_checkpoint_path=wae_gan_fixed_checkpoint_path,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            optimizer_epsilon=optimizer_epsilon,
            amsgrad=amsgrad,
            *args, **kwargs)
        self._use_wae_gan = use_wae_gan
        self._learn_gan = use_wae_gan and (wae_gan_fixed_checkpoint_path is None)
        self._n_train_calls = 0
        self._n_observation_wae_gan = n_observation_wae_gan
        self._dataset_iterator = None
        self._wae_mdp_step = wae_mdp_step
        self._observation_encoder = (
            self._wae_mdp.encode_observation
            if use_wae_gan
            else tfk.Sequential([self._wae_mdp.pre_processing_layers[1]])
        )
        self._observation_decoder = (
            self._wae_mdp.decode_observation
            if use_wae_gan
            else self._wae_mdp.reconstruction_network.observation_distribution
        )
        self._emb_observation_size = emb_observation_size
        self._recover_fn_obs = recover_fn_obs
        self._obs_variance = np.zeros(self._wae_mdp._obs_variance.shape, dtype=np.float32)
        self._use_running_variance = use_running_variance
        self.cost_fn_obs = cost_fn_obs
        self.null_action = null_action
        self.ml_fn_obs = ml_fn_obs
        self.ml_fn_state = ml_fn_state

    def set_dataset(self, dataset: tf.data.Dataset):
        self._dataset_iterator = iter(dataset)

    def pretrain_wae_gan_encoder(self):
        if self._learn_gan and (self.wae_gan_pretrain_steps is not None):
            logger.warning(f"Pretraining the wae gan encoder for {self.wae_gan_pretrain_steps} steps.")
            for _ in range(self.wae_gan_pretrain_steps):
                state, _, _, _, next_state, _ = next(self._dataset_iterator)
                _, observation = state
                _, next_observation = next_state
                input_ = tf.concat([observation, next_observation], axis=0)
                self._wae_mdp._observation_wae.pretrain(input_)

    def pretrain(self, n):
        if self._learn_gan:
            logger.warning(f"Pretraining the wae gan for {n} steps.")
            for i in range(n):
                self._train_wae_gan()
            self.wae_mdp.update_wae_gan_target(0, 1, True)
        logger.warning(f"Pretraining the wae mdp for {n} steps.")
        for i in range(n):
            self._train_wae_mdp()

    def _train_wae_gan(self):
        gan_loss = {}
        if self._learn_gan:
            gan_loss = []
            for _ in range(self._n_observation_wae_gan):
                state, _, _, _, next_state, _ = next(self._dataset_iterator)
                _, observation = state
                _, next_observation = next_state
                gan_loss_ = ({
                    'gan_' + key: value
                    for key, value in self._wae_mdp._observation_wae.compute_and_apply_gradients(
                        tf.concat([observation, next_observation], axis=0)
                    ).items()
                })
                gan_loss.append(dict_array_to_dict_np(gan_loss_))
            self._wae_mdp.update_wae_gan_target(train_call=self._n_train_calls)
        return gan_loss

    def _train_wae_mdp(self):
        self._wae_mdp._obs_variance.assign(tf.zeros(self._wae_mdp._obs_variance.shape))
        self._wae_mdp._obs_variance_counter.assign(0)
        for _ in range(self._wae_mdp.n_critic):
            loss = self._wae_mdp.training_step(
                dataset_iterator=self._dataset_iterator,
                global_step=self._wae_mdp_step,
                dataset=None, batch_size=None, annealing_period=0,
                display_progressbar=False, start_step=None, epoch=None,
                progressbar=None, eval_and_save_model_interval=float('inf'),
                eval_steps=0, save_directory=None, log_name=None, train_summary_writer=None,
                log_interval=float('inf'), start_annealing_step=0, )
        self._obs_variance[:] = (self._wae_mdp._obs_variance.value() / tf.cast(self._wae_mdp._obs_variance_counter.value(), tf.float32)).numpy()
        return loss

    def get_obs_variance(self, latent_state: Optional[Float] = None):
        if latent_state is None:
            return tf.convert_to_tensor(self._obs_variance)
        else:
            return self._wae_mdp.get_obs_variance(latent_state=latent_state)

    def _train_metrics(self, gan_loss, loss):
        metrics = {
            **loss,
            # **gan_loss,
            **{key: value.result() for key, value in self._wae_mdp.loss_metrics.items()},
            **self._wae_mdp.temperature_metrics
        }
        self._wae_mdp.reset_metrics()

        irrelevant_keys = ['action_mse', 'gradients_max', 'gradients_min', 'latent_policy_entropy', 'marginal_variance']
        metrics = {k: v.numpy().item() for k, v in metrics.items() if k not in irrelevant_keys}

        return metrics

    def train(self) -> dict:
        self._n_train_calls += 1
        gan_loss = self._train_wae_gan()
        loss = self._train_wae_mdp()
        metrics = self._train_metrics(gan_loss, loss)
        return metrics

    @property
    def observation_encoded_shape(self):
        return self._emb_observation_size

    def evaluate(self):
        pass

    @property
    def wae_mdp(self):
        return self._wae_mdp

    def set_evaluation_dataset(self, inputs):
        return self._wae_mdp.set_evaluation_dataset(inputs)

    @property
    def observation_encoder(self) -> tfk.Model:
        return self._observation_encoder

    @property
    def step(self) -> int:
        return self._wae_mdp_step.value() // self._wae_mdp.n_critic

    @property
    def latent_transition(self) -> Callable[[Float, Float], tfd.Distribution]:
        """
        Retrieved the latent transition function of the WAE-MDP
        """
        return self._wae_mdp.relaxed_latent_transition

    @property
    def discrete_latent_transition(self) -> Callable[[Float, Float], tfd.Distribution]:
        return self._wae_mdp.discrete_latent_transition

    def latent_state_to_obs_(self, latent_state):
        obs = self.wae_mdp.reconstruction_network.observation_distribution(latent_state).mean()
        obs = self._recover_fn_obs(obs)
        return obs

    @property
    def obs_filter(self) -> Callable[[tf.Tensor, Float], tfd.Distribution]:
        """
        Gives the observation filter linked to the WAE-MDP attached to this worker.
        The observation filter is a function F: O x R -> Distr(Z), where
            O is the observation space, R is the set of real numbers, and Z is the set of latent states.
        Given an observation and a variance, F yields a distribution over latent states for which the observation
        function (learned through the WAE-MDP) gives the same observation.
        In practice, a Normal N centered in the input observation and scaled according to the variance is generated,
        and the log_prob function gives N.log_prob(observation_function(z)), where z is a latent state.

        Returns: the function F described above
        """
        worker = self
        wae_mdp = self._wae_mdp
        use_wae_gan = self._use_wae_gan
        recover_fn_obs = self._recover_fn_obs
        use_running_variance = self._use_running_variance and not self.wae_mdp.observation_regularizer

        class ObservationFilter(tfd.Distribution):

            def __init__(self, observation, variance: Float, dtype=tf.float32, validate_args=False,
                         allow_nan_stats=False):
                super().__init__(
                    dtype=dtype,
                    reparameterization_type=tfd.NOT_REPARAMETERIZED,
                    validate_args=validate_args,
                    allow_nan_stats=allow_nan_stats)

                observation_shape = tf.shape(observation)[1:]
                flat_observation = tf.cast(tf.reshape(observation, (tf.shape(observation)[0], -1)), tf.float32)

                if not use_running_variance:
                    variance = variance * tf.ones_like(flat_observation)

                self.normal_filter = tfd.TransformedDistribution(
                    distribution=tfd.MultivariateNormalDiag(
                        loc=flat_observation,
                        scale_diag=tf.sqrt(variance)),
                    bijector=tfb.Reshape(observation_shape))

            def _sample_n(self, n, seed=None, **kwargs):
                return NotImplemented

            def _log_prob(self, latent_state: Float, **kwargs):
                value = wae_mdp.reconstruction_network.observation_distribution(latent_state).mean()
                if not use_wae_gan:  
                    value = recover_fn_obs(value)
                value = tf.reshape(value, self.normal_filter.batch_shape + self.normal_filter.event_shape)
                return self.normal_filter.log_prob(value, **kwargs)

            def _prob(self, latent_state, **kwargs):
                value = wae_mdp.reconstruction_network.observation_distribution(latent_state).mean()
                if not use_wae_gan:  
                    value = recover_fn_obs(value)
                return self.normal_filter.prob(value, **kwargs)

            def _event_shape(self):
                return tf.TensorShape(wae_mdp.latent_state_size, )

            def _event_shape_tensor(self):
                return tf.constant(self._event_shape(), dtype=tf.int32)

            def _batch_shape(self):
                return self.normal_filter.batch_shape

            def _batch_shape_tensor(self, **parameter_kwargs):
                return self.normal_filter.batch_shape_tensor()

            @property
            def use_latent_observation(self):
                return use_wae_gan

        return ObservationFilter

    def state_observation_embedding_fn(self, state: Float, observation: Float, label: Float) -> Float:
        embedding = self.wae_mdp.relaxed_state_encoding(
            state=[state, observation],
            temperature=self.wae_mdp.state_encoder_temperature,
            label=label,
        ).sample()
        return embedding

    def discrete_state_observation_embedding_fn(self, state: Float, observation: Float, label: Float) -> Float:
        embedding = self.wae_mdp.binary_encode_state(
            state=[state, observation],
            label=label
        ).sample()
        return embedding

    @property
    def action_encoder(self):
        return tfb.Identity()

    @property
    def n_train_called(self):
        return self._n_train_calls

    @n_train_called.setter
    def n_train_called(self, value):
        value = int(value)
        self._n_train_calls = value
        if self._use_wae_gan and self._wae_mdp.wae_gan_target_update_freq is not None:
            self._wae_mdp._gan_target_last_update = None

    def save(self, path):
        self.wae_mdp.save(path, "wae_mdp")

    def load(self, path):
        self.wae_mdp.load(os.path.join(path, "wae_mdp"))

    def make_imaginary_env(self):
        worker = self
        null_action = tf.one_hot(tf.convert_to_tensor(self.null_action), worker.wae_mdp.number_of_discrete_actions)[None]

        @tf.function
        def is_done(latent_state):
            done = tf.round(latent_state[:, 0]) == 1
            return done

        @tf.function
        def reset_needed_latent_state(latent_state):
            action = tf.repeat(null_action, latent_state.shape[0], axis=0)
            latent_state_bis = worker.latent_transition(latent_state, action).sample()
            done = is_done(latent_state)
            done = tf.cast(done, tf.float32)[..., None]
            latent_state = (1 - done) * latent_state + done * latent_state_bis
            return [latent_state, ]

        @tf.function
        def is_reset_needed(latent_state):
            done = is_done(latent_state)
            return tf.reduce_any(done)

        class ImaginaryEnv:
            def __init__(self):
                self.current_latent_state = None
                self.n_action = worker.wae_mdp.number_of_discrete_actions

            @tf.function
            def init(self, state_obs_dict: dict):
                state_obs = [state_obs_dict[STATE], state_obs_dict[OBS]]
                latent_state = worker.wae_mdp.relaxed_state_encoding(
                    state_obs,
                    label=tf.zeros((state_obs[0].shape[0], 1)),
                    temperature=worker.wae_mdp.state_prior_temperature,
                ).sample()
                return latent_state

            @tf.function
            def step(self, latent_state, action):  
                latent_action = tf.one_hot(action, self.n_action)
                next_latent_state_with_reset = worker.latent_transition(latent_state, latent_action).sample()
                done = is_done(next_latent_state_with_reset)
                reward = worker.wae_mdp.reward_network([latent_state, latent_action, next_latent_state_with_reset])
                next_latent_state = tf.while_loop(
                    is_reset_needed,
                    reset_needed_latent_state,
                    [next_latent_state_with_reset])[0]
                return next_latent_state, reward, done

            @tf.function
            def latent_state_state_obs(self, latent_state):
                state, obs = worker.wae_mdp.reconstruction_network.distribution(latent_state).sample()
                obs = worker.ml_fn_obs(obs)
                state = worker.ml_fn_state(state)
                return state, obs


        return ImaginaryEnv()


def verify_wae_mdp_config(
        state_shape: Tuple[int, ...],
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        reward_shape: Tuple[int, ...],
        latent_state_size: int,
        emb_observation_size: int,
        emb_state_size: int,
        state_encoder_fc_arch: ModelArchitecture,
        state_decoder_fc_arch: ModelArchitecture,
        observation_encoder_fc_arch: ModelArchitecture,
        observation_decoder_fc_arch: ModelArchitecture,
        observation_encoder_cnn_arch: ModelArchitecture,
        observation_decoder_tcnn_arch: ModelArchitecture,
        latent_embedding_arch: ModelArchitecture,
        latent_deembedding_observation_arch: ModelArchitecture,
        latent_deembedding_state_arch: ModelArchitecture,
        use_wae_gan: bool,
        wae_gan_discriminator_arch: Optional[ModelArchitecture] = None,
        disable_common_embedding: bool = False,
        state_encoder_cnn_arch: Optional[ModelArchitecture] = None,
        state_decoder_tcnn_arch: Optional[ModelArchitecture] = None,
):
    assert observation_encoder_fc_arch is not None, "Observation encoder fc arch cannot be None"
    assert observation_decoder_fc_arch is not None, "Observation decoder fc arch cannot be None"
    assert state_encoder_fc_arch is not None, "State encoder fc arch cannot be None"
    assert state_decoder_fc_arch is not None, "State decoder fc arch cannot be None"

    if len(state_shape) > 1:
        assert state_encoder_cnn_arch is not None, "State is an image but no cnn architecture provided for encoding."
        if state_encoder_cnn_arch.input_dim is None:
            state_encoder_cnn_arch = state_encoder_cnn_arch.replace(input_dim=state_shape)
        assert state_encoder_cnn_arch.input_dim == state_shape, "State_encoder cnn arch input dim does not match"
        state_shape_post_cnn = get_model(state_encoder_cnn_arch, as_model=True).output_shape[1:]
        if state_encoder_fc_arch.input_dim is None:
            state_encoder_fc_arch = state_encoder_fc_arch.replace(input_dim=state_shape_post_cnn)
        assert state_encoder_fc_arch.input_dim == state_shape_post_cnn, \
            "State fc encoder input dim does not match cnn output shape."
        if state_encoder_fc_arch.output_dim is None:
            state_encoder_fc_arch = state_encoder_fc_arch.replace(output_dim=(emb_state_size,))
        assert state_encoder_fc_arch.output_dim == (emb_state_size,), \
            "State fc encoder output dim does not match emb observation size."
    else:
        if state_encoder_fc_arch.input_dim is None:
            state_encoder_fc_arch = state_encoder_fc_arch.replace(input_dim=state_shape)
        assert state_encoder_fc_arch.input_dim == state_shape, "State_encoder fc arch input dim does not match"
        if state_encoder_fc_arch.output_dim is None:
            state_encoder_fc_arch = state_encoder_fc_arch.replace(output_dim=(emb_state_size,))
        assert state_encoder_fc_arch.output_dim == (emb_state_size,), "State_encoder fc arch output dim does not match"

    if disable_common_embedding:
        state_encoder_fc_arch = state_encoder_fc_arch.replace(raw_last=True)

    if len(observation_shape) > 1:
        assert observation_encoder_cnn_arch is not None, "Observation is an image but no cnn architecture provided " \
                                                         "for encoding."
        if observation_encoder_cnn_arch.input_dim is None:
            observation_encoder_cnn_arch = observation_encoder_cnn_arch.replace(input_dim=observation_shape)
        assert observation_encoder_cnn_arch.input_dim == observation_shape, "Observation encoder cnn input shape " \
                                                                            "does not match."
        obs_shape_post_cnn = get_model(observation_encoder_cnn_arch, as_model=True).output_shape[1:]  # .as_numpy()
        if observation_encoder_fc_arch.input_dim is None:
            observation_encoder_fc_arch = observation_encoder_fc_arch.replace(input_dim=obs_shape_post_cnn)
        assert observation_encoder_fc_arch.input_dim == obs_shape_post_cnn, \
            "Observation fc encoder input dim does not match cnn output shape."
        if observation_encoder_fc_arch.output_dim is None:
            observation_encoder_fc_arch = observation_encoder_fc_arch.replace(output_dim=(emb_observation_size,))
        assert observation_encoder_fc_arch.output_dim == (emb_observation_size,), \
            "Observation fc encoder output dim does not match emb observation size."
    else:
        if observation_encoder_fc_arch.input_dim is None:
            observation_encoder_fc_arch = observation_encoder_fc_arch.replace(input_dim=observation_shape)
        assert observation_encoder_fc_arch.input_dim == observation_shape, \
            "Observation encoder fc input dim does not match observation shape"
        if observation_encoder_fc_arch.output_dim is None:
            observation_encoder_fc_arch = observation_encoder_fc_arch.replace(output_dim=(emb_observation_size,))
        assert observation_encoder_fc_arch.output_dim == (emb_observation_size,), \
            "Observation encoder fc output dim does not match emb observation size."

    if disable_common_embedding:
        observation_encoder_fc_arch = observation_encoder_fc_arch.replace(raw_last=True)
        assert emb_observation_size + emb_state_size + 1 == latent_state_size, \
            "Common embedding is disabled but latent state size do not match obs+state embedding."

    assert (latent_embedding_arch is not None) or disable_common_embedding, \
        "Latent embedding arch cannot be None unless common embedding is disabled"
    if not disable_common_embedding:
        if latent_embedding_arch.input_dim is None:
            latent_embedding_arch = latent_embedding_arch.replace(input_dim=(emb_observation_size + emb_state_size,))
        assert latent_embedding_arch.input_dim == (emb_observation_size + emb_state_size,), \
            "Latent embedding input dim does not match emb observation size + emb state size."
        if latent_embedding_arch.output_dim is None:
            latent_embedding_arch = latent_embedding_arch.replace(output_dim=(latent_state_size,))
        assert latent_embedding_arch.output_dim == (latent_state_size,), \
            "Latent embedding output dim does not match latent state size."
        assert latent_embedding_arch.raw_last, \
            "Latent embedding should not apply an activation at the end. There is an output_softclip option for that"
    else:
        latent_embedding_arch = None

    assert (latent_deembedding_state_arch is not None) or disable_common_embedding, \
        "Latent de-embedding state arch cannot be None"
    if not disable_common_embedding:
        if latent_deembedding_state_arch.input_dim is None:
            latent_deembedding_state_arch = latent_deembedding_state_arch.replace(input_dim=(latent_state_size,))
        assert latent_deembedding_state_arch.input_dim == (latent_state_size,), \
            "Latent de-embedding state input dim does not match latent state size."
        if latent_deembedding_state_arch.output_dim is None:
            latent_deembedding_state_arch = latent_deembedding_state_arch.replace(output_dim=(emb_state_size,))
        assert latent_deembedding_state_arch.output_dim == (emb_state_size,), \
            "Latent de-embedding state output dim does not match embedding state size."
    else:
        latent_deembedding_state_arch = None

    if state_decoder_fc_arch.input_dim is None:
        state_decoder_fc_arch = state_decoder_fc_arch.replace(input_dim=(emb_state_size,))
    assert state_decoder_fc_arch.input_dim == (emb_state_size,), "State decoder fc arch input dim does not match"

    if len(state_shape) == 1:
        if state_decoder_fc_arch.output_dim is None:
            state_decoder_fc_arch = state_decoder_fc_arch.replace(output_dim=state_shape)
        assert state_decoder_fc_arch.output_dim == state_shape, "State decoder fc arch output dim does not match"
        assert state_decoder_fc_arch.raw_last, "State decoder fc arch should be raw last."
    else:
        assert state_decoder_tcnn_arch is not None, \
            "State decoder transpose cnn not provided."
        assert state_decoder_tcnn_arch.input_dim is not None, \
            "State decoder transpose cnn input dim not provided."
        assert state_decoder_tcnn_arch.transpose, \
            "State decoder tcnn is not flagged as transpose."

        state_tcnn_input_dim = state_decoder_tcnn_arch.input_dim
        if state_decoder_fc_arch.output_dim is None:
            state_decoder_fc_arch = state_decoder_fc_arch.replace(output_dim=state_tcnn_input_dim)
        assert state_decoder_fc_arch.output_dim == state_tcnn_input_dim, \
            "State decoder fc arch output dim does not match"

        assert not state_decoder_fc_arch.raw_last, "State decoder fc arch should not be raw last."

        if state_decoder_tcnn_arch.output_dim is None:
            state_decoder_tcnn_arch = state_decoder_tcnn_arch.replace(output_dim=state_shape)
        assert state_decoder_tcnn_arch.output_dim == state_shape, \
            "state decoder tcnn arch output dim does not match"
        assert state_decoder_tcnn_arch.raw_last, "State decoder tcnn should be raw last."

    assert (latent_deembedding_observation_arch is not None) or disable_common_embedding, \
        "Latent de-embedding observation arch cannot be None"
    if not disable_common_embedding:
        if latent_deembedding_observation_arch.input_dim is None:
            latent_deembedding_observation_arch = latent_deembedding_observation_arch.replace(
                input_dim=(latent_state_size,))
        assert latent_deembedding_observation_arch.input_dim == (latent_state_size,), \
            "Latent de-embedding observation input dim does not match latent state size."
        if latent_deembedding_observation_arch.output_dim is None:
            latent_deembedding_observation_arch = latent_deembedding_observation_arch.replace(
                output_dim=(emb_observation_size,))
        assert latent_deembedding_observation_arch.output_dim == (emb_observation_size,), \
            "Latent de-embedding observation output dim does not match embedding observation size."
        if use_wae_gan:
            assert latent_deembedding_observation_arch.raw_last, "Latent deembedding observation needs to be raw last if using a gan."
    else:
        latent_deembedding_observation_arch = None

    if observation_decoder_fc_arch.input_dim is None:
        observation_decoder_fc_arch = observation_decoder_fc_arch.replace(input_dim=(emb_observation_size,))
    assert observation_decoder_fc_arch.input_dim == (emb_observation_size,), \
        "observation decoder fc arch input dim does not match"

    if len(observation_shape) == 1:
        if observation_decoder_fc_arch.output_dim is None:
            observation_decoder_fc_arch = observation_decoder_fc_arch.replace(output_dim=observation_shape)
        assert observation_decoder_fc_arch.output_dim == observation_shape, \
            "observation decoder fc arch output dim does not match"
        assert observation_decoder_fc_arch.raw_last, "Observation decoder fc should be raw last."
    else:
        assert observation_decoder_tcnn_arch is not None, \
            "Observation decoder transpose cnn not provided."
        assert observation_decoder_tcnn_arch.input_dim is not None, \
            "Observation decoder transpose cnn input dim not provided."
        assert observation_decoder_tcnn_arch.transpose, \
            "Observation decoder tcnn is not flagged as transpose."

        obs_tcnn_input_dim = observation_decoder_tcnn_arch.input_dim
        if observation_decoder_fc_arch.output_dim is None:
            observation_decoder_fc_arch = observation_decoder_fc_arch.replace(output_dim=obs_tcnn_input_dim)
        assert observation_decoder_fc_arch.output_dim == obs_tcnn_input_dim, \
            "observation decoder fc arch output dim does not match"

        assert not observation_decoder_fc_arch.raw_last, "Observation decoder fc arch should not be raw last."

        if observation_decoder_tcnn_arch.output_dim is None:
            observation_decoder_tcnn_arch = observation_decoder_tcnn_arch.replace(output_dim=observation_shape)
        assert observation_decoder_tcnn_arch.output_dim == observation_shape, \
            "observation decoder tcnn arch output dim does not match"
        assert observation_decoder_tcnn_arch.raw_last, "Observation decoder tcnn should be raw last."

    if use_wae_gan:
        assert wae_gan_discriminator_arch is not None, \
            "Wae Gan discriminator cannot be None"
        if wae_gan_discriminator_arch.input_dim is None:
            wae_gan_discriminator_arch = wae_gan_discriminator_arch.replace(input_dim=(emb_observation_size,))
        assert wae_gan_discriminator_arch.input_dim == (emb_observation_size,), \
            "Wae gan discriminator input dim is not emb observation size"
        if wae_gan_discriminator_arch.output_dim is None:
            wae_gan_discriminator_arch = wae_gan_discriminator_arch.replace(output_dim=(1,))
        assert wae_gan_discriminator_arch.output_dim == (1,), \
            "Wae gan discriminator output dim is not 1"
        assert wae_gan_discriminator_arch.raw_last, \
            "Wae gan discriminator should not apply activation at the end"
        assert observation_encoder_fc_arch.raw_last, \
            "Observation encoder fc should not apply activation at the end"

    return state_encoder_fc_arch, state_decoder_fc_arch, observation_encoder_fc_arch, observation_decoder_fc_arch, \
           observation_encoder_cnn_arch, observation_decoder_tcnn_arch, latent_embedding_arch, \
           latent_deembedding_observation_arch, latent_deembedding_state_arch, wae_gan_discriminator_arch, \
           latent_state_size, state_encoder_cnn_arch, state_decoder_tcnn_arch
