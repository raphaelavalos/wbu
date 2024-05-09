import os.path
from typing import Callable, Tuple, Optional, Dict, Union

import numpy as np
import tensorflow as tf
from keras.metrics import Mean, MeanSquaredError
from tf_agents.typing.types import Float
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd

from belief_learner.utils import get_logger
from belief_learner.utils.decorators import log_usage
from belief_learner.networks.decoders import Decoder, Encoder
from belief_learner.utils.costs import get_cost_fn
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.networks.get_model import get_model
logger = get_logger(__name__)


class WaeGan(tf.Module):
    @log_usage
    def __init__(
            self,
            encoder_fc_arch: ModelArchitecture,
            decoder_fc_arch: ModelArchitecture,
            latent_discriminator: ModelArchitecture,
            input_shape: Tuple[int, ...],
            minimizer_lr: Optional[float] = None,
            maximizer_lr: Optional[float] = None,
            latent_space_size: int = 8,
            encoder_cnn_arch: Optional[ModelArchitecture] = None,
            decoder_tcnn_arch: Optional[ModelArchitecture] = None,
            cost_fn: Union[str, Callable[[Float, Float], Float]] = "l22",
            latent_regularizer_scale_factor: float = 10.,
            clip_by_global_norm: Optional[float] = None,
            n_adversarial_updates: int = 1,
    ):
        self.clip_by_global_norm = clip_by_global_norm
        self._max_optimizer = None
        self._min_optimizer = None
        if maximizer_lr is not None:
            self._max_optimizer = tfk.optimizers.Adam(
                learning_rate=maximizer_lr,
                beta_1=0.5,
                beta_2=0.999, )
        if minimizer_lr is not None:
            self._min_optimizer = tfk.optimizers.Adam(
                learning_rate=minimizer_lr,
                beta_1=0.5,
                beta_2=0.999, )

        self.encoder = Encoder(
            latent_variable=tfkl.Input(shape=input_shape, name="wae_gan_input"),
            encoder_fc_arch=encoder_fc_arch,
            encoder_cnn_arch=encoder_cnn_arch,
            name='wae_gan_encoder',
            output_softclip=lambda x: x,
            # output_softclip=tf.nn.tanh
        )

        self.latent_discriminator = get_model(latent_discriminator, as_model=True)

        self.decoder = Decoder(
            latent_variable=tfkl.Input(shape=(latent_space_size,)),
            decoder_fc_arch=decoder_fc_arch,
            decoder_tcnn_arch=decoder_tcnn_arch,
            name='wae_gan_decoder')
        if isinstance(cost_fn, str):
            cost_fn = get_cost_fn(cost_fn)
        self.cost_fn = lambda *x: .2 * cost_fn(*x)
        self.regularizer_scale = latent_regularizer_scale_factor
        self._latent_space_size = latent_space_size
        self.n_adversarial_updates = n_adversarial_updates
        self.loss_metrics = {
            'mse': MeanSquaredError(name='observation_mse'),
            'reconstruction_loss': Mean(name='reconstruction_loss'),
            'adversarial_loss': Mean(name='adversarial_loss'),
            'minimizer_loss': Mean(name='minimizer_loss'),
            'latent_variable_mean': Mean(name='latent_variable_mean'),
            'latent_variable_std': Mean(name='latent_variable_std'),
            'penalty': Mean(name='penalty'),
            'grad_norm_min': Mean(name='grad_norm_min'),
            'grad_norm_max': Mean(name='grad_norm_max'),
        }

        self.optimizer_variable = {
            'max': (self._max_optimizer, self.latent_discriminator.trainable_variables),
            'min': (self._min_optimizer, self.encoder.trainable_variables + self.decoder.trainable_variables),
        }
        self.pretrain_sample_size = 1000
        self.pretrain_steps = 200
        self.input_shape = input_shape

        self.trick = False
        self._evaluation_dataset = None
        self._prev_eval_encode = None
        self._checkpointables = ['encoder', 'decoder', 'latent_discriminator', '_max_optimizer', '_min_optimizer']

        super().__init__()

    @property
    def latent_space_size(self):
        return self._latent_space_size

    @property
    def max_optimizer(self):
        return self._max_optimizer

    @property
    def min_optimizer(self):
        return self._min_optimizer

    @property
    def latent_prior(self):
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=(self.latent_space_size,)),
            scale_diag=tf.ones(shape=(self._latent_space_size,)))

    @tf.function
    def z_adversary(self, inputs):
        output = self.latent_discriminator(inputs)
        if self.trick:
            sigma2_p = 1.
            normsq = tf.reduce_sum(tf.square(inputs), 1)
            output = output - normsq / (2. * sigma2_p) \
                     - .5 * tf.math.log(2. * np.pi) \
                     - .5 * self.latent_space_size * tf.math.log(sigma2_p)
        return output

    def gan_penalty(self, z_encoder, z_prior):
        logits_prior = self.z_adversary(z_prior)
        logits_encoder = self.z_adversary(z_encoder)
        loss_prior = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_prior, labels=tf.ones_like(logits_prior)
        )
        loss_encoder = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_encoder, labels=tf.zeros_like(logits_encoder)
        )
        loss_encoder_trick = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_encoder, labels=tf.ones_like(logits_encoder)
        )
        loss_match = loss_encoder_trick
        loss_adversary = (loss_encoder + loss_prior)
        return loss_adversary[:, 0], loss_match[:, 0]

    @tf.function
    def adversarial_call(self, encoded_samples):
        """
        Compute log(sigmoid(D(z_prior))) + log(1 - sigmoid(D(z_encoder)))
        where z_prior is sampled from the latent prior distribution,
        z_encoder is sampled from the encoder, and D is the latent discriminator

        Args:
            encoded_samples: encoded latent samples (z_encoder)
        Returns:
            log(sigmoid(D(z_prior))) + log(1 - sigmoid(D(z_encoder)))
        """
        batch_size = tf.shape(encoded_samples)[0]
        z_prior = self.latent_prior.sample(batch_size)
        z_encoder = encoded_samples
        # stable computation of
        return tf.math.log(tf.nn.sigmoid(self.latent_discriminator(z_prior))) + \
               tf.math.log(1. - tf.nn.sigmoid(self.latent_discriminator(z_encoder)))

    @tf.function
    def __call__(self, inputs, training: bool = False, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]
        z_encoder = self.encoder(inputs, training=training)
        z_prior = self.latent_prior.sample((batch_size,))
        gan_loss, penalty = self.gan_penalty(z_encoder, z_prior)

        decoded_inputs = self.decoder(z_encoder, training=training)
        reconstruction_loss = self.cost_fn(inputs, decoded_inputs)
        wae_loss = reconstruction_loss + self.regularizer_scale * penalty
        tf.assert_equal(wae_loss.shape, (batch_size,))
        self.loss_metrics['mse'](inputs, decoded_inputs)
        self.loss_metrics['adversarial_loss'](self.regularizer_scale * gan_loss)
        self.loss_metrics['reconstruction_loss'](reconstruction_loss)
        self.loss_metrics['minimizer_loss'](wae_loss)
        self.loss_metrics['penalty'](penalty)
        self.loss_metrics['latent_variable_mean'](tf.reduce_mean(z_encoder, axis=0))
        self.loss_metrics['latent_variable_std'](tf.math.reduce_std(z_encoder, axis=0))

        squared_error = tf.square((inputs - decoded_inputs))

        return {
            'adversarial_loss': self.regularizer_scale * gan_loss,
            'wae_loss': wae_loss,
            'mse': tf.reduce_mean(squared_error, axis=1),
            'max_se': tf.reduce_max(squared_error, axis=1),
            "reconstruction_loss": reconstruction_loss,
            "penalty": penalty,
        }

    @tf.function
    def loss_pretrain(self, inputs):
        batch_size = tf.shape(inputs)[0]
        z_prior = self.latent_prior.sample((batch_size,))
        z_encoder = self.encoder(inputs, training=True)
        mean_prior = tf.reduce_mean(z_prior, axis=0, keepdims=True)
        mean_encoder = tf.reduce_mean(z_encoder, axis=0, keepdims=True)
        mean_loss = tf.reduce_mean(tf.square(mean_encoder - mean_prior))
        cov_prior = tf.transpose(z_prior - mean_prior) @ (z_prior - mean_prior)
        cov_prior /= tf.cast(batch_size, tf.float32) - 1.
        cov_encoder = tf.transpose(z_encoder - mean_encoder) @ (z_encoder - mean_encoder)
        cov_encoder /= tf.cast(batch_size, tf.float32) - 1.
        cov_loss = tf.reduce_mean(tf.square(cov_encoder - cov_prior))
        return mean_loss + cov_loss

    @tf.function
    def pretrain(self, inputs):
        with tf.GradientTape() as tape:
            loss = self.loss_pretrain(inputs)
        self.min_optimizer.minimize(loss, self.encoder.trainable_variables, tape=tape)
        return loss

    @tf.function
    def compute_and_apply_gradients(self, inputs):
        step = 1

        with tf.GradientTape(persistent=True) as tape:
            loss = self(inputs, training=True)
            loss['max'] = tf.reduce_mean(loss['adversarial_loss'], axis=0)
            loss['min'] = tf.reduce_mean(loss['wae_loss'], axis=0)

        grad_norms = {}

        for direction, (optim, variables) in self.optimizer_variable.items():
            if direction == 'max' or (direction == 'min' and step % self.n_adversarial_updates == 0):
                gradients = tape.gradient(loss[direction], variables)
                if self.clip_by_global_norm:
                    gradients, grad_norm = tf.clip_by_global_norm(gradients, self.clip_by_global_norm)
                else:
                    grad_norm = tf.linalg.global_norm(gradients)
                self.loss_metrics["grad_norm_" + direction](grad_norm)
                grad_norms["grad_norm_" + direction] = grad_norm
                optim.apply_gradients(zip(gradients, variables))
        del tape
        return {'maximizer': -1. * loss['max'],
                'minimizer': loss['min'],
                'mse': tf.reduce_mean(loss['mse'], axis=0),
                'max_se': tf.reduce_mean(loss["max_se"], axis=0),
                "reconstruction_loss": tf.reduce_mean(loss["reconstruction_loss"], axis=0),
                "penalty": tf.reduce_mean(loss["penalty"], axis=0),
                **grad_norms
                }

    def summary(self):
        self.encoder.summary()
        self.latent_discriminator.summary()
        self.decoder.summary()

    def set_evaluation_dataset(self, inputs):
        self._evaluation_dataset = inputs.copy()
        self._prev_eval_encode = self.encoder(self._evaluation_dataset)

    @staticmethod
    def compare(x, y):
        """
        Returns the l2 distance as well as the cosine similarity between vectors x and y.
        The l2 distance is the usual Euclidean distance.
        The cosine similarity outputs a value that ranges from -1 to 1:
            * -1 means that the two vectors are exactly opposed
            * 1 means that the two vectors are exactly the same
            * 0 indicates the orthogonality of the two vectors, which is the complete decorrelation.
        """
        return {
            'l2': tf.reduce_mean(get_cost_fn('l2')(x, y)),
            'cosine_similarity': tf.reduce_mean(get_cost_fn('cosine_similarity')(x, y)),
        }

    def evaluate(self):
        # notice here that batch norm makes the behavior of the following operation is different to during training
        encoded = self.encoder(self._evaluation_dataset)
        stats = self.compare(encoded, self._prev_eval_encode)
        stats.update({
            'current_latent_mean': tf.reduce_mean(encoded, axis=0),
            'previous_latent_mean': tf.reduce_mean(self._prev_eval_encode, axis=0),
            'current_latent_std': tf.math.reduce_std(encoded, axis=0),
            'previous_latent_std': tf.math.reduce_std(self._prev_eval_encode, axis=0),
        })  # get current and previous mean, std
        self._prev_eval_encode = encoded
        return stats

    def get_weights(self):
        weights = {
            "encoder": self.encoder.get_weights(),
            "decoder": self.decoder.get_weights(),
            "latent_discriminator": self.latent_discriminator.get_weights(),
        }
        return weights

    def set_weights(self, weights: Dict, tau: Optional[Union[int, float]]):
        # logger.warning(f"WAE GAN target update {tau}")
        for name in ["encoder", "decoder", "latent_discriminator"]:
            if tau is None or tau >= 1:
                getattr(self, name).set_weights(weights[name])
            else:
                getattr(self, name).set_weights(
                    [tau * target_weight + (1 - tau) * network_weight for network_weight, target_weight in
                     zip(weights[name], getattr(self, name).get_weights())])


    def save(self, path: str):
        assert os.path.exists(path), f"Path {path} does not exist"
        for key in ['encoder', 'decoder', 'latent_discriminator']:
            model = getattr(self, key)
            model.save_weights(os.path.join(path, f"wae_gan_{key}.h5"))
        max_optimizer_weights = self._max_optimizer.get_weights()
        np.save(os.path.join(path, 'wae_gan_max_optimizer_weights.npy'), max_optimizer_weights, allow_pickle=True)
        min_optimizer_weights = self._min_optimizer.get_weights()
        np.save(os.path.join(path, 'wae_gan_min_optimizer_weights.npy'), min_optimizer_weights, allow_pickle=True)
        logger.warning(f"Saved WAE MDP at {path}.")

    def load(self, path: str):
        assert os.path.exists(path), f"Path {path} does not exist"
        for key in ['encoder', 'decoder', 'latent_discriminator']:
            model = getattr(self, key)
            model.load_weights(os.path.join(path, f"wae_gan_{key}.h5"))
        max_optimizer_weights = np.load(os.path.join(path, 'wae_gan_max_optimizer_weights.npy'), allow_pickle=True)
        if self._max_optimizer.iterations != 0:
            self._max_optimizer.set_weights(max_optimizer_weights)
        else:
            logger.warning("Max optimizer was not loaded.")
        min_optimizer_weights = np.load(os.path.join(path, 'wae_gan_min_optimizer_weights.npy'), allow_pickle=True)
        if self._min_optimizer.iterations != 0:
            self._min_optimizer.set_weights(min_optimizer_weights)
        else:
            logger.warning("Min optimizer was not loaded.")
        logger.warning(f"WAE GAN was loaded from {path}.")

