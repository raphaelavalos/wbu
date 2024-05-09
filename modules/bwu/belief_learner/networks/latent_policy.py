import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow_probability.python.distributions as tfd
from tf_agents.typing.types import Float

from belief_learner.networks.base_models import DiscreteDistributionModel
from belief_learner.networks.tools import rep_layers


class LatentPolicyNetwork(DiscreteDistributionModel):

    def __init__(
            self,
            latent_state: tfkl.Input,
            latent_policy_network: tfk.Input,
            number_of_discrete_actions: int,
    ):
        _latent_policy_network = latent_policy_network(latent_state)
        _latent_policy_network = tfkl.Dense(
            units=number_of_discrete_actions,
            activation=None,
            name='latent_policy_categorical_logits'
        )(_latent_policy_network)
        super(LatentPolicyNetwork, self).__init__(
            inputs=latent_state,
            outputs=_latent_policy_network,
            name='latent_policy_network')

    def __repr__(self):
        name = getattr(self, 'name', self.__class__.__name__)
        layers = rep_layers(self.layers)
        return f"{name} : {layers}"

    def relaxed_distribution(
            self,
            latent_state: Float,
            temperature: Float
    ) -> tfd.Distribution:
        return tfd.RelaxedOneHotCategorical(
            logits=self(latent_state),
            temperature=temperature,
            allow_nan_stats=False)

    def discrete_distribution(self, latent_state: Float) -> tfd.Distribution:
        return tfd.OneHotCategorical(logits=self(latent_state), dtype=self.dtype)

    def get_config(self):
        config = super(LatentPolicyNetwork, self).get_config()
        return config
