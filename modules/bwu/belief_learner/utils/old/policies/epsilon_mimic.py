from tf_agents.policies import tf_policy
from tf_agents.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from tf_agents.typing import types

from belief_learner.utils.old.policies.latent_policy import LatentPolicyOverRealStateAndActionSpaces


class EpsilonMimicPolicy(EpsilonGreedyPolicy):

    def __init__(
            self,
            policy: tf_policy.TFPolicy,
            latent_policy: LatentPolicyOverRealStateAndActionSpaces,
            epsilon: types.FloatOrReturningFloat,
    ):
        super().__init__(policy, epsilon)
        self._random_policy = latent_policy
