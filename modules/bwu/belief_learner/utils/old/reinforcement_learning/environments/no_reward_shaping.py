from typing import Any, Optional

from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from tf_agents.typing.types import Float


class NoRewardShapingWrapper(PyEnvironmentBaseWrapper):

    def __init__(self, env: Any, prev_shaping: Optional[Float] = None):
        super().__init__(env)
        self._prev_shaping = prev_shaping

    def _step(self, action):
        if hasattr(self, 'gym'):
            self.gym.unwrapped.prev_shaping = self._prev_shaping
        return super(NoRewardShapingWrapper, self)._step(action)
