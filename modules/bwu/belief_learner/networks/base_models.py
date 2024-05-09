import abc

import tensorflow.keras as tfk
import tensorflow_probability.python.distributions as tfd


class DiscreteDistributionModel(tfk.Model):

    @abc.abstractmethod
    def relaxed_distribution(self, *args, **kwargs) -> tfd.Distribution:
        return NotImplemented

    @abc.abstractmethod
    def discrete_distribution(self, *args, **kwargs) -> tfd.Distribution:
        return NotImplemented

    def get_config(self):
        config = super(DiscreteDistributionModel, self).get_config()
        config.update({
            "relaxed_distribution": self.relaxed_distribution,
            "discrete_distribution": self.discrete_distribution})
        return config


class DistributionModel(tfk.Model):

    @abc.abstractmethod
    def distribution(self, *args, **kwargs) -> tfd.Distribution:
        return NotImplemented

    def get_config(self):
        config = super(DistributionModel, self).get_config()
        config.update({"distribution": self.discrete_distribution})
        return config
