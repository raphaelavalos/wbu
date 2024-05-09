from typing import Callable

import toml
from copy import deepcopy

from belief_learner.config.registery import get_from_config_register
from belief_learner.networks.model_architecture import ModelArchitecture
from belief_learner.wae.mdp.wasserstein_mdp import WassersteinRegularizerScaleFactor
import importlib

def model_arch_to_dict(dict_):
    for key, value in dict_.items():
        if isinstance(value, ModelArchitecture):
            dict_[key] = value.short_dict()
        elif isinstance(value, dict):
            dict_[key] = model_arch_to_dict(value)
    return dict_

def named_tuple_to_dict(dict_):
    for key, value in dict_.items():
        if isinstance(value, (ModelArchitecture, WassersteinRegularizerScaleFactor)):
            dict_[key] = value.short_dict()
        elif isinstance(value, dict):
            dict_[key] = named_tuple_to_dict(value)
    return dict_

def function_to_name(dict_):
    for key, value in dict_.items():
        if isinstance(value, Callable):
            dict_[key] = {"function": {"module": value.__module__, "name": value.__name__}}
        elif isinstance(value, dict):
            dict_[key] = function_to_name(value)
    return dict_

def config_to_dict(config):
    config = deepcopy(config)
    config = named_tuple_to_dict(config)
    config = function_to_name(config)
    return config

def config_to_toml(config):
    config = config_to_dict(config)
    return toml.dumps(config)


def dict_to_model_arch(dict_):
    if set(dict_.keys()).issubset(set(ModelArchitecture._fields)):
        return ModelArchitecture(**dict_)
    for key, value in dict_.items():
        if isinstance(value, dict):
            if set(value.keys()).issubset(set(ModelArchitecture._fields)):
                dict_[key] = ModelArchitecture(**value)
            elif set(value.keys()).issubset(set(WassersteinRegularizerScaleFactor._fields)):
                dict_[key] = WassersteinRegularizerScaleFactor(**value)
            else:
                dict_[key] = dict_to_model_arch(value)
        if isinstance(value, list) and len(value) == 2:
            if value[0] == "ModelArchFile":
                dict_[key] = ModelArchitecture.read_from_toml(value[1])
            elif value[0] == "ModelArchRegistery":
                dict_[key] = get_from_config_register(value[1])
    return dict_

def load_functions(dict_):
    if 'function' in dict_ and len(dict_) == 1:
        module = dict_["function"]["module"]
        name = dict_["function"]["name"]
        fun = getattr(importlib.import_module(module), name)
        return fun
    for key, value in dict_.items():
        if isinstance(value, dict):
            dict_[key] = load_functions(value)
    return dict_

def dict_to_config(dict_):
    config = dict_to_model_arch(dict_)
    config = load_functions(config)
    return config

def toml_to_config(path):
    config = toml.load(path)
    config = dict_to_config(config)
    return config

