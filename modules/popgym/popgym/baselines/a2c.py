# This script will run the PPO experiments listed in the paper
# Feel free to set the environment variables to whatever envs/models
# you would like to test

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
from typing import Any, List

import ray
import torch
# from ray.air.callbacks.wandb import WandbLoggerCallback
from ray.tune.registry import register_env  # noqa: F401

import popgym  # noqa: F401
from popgym.baselines.ray_models.ray_diffnc import DiffNC  # noqa: F401
from popgym.baselines.ray_models.ray_elman import Elman
from popgym.baselines.ray_models.ray_frameconv import Frameconv
from popgym.baselines.ray_models.ray_framestack import Framestack
from popgym.baselines.ray_models.ray_fwp import (
    BigFastWeightProgrammer,
    FastWeightProgrammer,
)
from popgym.baselines.ray_models.ray_gru import GRU
from popgym.baselines.ray_models.ray_indrnn import IndRNN
from popgym.baselines.ray_models.ray_linear_attention import (
    BigLinearAttention,
    LinearAttention,
)
from popgym.baselines.ray_models.ray_lmu import LMU
from popgym.baselines.ray_models.ray_lstm import LSTM
from popgym.baselines.ray_models.ray_mlp import MLP, BasicMLP
from popgym.baselines.ray_models.ray_s4d import S4D

env_names: List[Any] = []

env_types = os.environ.get("POPGYM_EXPERIMENT", "ALL_ENVS")
desired_models = os.environ.get("POPGYM_MODELS", "ALL")
num_splits = int(os.environ.get("POPGYM_NUM_SPLITS", 1))
split_id = int(os.environ.get("POPGYM_SPLIT_ID", 0))
project_id = os.environ.get("POPGYM_PROJECT", "popgym-debug")
gpu_per_worker = float(os.environ.get("POPGYM_GPU", 0.25))
max_steps = int(os.environ.get("POPGYM_STEPS", 1e6))
storage_path = os.environ.get("POPGYM_STORAGE", "/tmp/ray_results")
num_samples = int(os.environ.get("POPGYM_SAMPLES", 1))

# Hidden size of linear layers
h = 128
# Hidden size of memory
h_memory = 128
# Maximum episode length and backprop thru time truncation length
bptt_cutoff = 4
num_workers = 0
num_envs_per_worker = 16
train_batch_size = bptt_cutoff * max(num_workers, 1) * num_envs_per_worker
max_steps = 2e6

# Register all envs with ray
envs = popgym.ALL_ENVS
for cls, info in envs.items():
    env_name = info["id"]
    register_env(env_name, lambda x: cls())


# Of the registered envs, pick out the ones we actually want to run
env_names = []
for e in env_types.split(","):
    # getattr will either return a dict of {class: info[name}}
    # or just a class
    res = getattr(popgym, e)
    if isinstance(res, dict):
        desired_envs = list(res.keys())
    else:
        desired_envs = [res]

    for d in desired_envs:
        env_names.append(envs[d]["id"])

env_names = env_names[split_id::num_splits]

# Setup the models we want to train
attn_models = [
    LinearAttention,
    FastWeightProgrammer,
]
rnn_models = [LSTM, GRU, Elman, LMU, IndRNN, DiffNC]
conv_models = [S4D]
basic_models = [
    BasicMLP,
    MLP,
    Framestack,
    Frameconv,
]
models = rnn_models + attn_models + conv_models + basic_models

# Filter models by env variable
if desired_models != "ALL":
    models = [m for m in models if m.__name__ in desired_models.split(",")]


def trial_name(trial):
    env = trial.config["env"].replace("popgym-", "").replace("-v0", "")
    model = trial.config["model"]["custom_model"].__name__
    emb = trial.config["model"]["custom_model_config"].get("embedding", None)
    emb_size = trial.config["model"]["custom_model_config"].get("embedding_size", None)

    return "-".join([str(s) for s in [env, model, emb, emb_size] if s is not None])


config = {
    "evaluation_interval": 1,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    # Environments or env names
    # "env": ray.tune.grid_search(env_names),
    "env": "popgym-StatelessCartPoleHard-v0",
    # Should always be torch
    "framework": "torch",
    # Number of rollout workers
    "num_workers": num_workers,
    # Number of envs per rollout worker
    "num_envs_per_worker": num_envs_per_worker,
    # Num gpus used for the train worker
    "num_gpus": 0.,
    # Loss coeff for the ppo value function
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    # Num transitions in each training epoch
    "train_batch_size": train_batch_size,
    # Chunk size of transitions sent from rollout workers to trainer
    "rollout_fragment_length": bptt_cutoff,
    # Size of minibatches within epoch
    # "sgd_minibatch_size": 8 * bptt_cutoff,
    # decay gamma
    "gamma": 0.99,
    # Required due to RLlib PPO bugs
    # "horizon": bptt_cutoff,
    # RLlib bug with truncate_episodes:
    # each batch the temporal dim shrinks by one
    # for now, just use complete_episodes
    # "batch_mode": "complete_episodes",
    "batch_mode": "truncate_episodes",
    # "min_sample_timesteps_per_reporting": train_batch_size,
    # "min_sample_timesteps_per_iteration": train_batch_size,

    "min_sample_timesteps_per_iteration":10000,
    # Describe your RL model here
    "model": {
        # Truncate sequences into no more than this many timesteps
        "max_seq_len": bptt_cutoff,
        # Custom model class
        "custom_model": GRU, # ray.tune.grid_search(models),
        # Config passed to custom model constructor
        # see base_model.py to see how these are used
        "custom_model_config": {
            "preprocessor_input_size": h,
            "preprocessor": torch.nn.Sequential(
                torch.nn.Linear(h, h),
                torch.nn.LeakyReLU(inplace=True),
            ),
            "preprocessor_output_size": h,
            "hidden_size": h_memory,
            "postprocessor": torch.nn.Identity(),
            "actor": torch.nn.Sequential(
                torch.nn.Linear(h_memory, h),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(h, h),
                torch.nn.LeakyReLU(inplace=True),
            ),
            "critic": torch.nn.Sequential(
                torch.nn.Linear(h_memory, h),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(h, h),
                torch.nn.LeakyReLU(inplace=True),
            ),
            "postprocessor_output_size": h,
        },
    },
}
# When to stop training
stop = {"timesteps_total": max_steps}

# Write your own wandb entity here
logging_callbacks = [
    # WandbLoggerCallback(project=project_id, entity="prorok-lab", log_config=True)
]
if __name__ == '__main__':

    ray.init(local_mode=True)
    ray.tune.run(
        "A2C",
        config=config,
        stop=stop,
        callbacks=logging_callbacks,
        trial_name_creator=trial_name,
        # verbose=1,
        num_samples=num_samples,
        local_dir=storage_path,
    )