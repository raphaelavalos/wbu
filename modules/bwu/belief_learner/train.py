import datetime
import math
import pickle

import wandb
import time
import os.path
from collections import namedtuple
from typing import Optional, Dict, Callable, Tuple, NamedTuple
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import gym
import json
import tensorflow as tf
from tf_agents.typing.types import Float
import tensorflow_probability.python.bijectors as tfb
from tensorflow import keras as tfk

from belief_learner import ReplayBuffer, STATE, OBS, ACTION, REWARD, SUB_BELIEF
from belief_learner.utils.costs import get_cost_fn
from belief_learner.utils.env.observation_flattener import boxify_to_loss, boxify_to_recover, boxify_to_ml
from belief_learner.utils.exceptions import EarlyStoppingException
from belief_learner.workers.belief_a2c_worker import BeliefA2CWorker
from belief_learner.workers.belief_worker import BeliefWorker
from belief_learner.config.utils import config_to_toml, config_to_dict
from belief_learner.workers.rl_worker import RLWorker, A2CWorker, DQNWorker
from belief_learner.utils import elements_scheme_builder, array_to_np
from belief_learner.utils.debug import is_debug
from belief_learner.utils.array import dict_array_to_dict_python
from belief_learner.utils.train import get_random_string
from belief_learner.workers.wae_mdp_worker import WasserteinMDPWorker
from belief_learner.utils.stats import TimeManager, combine_stats, get_mem
from belief_learner.utils import get_logger, time_memory, file_handler_to_logger

from pprint import pprint

logger = get_logger(__name__)


class Trainer:

    @time_memory
    def __init__(self,
                 config: Dict,
                 experiment_path: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 checkpoint_every: int = 0,
                 wae_stationary_latent_belief_sampler: bool = False,
                 disable_wandb: bool = False,
                 disable_early_stopping: bool = False,
                 ):
        self.experiment_path = experiment_path
        if self.experiment_path is not None:
            if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path)
            assert os.path.isdir(self.experiment_path)
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_' + get_random_string(8)
        self.experiment_name = experiment_name
        os.makedirs(os.path.join(self.experiment_path, self.experiment_name))
        file_handler_to_logger(os.path.join(self.experiment_path, self.experiment_name, 'logs.txt'))
        with open(os.path.join(self.experiment_path, self.experiment_name, "config.toml"), 'w') as f:
            print(config_to_toml(config.copy()), file=f)

        if disable_wandb:
            self.wandb_run = None
        else:
            self.wandb_run = wandb.init(
                project=config['env']['config']['env_name'].replace('/', '-'),
                entity='beliefwae',
                name=self.experiment_name,
                config=config_to_dict(config),
                settings=wandb.Settings(_disable_stats=True))

        self.disable_early_stopping = disable_early_stopping

        logger.warning(f"The experiment is logged in "
                       f"{os.path.abspath(os.path.join(self.experiment_path, self.experiment_name))}")

        self.config = config

        # Create env for setting everything up
        env_config = self.config["env"]["config"]
        env_maker = self.config["env"]["env_maker"]
        env = env_maker(**env_config)

        num_workers = 2 if config.get("belief_a2c_worker", False) else 3
        round_robin_weights = config.get("round_robin_weights", None)
        if round_robin_weights is None:
            round_robin_weights = [1, ] * num_workers
        else:
            assert len(round_robin_weights) == num_workers
        self._round_robin_weights = round_robin_weights

        self.replay_buffer = self._build_replay_buffer(env)
        self.wae_mdp_worker = self._build_wae_mpd_worker(env)
        if config.get("belief_a2c_worker", False):
            self.belief_a2c_worker = self._build_belief_a2c_worker()
            self.rl_worker = self.belief_a2c_worker  # for compatibility
            self.belief_worker = self.belief_a2c_worker  # for compatibility
            self.workers = [self.wae_mdp_worker, self.belief_a2c_worker]
        else:
            self.belief_worker = self._build_belief_worker()
            self.rl_worker = self._build_rl_worker()
            self.workers = [self.wae_mdp_worker, self.belief_worker, self.rl_worker]
        if wae_stationary_latent_belief_sampler:
            self.wae_mdp_worker.wae_mdp.set_latent_belief_sampler(self._latent_belief_sampler())

        self.checkpoint_every = checkpoint_every

        self._epoch = 0

    @time_memory
    def _build_replay_buffer(self, env) -> ReplayBuffer:
        env_config = self.config["env"]["config"]
        env_maker = self.config["env"]["env_maker"]
        wae_capacity = self.config["wae"]["buffer_size"]
        belief_capacity = self.config["belief"]["buffer_size"]
        policy_capacity = self.config["policy"]["buffer_size"]
        capacity = max(policy_capacity, belief_capacity, wae_capacity)
        nbr_envs = self.config["policy"]["nbr_environments"]
        sub_belief_shape = self.config["belief"]["sub_belief_shape"]
        reset_obs_state = env._reset_state.copy()
        horizon = env.horizon
        sub_belief_space = gym.spaces.Box(float('-inf'), float('+inf'), sub_belief_shape)
        elements_scheme = elements_scheme_builder(env_maker=env_maker, env_config=env_config,
                                                  sub_belief_space=sub_belief_space, env=env)

        replay_buffer = ReplayBuffer(elements_scheme=elements_scheme,
                                     capacity=capacity,
                                     policy_capacity=policy_capacity,
                                     belief_capacity=belief_capacity,
                                     wae_capacity=wae_capacity,
                                     horizon=horizon,
                                     reset_obs_state=reset_obs_state,
                                     nbr_feeds=nbr_envs,
                                     default_prev_action=None,
                                     )
        return replay_buffer

    def _latent_belief_sampler(self) -> Callable[[int], NamedTuple]:
        """
        Create a stationary belief sampler allowing drawing z, a, z' where:
        b, a ~ RB ; z ~ b ; z' ~ P(. | z, a)
        RB is the replay buffer, b is a latent belief, a is an action, and
        z, z' are respectively current and successor latent states.
        """
        logger.warning("Latent belief sampler ignores the batch size and replace it with "
                       "the wae batch size.")

        def _sample() -> Tuple[Float, Float, Float]:
            batch_size = self.config["wae"]["batch_size"]
            batch = self.replay_buffer.sample(batch_size=batch_size, worker='wae')
            sub_belief = batch[SUB_BELIEF]
            action = tf.one_hot(batch[ACTION], self.replay_buffer.nbr_actions)
            latent_state = self.belief_worker._made.relaxed_distribution(conditional_input=sub_belief).sample()
            next_latent_state = self.wae_mdp_worker.latent_transition(latent_state, action).sample()
            return latent_state, action, next_latent_state

        def _sampler(batch_size: int):
            return namedtuple(
                'StationaryBeliefSampler', ['sample'], )(_sample)

        _sampler._variables = self.wae_mdp_worker.wae_mdp.transition_network.trainable_variables

        return _sampler

    @time_memory
    def _build_wae_mpd_worker(self, env) -> WasserteinMDPWorker:
        state_shape = self.replay_buffer.elements_scheme[STATE].shape
        obs_shape = self.replay_buffer.elements_scheme[OBS].shape
        wae_config = deepcopy(self.config['wae'])
        reward_shape = self.replay_buffer.elements_scheme[REWARD].shape
        if reward_shape == ():
            reward_shape = (1,)
        cost_fn_obs = None
        cost_fn_state = None
        cost_fn = wae_config['cost_fn']
        recover_fn_state = tf.identity
        recover_fn_obs = None
        ml_fn_obs = None
        ml_fn_state = None

        if hasattr(env, "pre_boxify_obs_space"):
            cost_fn_obs = boxify_to_loss(env.pre_boxify_obs_space, env.pre_boxify_obs_space_shapes, cost_fn)
            cost_fn_state = boxify_to_loss(env.pre_boxify_state_space, env.pre_boxify_state_space_shapes, cost_fn)
            recover_fn_obs = boxify_to_recover(env.pre_boxify_obs_space, env.pre_boxify_obs_space_shapes)
            ml_fn_obs = boxify_to_ml(env.pre_boxify_obs_space, env.pre_boxify_obs_space_shapes)
            ml_fn_state = boxify_to_ml(env.pre_boxify_state_space, env.pre_boxify_state_space_shapes)
        if hasattr(env, 'ale'):
            # Atari env
            cost_fn_obs = get_cost_fn(wae_config['cost_fn'])
            cost_fn_state = get_cost_fn("binary_cross_entropy")
            recover_fn_obs = tf.identity
        if 'POMinAtar' in env.spec.id:
            cost_fn = wae_config['cost_fn']
            cost_fn_obs = get_cost_fn(cost_fn)
            cost_fn_state = get_cost_fn(cost_fn)
            if cost_fn == 'binary_cross_entropy':
                recover_fn_state = tf.nn.sigmoid
                recover_fn_obs = tf.nn.sigmoid
            else:
                recover_fn_state = tf.identity
                recover_fn_obs = tf.identity


        wae_worker = WasserteinMDPWorker(
            state_shape=state_shape,
            observation_shape=obs_shape,
            action_shape=(self.replay_buffer.nbr_actions,),
            reward_shape=reward_shape,
            cost_fn_obs=cost_fn_obs,
            cost_fn_state=cost_fn_state,
            recover_fn_state=recover_fn_state,
            recover_fn_obs=recover_fn_obs,
            use_running_variance=self.config["belief"]["use_running_variance"],
            null_action=self.replay_buffer.default_prev_action,
            ml_fn_obs=ml_fn_obs,
            ml_fn_state=ml_fn_state,
            env_name=self.config["env"]["config"]["env_name"],
            n_updates=(
                    self.config["total_steps"]
                    // (self.config["policy"]["env_step_per_batch"] * self.config["policy"]["nbr_environments"])
                    * (self._round_robin_weights[0] // self._round_robin_weights[1])
            ),
            **wae_config,
        )
        return wae_worker

    @time_memory
    def _build_belief_a2c_worker(self) -> BeliefA2CWorker:
        rl_config = self.config["policy"].copy()
        env_config = self.config["env"]["config"]
        env_maker = self.config["env"]["env_maker"]
        belief_config = self.config["belief"].copy()
        observation_encoded_shape = self.wae_mdp_worker.observation_encoded_shape
        observation_encoder = self.wae_mdp_worker.observation_encoder

        config = {**belief_config, **rl_config}
        problem_keys = []
        for common_key in set(rl_config).intersection(belief_config):
            if common_key == "buffer_size":
                continue
            if belief_config[common_key] != rl_config[common_key]:
                problem_keys.append(common_key)
        if problem_keys:
            rl_s = {key: value for key, value in rl_config.items() if key in problem_keys}
            bf_s = {key: value for key, value in belief_config.items() if key in problem_keys}
            logger.warning(f"Conflicting parameters in rl_config and belief_config while building BeliefA2CWorker: "
                           f"rl_config {rl_s}, belief_config {bf_s}. Going with rl_config parameters.")
            config.update({"belief_" + key: value for key, value in bf_s.items()})

        if not self.config["wae"]["use_wae_gan"]:
            observation_encoded_shape = self.replay_buffer.elements_scheme[OBS].shape
            observation_encoder = tfb.Identity()

        observation_encoder_cnn_arch = self.config["wae"].get("observation_encoder_cnn_arch", None)

        if config.get('categorical_beliefs', False):
            latent_transition = self.wae_mdp_worker.discrete_latent_transition
            state_embedding_fn = self.wae_mdp_worker.discrete_state_observation_embedding_fn
        else:
            latent_transition = self.wae_mdp_worker.latent_transition
            state_embedding_fn = self.wae_mdp_worker.state_observation_embedding_fn

        worker = BeliefA2CWorker(
            replay_buffer=self.replay_buffer,
            observation_encoder=observation_encoder,
            action_encoder=self.wae_mdp_worker.action_encoder,
            latent_transition=latent_transition,
            obs_filter=self.wae_mdp_worker.obs_filter,
            observation_encoded_shape=observation_encoded_shape,
            action_encoded_shape=(self.replay_buffer.nbr_actions,),
            latent_state_size=self.config["wae"]["latent_state_size"],
            env_creator=env_maker,
            env_config=env_config,
            latent_state_to_obs_=self.wae_mdp_worker.latent_state_to_obs_,
            cost_fn_obs=self.wae_mdp_worker.cost_fn_obs if not self.config["wae"]["use_wae_gan"] else None,
            dual_optim=self.config["belief_a2c_dual_optim"],
            get_running_variance=self.wae_mdp_worker.get_obs_variance,
            env_img=self.wae_mdp_worker.make_imaginary_env(),
            state_observation_embedding_fn=state_embedding_fn,
            latent_reward=self.wae_mdp_worker.wae_mdp.reward_distribution,
            use_learned_variance=self.config["wae"].get("observation_regularizer", False),
            maximizer_lr=self.config['wae']['maximizer_lr'],
            transition_lipschitz_net=self.wae_mdp_worker.wae_mdp.create_transition_loss_lip_net(),
            maximizer_batch_size=self.config['wae']['batch_size'],
            observation_encoder_cnn_arch=observation_encoder_cnn_arch,
            n_belief_updates=(
                    self.config["total_steps"]
                    // (self.config["policy"]["env_step_per_batch"] * self.config["policy"]["nbr_environments"])
            ),
            **config
        )
        return worker

    @time_memory
    def _build_belief_worker(self) -> BeliefWorker:
        belief_config = self.config["belief"].copy()

        observation_encoded_shape = self.wae_mdp_worker.observation_encoded_shape
        observation_encoder = self.wae_mdp_worker.observation_encoder
        if not self.config["wae"]["use_wae_gan"]:
            observation_encoded_shape = self.replay_buffer.elements_scheme[OBS].shape
            observation_encoder = tfb.Identity()

        belief_worker = BeliefWorker(
            replay_buffer=self.replay_buffer,
            observation_encoder=observation_encoder,
            action_encoder=self.wae_mdp_worker.action_encoder,
            latent_transition=self.wae_mdp_worker.latent_transition,
            obs_filter=self.wae_mdp_worker.obs_filter,
            observation_encoded_shape=observation_encoded_shape,
            action_encoded_shape=(self.replay_buffer.nbr_actions,),
            latent_state_size=self.config["wae"]["latent_state_size"],
            latent_state_to_obs_=self.wae_mdp_worker.latent_state_to_obs_,
            **belief_config,
        )
        logger.debug("Belief Worker built.")
        logger.debug(f"Memory {get_mem()} GiB")
        return belief_worker

    @time_memory
    def _build_rl_worker(self) -> RLWorker:
        rl_config = self.config["policy"].copy()
        alg: str = rl_config["alg"]
        env_config = self.config["env"]["config"]
        env_maker = self.config["env"]["env_maker"]
        if alg.lower() == 'a2c':
            return A2CWorker(
                env_creator=env_maker,
                env_config=env_config,
                replay_buffer=self.replay_buffer,
                sub_belief_updater=self.belief_worker.sub_belief_encode,
                sub_belief_weights=self.belief_worker.get_sub_belief_weights(),
                belief_optimizer=self.belief_worker.optimizer,
                clip_by_global_norm_sub_belief=self.belief_worker.clip_by_global_norm,
                **rl_config,
            )
        elif alg.lower() == 'dqn':
            return DQNWorker(
                env_creator=env_maker,
                env_config=env_config,
                replay_buffer=self.replay_buffer,
                sub_belief_updater=self.belief_worker.sub_belief_encode,
                **rl_config,
            )
        logger.debug("RL Worker built.")
        logger.debug(f"Memory {get_mem()} GiB")

    @time_memory
    def _pre_fill_replay_buffer(self):
        logger.info(f"Interacting {self.config['start_interaction']} times with the each environment to "
                    f"fill the replay buffer.")
        self.rl_worker.interact(self.config['start_interaction'] // self.rl_worker.nbr_env, training=False,
                                force_random_action=True, fill_replay_buffer=True)
        # self.rl_worker.restart_all_envs_next_interact()
        # self.replay_buffer.reset_feeds()

    @time_memory
    def pre_training(self, n: Optional[int] = None):
        if n is None:
            n = self.config.get('start_wae', 0)
        if self.config["wae"]["use_wae_gan"]:
            for i in range(n):
                self.wae_mdp_worker._train_wae_gan()
        for i in range(n):
            self.wae_mdp_worker._train_wae_mdp()

    @time_memory
    def train(self,
              env_steps: Optional[int] = None,
              env_steps_per_epochs: Optional[int] = None,
              ):
        assert env_steps % env_steps_per_epochs == 0, f"env_steps ({env_steps}) should be a multiple of" \
                                                      f" env_steps_per_epoch ({env_steps_per_epochs})"
        logger.warning(f"Training over {env_steps} env steps in {env_steps // env_steps_per_epochs} epochs.")

        if not (bool(self.replay_buffer.index) or self.replay_buffer.full):
            self._pre_fill_replay_buffer()
        self.wae_mdp_worker.set_dataset(self.replay_buffer.as_dataset(self.config["wae"]["batch_size"]))
        self.belief_worker.set_maximizer_dataset()

        self._set_wae_gan_eval_dataset()

        self.wae_mdp_worker.pretrain_wae_gan_encoder()
        pretrain_steps = self.config.get('start_wae', 0)
        if pretrain_steps > 0:
            self.wae_mdp_worker.pretrain(pretrain_steps)

        self.wae_mdp_worker.n_train_called = 0
        self.wae_mdp_worker.wae_mdp._wae_gan_target_update = 0

        self._evaluate("Pre-training", n=self.config['evaluate_n'])
        logger.info("Resetting env step and training calls to 0.")

        epoch_stats_list = []

        start_env_step = self.env_step
        start_env_step_epoch = self.env_step

        training_max_cum_reward = float('-inf')
        training_max_cum_reward_undiscounted = float('-inf')
        eval_max_cum_reward = float('-inf')

        while self.env_step < start_env_step + env_steps:
            logger.info(f"Starting epoch {self._epoch + 1}/{env_steps // env_steps_per_epochs}.")
            epoch_time_manager = TimeManager("epoch")
            _train_step_time_manager = TimeManager("_train_step")
            step_stats = []
            step = 0
            with epoch_time_manager:
                while self.env_step < start_env_step_epoch + env_steps_per_epochs:
                    with _train_step_time_manager:
                        step_stats.append(self._train_step())
                    step += 1
            start_env_step_epoch += env_steps_per_epochs

            eval_stats = self._evaluate(title=f"Epoch {self._epoch + 1} - env step {self.env_step - start_env_step}",
                                        n=self.config['evaluate_n'])
            wae_gan_stats = self._evaluate_wae_gan()
            epoch_stats = combine_stats(step_stats)
            epoch_stats[self.wae_mdp_worker.__name__]["eval_wae_gan"] = wae_gan_stats
            epoch_stats[self.rl_worker.__name__]["evaluation_stats"] = eval_stats
            epoch_stats["time"]["epoch"] = epoch_time_manager.total_duration
            epoch_stats["info"] = {
                "epoch": self._epoch + 1,
                "experiment_name": self.experiment_name,
                "mem (GiB - vms)": round(get_mem(), 3),
                "env_steps (excluding pre-training)": self.rl_worker.env_steps - start_env_step,
                "timestamp": time.time() // 1000,
                "gan_target_update": self.wae_mdp_worker.wae_mdp.gan_target_update,
                **{f"{worker.__name__} train calls": worker.n_train_called for worker in self.workers}
            }
            if len(epoch_stats[self.rl_worker.__name__]["episode_ended_stats"]) == 0:
                epoch_stats[self.rl_worker.__name__]["episode_ended_stats"] = {}
            training_max_cum_reward = max(training_max_cum_reward,
                                          epoch_stats[self.rl_worker.__name__]["episode_ended_stats"].get("cum_reward",
                                                                                                          float(
                                                                                                              '-inf')))
            training_max_cum_reward_undiscounted = max(training_max_cum_reward_undiscounted,
                                          epoch_stats[self.rl_worker.__name__]["episode_ended_stats"].get("cum_reward_undiscounted",
                                                                                                          float(
                                                                                                              '-inf')))
            eval_max_cum_reward = max(eval_max_cum_reward,
                                      epoch_stats[self.rl_worker.__name__]["evaluation_stats"].get("cum_rewards",
                                                                                                   float('-inf')))
            epoch_stats["info"]["training_cum_reward_max"] = training_max_cum_reward
            epoch_stats["info"]["training_cum_reward_undiscounted_max"] = training_max_cum_reward_undiscounted
            if self.config['evaluate_n'] > 0:
                epoch_stats["info"]["eval_cum_reward_max"] = eval_max_cum_reward
            pprint(epoch_stats)
            epoch_stats_list.append(epoch_stats)
            with open(os.path.join(self.experiment_path, self.experiment_name, 'results.json'), 'a') as f:
                print(json.dumps(epoch_stats), file=f)
            if self.wandb_run is not None:
                while True:
                    try:
                        wandb.log(epoch_stats)
                        break
                    except wandb.Error as e:
                        logger.warning(f"wandb got error {e}")
                        time.sleep(60)
                    break
            self._epoch += 1
            self.make_checkpoint()
            if not self._epoch_stats_ok(epoch_stats):
                if not self.disable_early_stopping:
                    logger.critical("Epoch stats are not ok, stopping training")
                    raise EarlyStoppingException("Epoch stats are not ok, stopping training", epoch_stats_list)
                else:
                    logger.warning("Epoch stats are not ok, but continuing training because early stopping is disabled.")

        return epoch_stats_list

    def _has_nan(self, dict_):
        for k, v in dict_.items():
            if isinstance(v, dict):
                if self._has_nan(v):
                    return True
            elif v == np.nan:
                return True
        return False

    def _epoch_stats_ok(self, epoch_stats):
        marginal_state_encoder_entropy_ok = \
            epoch_stats[self.wae_mdp_worker.__name__]["marginal_state_encoder_entropy"] > 0.1
        has_nans = self._has_nan(epoch_stats)
        return marginal_state_encoder_entropy_ok and not has_nans

    def train_wae_mdp(self, n, ):
        stats = [{self.wae_mdp_worker.__name__: self.wae_mdp_worker.train()} for _ in range(n)]
        stats = combine_stats(stats)
        stats["info"] = {f"{self.wae_mdp_worker.__name__} train calls": self.wae_mdp_worker.n_train_called,
                         "epoch": self._epoch}
        if self.wandb_run is not None:
            while True:
                try:
                    wandb.log(stats)
                    break
                except wandb.Error as e:
                    logger.warning(f"wandb got error {e}")
                    time.sleep(60)
                break
        self._epoch += 1
        return stats

    def _train_step(self):
        _train_step_time_manager = TimeManager("_train_step")
        with _train_step_time_manager:
            step_stats = {'time': {}}
            for steps, worker in zip(self._round_robin_weights, self.workers):
                _step_stats = []
                with TimeManager() as time_manager:
                    for _ in range(steps):
                        stats_ = worker.train()
                        _step_stats.append(stats_)
                step_stats[worker.__name__] = _step_stats
                step_stats['time'][worker.__name__] = time_manager.mean_duration
        step_stats['time']['_train_step'] = _train_step_time_manager.total_duration
        return step_stats

    def _set_wae_gan_eval_dataset(self):
        if self.config["wae"]["use_wae_gan"]:
            self.wae_mdp_worker.set_evaluation_dataset(self.replay_buffer.sample('wae', 1024)[OBS])

    def make_checkpoint(self):
        if self.checkpoint_every > 0:
            if self._epoch % self.checkpoint_every == 0:
                checkpoint_dir = os.path.join(self.experiment_path, self.experiment_name,
                                              f"checkpoint_{self._epoch}")
                self.replay_buffer.checkpoint(checkpoint_dir=checkpoint_dir)
                for worker in self.workers:
                    worker.checkpoint(checkpoint_dir)

    def restore_from_checkpoint(self, checkpoint_dir):
        self.replay_buffer.restore(checkpoint_dir)
        for worker in self.workers:
            worker.restore(checkpoint_dir)

    def test(self, title: Optional[str] = None, n: int = 8):
        batch = self.replay_buffer.sample('rl', n)
        obs = batch["obs"]
        state = batch["state"]
        b, h, w, c = obs.shape
        latent_state = self.wae_mdp_worker.wae_mdp.relaxed_state_encoding(
            [tf.convert_to_tensor(batch["state"]), tf.convert_to_tensor(batch["obs"])],
            self.wae_mdp_worker.wae_mdp.state_encoder_temperature,
            tf.cast(tf.convert_to_tensor(batch['is_reset_state'][:, None]), tf.float32)).sample()

        decoded_state, decoded_obs = self.wae_mdp_worker.wae_mdp.decode_state(latent_state).sample()
        decoded_obs = array_to_np(decoded_obs)
        decoded_state = array_to_np(decoded_state)
        x = n // 2 // 4
        fig, ax = plt.subplots(n // 2, 4, figsize=(15, 15 * x))
        [axi.set_axis_off() for axi in ax.ravel()]
        for i in range(n // 2):
            ax[i, 0].imshow(batch['obs'][2 * i], vmin=0, vmax=1)
            ax[i, 2].imshow(batch['obs'][2 * i + 1], vmin=0, vmax=1)
            ax[i, 1].imshow(decoded_obs[2 * i], vmin=0, vmax=1)
            ax[i, 3].imshow(decoded_obs[2 * i + 1], vmin=0, vmax=1)
        if title is not None:
            fig.suptitle(title, fontsize=30)

        fig.tight_layout()
        fig.subplots_adjust(top=.95)
        fig.savefig(os.path.join(self.experiment_path,
                                 self.experiment_name,
                                 f"epoch_{self._epoch + 1}_obs.png"))
        if self.config["show_plots"]:
            fig.show()

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.repeat(np.repeat(state, 4, 0), 2, 1), vmin=0, vmax=1)
        ax[1].imshow(np.repeat(np.repeat(decoded_state, 4, 0), 2, 1), vmin=0, vmax=1)
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        fig.suptitle(f"{title} - states")
        fig.tight_layout()
        fig.subplots_adjust(top=.95)
        fig.savefig(os.path.join(self.experiment_path,
                                 self.experiment_name,
                                 f"epoch_{self._epoch + 1}_state.png"))
        if self.config["show_plots"]:
            fig.show()

        return batch['state'], decoded_state

    @time_memory
    def _evaluate_wae_gan(self):
        evaluate_dict = dict_array_to_dict_python(self.wae_mdp_worker._wae_mdp.evaluate_gan())
        self._set_wae_gan_eval_dataset()
        return evaluate_dict

    @time_memory
    def _evaluate(self, title: Optional[str] = None, n: int = 10):
        # if len(self.replay_buffer.elements_scheme[OBS].shape) == 3:
        #     self.test(title)
        # make that a parameter
        if n > 0:
            eval_stats = self.rl_worker.evaluate(n)
            return {key: np.mean([stat[key] for stat in eval_stats]).item()
                    for key in eval_stats[0]}
        if self.config["policy"].get("learn_with_img", False):
            episode_str = [self.belief_a2c_worker.a2c_worker.evaluate_img() for _ in range(5)]
            with open(os.path.join(self.experiment_path, self.experiment_name, f"img_episode_epoch_{self._epoch + 1}"),
                      'w') as f:
                print("\n#######\n\n".join(episode_str), file=f)
        return {}

    @property
    def env_step(self):
        return self.rl_worker.env_steps

    def close(self):
        if not is_debug():
            wandb.finish()

    def final_tsne(self, n: int = int(1e6)):
        datas = []
        steps = math.ceil(n / self.replay_buffer.policy_capacity)
        for _ in range(steps):
            self.rl_worker.interact(self.replay_buffer.policy_capacity // self.rl_worker.nbr_env, training=False,
                                    force_random_action=False, fill_replay_buffer=True)
            data = self.replay_buffer.sample('rl', self.replay_buffer.policy_capacity)
            data["value"] = self.belief_a2c_worker.a2c_worker.value_model(data[SUB_BELIEF], training=False).numpy().flatten()
            datas.append(data)
        data = {key: np.concatenate([d[key] for d in datas], axis=0) for key in datas[0]}
        with open(os.path.join(self.experiment_path, self.experiment_name,
                               f"tsne_{self._epoch + 1}_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"),
                  'wb') as f:
            pickle.dump(data, f)

