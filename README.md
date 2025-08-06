# The Wasserstein Believer
### Learning Belief Updates for Partially Observable Environments through Reliable Latent Space Models

ICLR 2024 paper: https://openreview.net/forum?id=KrtGfTGaGe

Wasserstein Belief Updater (WBU) is an **RNN free RL algorithm** for POMDPs that learns a representation of the history via an approximation of the belief update in a reliable latent space model, providing theoretical guarantees for learning the optimal value.

This work concerns agents learning how to behave, i.e., their control policy, through **reinforcement learning** (RL).
In real-world scenarios, the environment's state is very often perceived either through noisy sensors, cameras, or more geneally imperfect observations (e.g., visual observation vs. exact coordinates on a map). In that case, the observation is *non-Markovian* and the environment is *partially observable*.
This usually leads to complications compared to theoretical perfect-observation RL (i.e., with Markovian observation).
For optimal decision making, the agent must in that case base its decision either on (a) the full observation-action history, or (b) the distribution over the possible real states of the environment in which the agent could be at each time step. The latter is called the **belief** of the agent and is a **sufficient statistic** to optimize the agent's return.

The easiest method to tackle partial observability is to process the full history through an RNN to obain a compressed hidden state that can be fed to the policy of the learning agent. While appealing, RNNs don't yield any guarantee that the **representation** learned is actually useful (a sufficient statistic) to optimize the agent's return.

With WBU, we rather propose to learn a **representation of the belief**. Belief learning is difficult in RL because (1) the dynamics of the environments must be known to exactly compute the belief, and (2) it does not scale as it requires to integrate over the full state space (usually intractable).
To tackle those challenges, WBU
1. learns a world model, through [Wasserstein auto-encoded MDPs](https://github.com/florentdelgrange/wae_mdp).
This model comes with **theoretical abstraction quality guarantees**.
It is learned through **discrete latent spaces** which eases the computation of the belief through the latent space.
2. minimizes the discrepancy between the theoretical belief update rule and the latent belief computed. This yields **theoretical representation quality guarantees**: close points in the representation space of the beliefs are guaranteed to yield close expected returns (Lipschitz continuity). This guarantees to support policy learning.

<p align="center">
  <img src="wbu.png" alt="Wasserstein Belief Updater" width=75% />
</p>

## Installation
**Warning:** This code was tested on a linux environment with Python3.9
```shell
python3.9 -m venv venv
source venv/bin/activate
pip install --no-deps -r requirements.txt
pip install --no-deps -e modules/popgym
pip install --no-deps -e modules/POMinAtar
pip install --no-deps -e modules/bwu
```

## Usage

### Repeat Previous

```shell
python run.py --config modules/bwu/belief_learner/config/repeat_previous.toml
```
### Stateless Cartpole

```shell
python run.py --config modules/bwu/belief_learner/config/stateless_cartpole.toml
```

### Noisy Stateless Cartpole

```shell
python run.py --config modules/bwu/belief_learner/config/noisy_stateless_cartpole.toml
```

### Space Invaders

```shell
python run.py --config modules/bwu/belief_learner/config/space_invaders.toml --timesteps 2e6
```

### Noisy Space Invaders

```shell
python run.py --config modules/bwu/belief_learner/config/noisy_space_invaders.toml --timesteps 2e6
```

## Results

![results](./results.png)

## Cite
If you use this code, please cite it as:
```
@inproceedings{
avalosdelgrange2024wbu,
title={The Wasserstein Believer: Learning Belief Updates for Partially Observable Environments through Reliable Latent Space Models},
author={Rapha{\"e}l Avalos and Florent Delgrange and Ann Nowe and Guillermo Perez and Diederik M Roijers},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=KrtGfTGaGe}
}
```

## Acknowledgements

- MinAtar: https://github.com/kenjyoung/MinAtar
- PopGym: https://github.com/proroklab/popgym
