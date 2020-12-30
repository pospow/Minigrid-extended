import numpy as np
import torch
import gym
from minigrid.utils import AverageMeter, plot_learning_curve
from minigrid.utils import ParamDict
from minigrid.rollout_buffer import RolloutStorage
from minigrid.env_wrapper import FlatObsWrapper
from minigrid.policy.base_policy import Policy
from minigrid.window import Window
from minigrid.rollout_play import *
from minigrid.minigrid_utils import DeterministicCrossingEnv, DetHardSubgoalCrossingEnv, DetHardEuclidCrossingEnv, \
    HardSubgoalCrossingEnv, DetHardOptRewardCrossingEnv
import copy
import time
import os
import warnings
from datetime import datetime
from tqdm import tqdm


def train(env, rollouts, policy, params, ckpt_path, seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    rollout_time, update_time = AverageMeter(), AverageMeter()
    rewards, success_rates = [], []

    print("Training model with {} parameters...".format(policy.num_params))

    for j in range(params.num_updates):

        window = Window(params.env_name)

        avg_eps_reward, avg_success_rate = AverageMeter(), AverageMeter()
        done = False
        prev_obs = env.reset()
        prev_obs = torch.tensor(prev_obs, dtype=torch.float32)

        eps_reward = 0.
        start_time = time.time()

        # Keep rolling out (generating data) for rollout_size steps given the current policy (no training)
        for step in tqdm(range(rollouts.rollout_size)):
            # Rollout using current on-policy (fixed)
            if done:
                # Update the average reward and success rate after each episode
                avg_eps_reward.update(eps_reward)
                if 'success' in info:
                    avg_success_rate.update(int(info['success']))

                obs = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
                eps_reward = 0.
            else:
                obs = prev_obs

            action, log_prob = policy.act(obs)
            obs, reward, done, info = env.step(action)

            done = torch.tensor(done, dtype=torch.float32)
            reward = torch.tensor(reward, dtype=torch.float32)
            rollouts.insert(step, done, action, log_prob, reward, prev_obs)

            prev_obs = torch.tensor(obs, dtype=torch.float32)

            eps_reward += reward

        rollouts.compute_returns(params.discount)
        rollout_done_time = time.time()

        # Update the policy network after one rollout
        policy.update(rollouts)

        update_done_time = time.time()

        # Displays one episode of freshly trained policy
        updated_policy_rollout(policy, env, window, j)

        rollouts.reset()

        rewards.append(avg_eps_reward.avg)

        if avg_success_rate.count > 0:
            success_rates.append(avg_success_rate.avg)
        rollout_time.update(rollout_done_time - start_time)
        update_time.update(update_done_time - rollout_done_time)
        print('it {}: avgR: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, avg_eps_reward.avg,
                                                                                                rollout_time.avg,
                                                                                                update_time.avg))

    file_name = policy.name + '_lr_' + str(policy_params.learning_rate) + '_bat_' + str(
        policy_params.batch_size) + '_ent_' + str(policy_params.entropy_coef) + '_seed_' + str(seed) + '.png'
    plot_learning_curve(rewards, success_rates, params.num_updates, ckpt_path, file_name)

    torch.save(policy.actor.state_dict(), ckpt_path + '/trained_actor_' + str(seed) + '.pt')

    return rewards, success_rates


def instantiate(params_in, nonwrapped_env=None):
    params = copy.deepcopy(params_in)

    if nonwrapped_env is None:
        nonwrapped_env = gym.make(params.env_name)

    env = FlatObsWrapper(nonwrapped_env)
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    rollouts = RolloutStorage(params.rollout_size, obs_size)
    policy_class = params.policy_params.pop('policy_class')

    policy = policy_class(obs_size, num_actions, **params.policy_params)
    return env, rollouts, policy


if __name__ == "__main__":
    train_policy = True
    n_seeds = np.random.randint(20, size=(5,))
    # hyperparameters
    policy_params = ParamDict(
        policy_class=Policy,  # Policy class to use (replaced later)
        hidden_dim=32,  # dimension of the hidden state in actor network
        learning_rate=1e-4,  # learning rate of policy update
        batch_size=512,  # batch size for policy update
        policy_epochs=4,  # number of epochs per policy update
        entropy_coef=0.005,  # hyperparameter to vary the contribution of entropy loss
    )
    params = ParamDict(
        policy_params=policy_params,
        rollout_size=2050,  # number of collected rollout steps per policy update
        num_updates=100,  # number of training policy iterations
        discount=0.99,  # discount factor
        plotting_iters=10,  # interval for logging graphs and policy rollouts
        env_name='DetHardSubgoalCrossingEnv',  # we are using a tiny environment here for testing
    )

    if train_policy:
        rewards_arr, success_rates_arr = [], []
        ckpt_path = os.path.join(os.getcwd(), 'log_') + str(
            datetime.now().strftime('%H%M%S')) + '_' + params.env_name
        for i in n_seeds:
            print('Start training run {}'.format(i))

            env, rollouts, policy = instantiate(params, nonwrapped_env=DetHardOptRewardCrossingEnv())

            r, sr = train(env, rollouts, policy, params, ckpt_path, seed=int(i))

            rewards_arr.append(r)
            success_rates_arr.append(sr)
            print("End training run {}".format(i))

        print('All training runs completed!')
        plot_learning_curve(rewards_arr, success_rates_arr, params.num_updates, ckpt_path, 'std.png', plot_std=True)

    else:
        ckpt_path = os.path.join(os.getcwd(), 'log_') + '144925' + '_' + params.env_name
        assert os.path.exists(ckpt_path), 'No such directory {}'.format(ckpt_path)
        ckpt_file = os.path.join(ckpt_path, 'trained_actor') + '_17.pt'
        assert os.path.exists(ckpt_file), 'No such ckpt file {}'.format(ckpt_file)
        env, rollouts, policy = instantiate(params, nonwrapped_env=DetHardEuclidCrossingEnv())
        window = Window(params.env_name)
        loaded_policy_rollout(env, policy, window, ckpt_file)
