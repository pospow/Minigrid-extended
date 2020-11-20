import numpy as np
import torch
import gym
from IPython.display import clear_output
from utils import AverageMeter, plot_learning_curve
from plot_video import log_policy_rollout
from utils import ParamDict
from rollout_buffer import RolloutStorage
from env_wrapper import FlatObsWrapper
from policy import Policy
import copy
import time


def train(env, rollouts, policy, params, seed=123):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    rollout_time, update_time = AverageMeter(), AverageMeter()
    rewards, success_rate = [], []

    print('Training model with {} parameters...'.format(policy.num_params))

    for j in range(params.num_updates):
        avg_eps_reward, avg_success_rate = AverageMeter(), AverageMeter()
        done = False
        prev_obs = env.reset()
        prev_obs = torch.tensor(prev_obs, dtype=torch.float32)
        eps_reward = 0.0
        start_time = time.time()

        for step in range(rollouts.rollout_size):
            if done:
                avg_eps_reward.update(eps_reward)
                if 'success' in info:
                    avg_success_rate.update(int(info['success']))

                obs = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32)
                eps_reward = 0.0

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

        policy.update(rollouts)

        update_done_time = time.time()
        rollouts.reset()

        rewards.append(avg_eps_reward)

        if avg_success_rate.count > 0:
            success_rate.append(avg_success_rate.avg)
        rollout_time.update(rollout_done_time - start_time)
        update_time.update(update_done_time - rollout_done_time)
        print('it {}: avgR: {:.3f} -- rollout_time: {:.3f}sec -- update_time: {:.3f}sec'.format(j, avg_eps_reward.avg,
                                                                                                rollout_time.avg,
                                                                                                update_time.avg))

        if j % params.plotting_iters == 0 and j != 0:
            plot_learning_curve(rewards, success_rate, params.num_updates)
            log_policy_rollout(policy, params.env_name, pytorch_policy=True)
        clear_output()
        return rewards, success_rate


def instantiate(params_in, nonwrapped_env=None):
    params = copy.deepcopy(params_in)

    if nonwrapped_env is None:
        nonwrapped_env = gym.make(params.env_name)

    env = None
    env = FlatObsWrapper(nonwrapped_env)
    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    rollouts = RolloutStorage(params.rollout_size, obs_size)
    policy_class = params.policy_params.pop('policy_class')

    policy = policy_class(obs_size, num_actions, **params.policy_params)
    return env, rollouts, policy


if __name__ == "__main__":
    # hyperparameters
    policy_params = ParamDict(
        policy_class=Policy,  # Policy class to use (replaced later)
        hidden_dim=32,  # dimension of the hidden state in actor network
        learning_rate=1e-3,  # learning rate of policy update
        batch_size=1024,  # batch size for policy update
        policy_epochs=4,  # number of epochs per policy update
        entropy_coef=0.001,  # hyperparameter to vary the contribution of entropy loss
    )
    params = ParamDict(
        policy_params=policy_params,
        rollout_size=2050,  # number of collected rollout steps per policy update
        num_updates=50,  # number of training policy iterations
        discount=0.99,  # discount factor
        plotting_iters=10,  # interval for logging graphs and policy rollouts
        env_name='MiniGrid-Empty-5x5-v0',  # we are using a tiny environment here for testing
    )

    env, rollouts, policy = instantiate(params)
    rewards, success_rate = train(env, rollouts, policy, params)
    print("Training completed!")
