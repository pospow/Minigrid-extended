from gym.wrappers import Monitor
from env_wrapper import FlatObsWrapper
from utils import show_video
import torch
import gym


def wrap_env(env):
    env = Monitor(env, './video', force=True)
    return env


def gen_wrapped_env(env_name):
    return wrap_env(FlatObsWrapper(gym.make(env_name)))


class RandPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, *unsued_args):
        return self.action_space.sample(), None


# This function plots cideos of rollouts (episodes) of a given policy and environment

def log_policy_rollout(policy, env_name, pytorch_policy=False):
    env = gen_wrapped_env(env_name)

    observation = env.reset()

    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        if pytorch_policy:
            observation = torch.tensor(observation, dtype=torch.float32)
            action = policy.act(observation)[0].data.cpu().numpy()
        else:
            action = policy.act(observation)[0]
        observation, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1

    print('Total reward: ', episode_reward)
    print('Total length: ', episode_length)

    env.close()

    show_video()

if __name__ == "__main__":
    test_env_name = 'MiniGrid-Empty-8x8-v0'
    rand_policy = RandPolicy(FlatObsWrapper(gym.make(test_env_name)).action_space)
    log_policy_rollout(rand_policy, test_env_name)
