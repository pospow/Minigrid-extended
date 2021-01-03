import numpy as np
# import tensorflow as tf
# from collections import deque
import gym
from statistics.running_mean import RunningMean

if __name__ == '__main__':
    # print(np.random.rand())
    # a = deque()
    # a.append(1.2)
    # a.append(3.4)
    # a.append(2.9)
    # print(np.mean(a))
    # print(np.square(a - 3.2))


    test_rm = RunningMean(size=10, is_state=True, episode_steps=200)
    for i in range(10):
        number = np.random.rand()
        print(number)
        test_rm.update(number)
    # print(test_rm.get_normalized_by_rms())
    # print(test_rm.get_normalized_by_mean())
    print(test_rm.get_normalized_by_gaussian())


    # env = gym.make('CartPole-v0')
    # env.reset()
    # print(env.action_space)