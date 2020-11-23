import gym
import gym_minigrid
from env_wrapper import FlatObsWrapper


if __name__ == '__main__':
    env = FlatObsWrapper(gym.make('MiniGrid-Empty-8x8-v0'))

    env.reset()

    action = env.actions.forward

    obs, reward, done, info = env.step(action)

    img = env.render('rgb_array')

    print('Observation:', obs)
    print('Reward: ', reward)
    print('Done: ', done)
    print('Info: ', info)
    print('Image shape: ', img.shape)