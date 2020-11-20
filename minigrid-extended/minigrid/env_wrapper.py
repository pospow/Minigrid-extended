import gym
import numpy as np
from gym import spaces
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

max_env_steps = 50


class FlatObsWrapper(gym.core.ObservationWrapper):
    """Fully observable gridworld return in a flat grid encoding"""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=0, high=255, shape=((self.env.width - 2) * (self.env.height - 2) * 3,),
                                            dtype='uint8')
        self.unwrapped.max_steps = max_env_steps

    def observation(self, obs):
        # this method is called in the step() function to get the observation
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([OBJECT_TO_IDX['agent'],
                                                                  COLOR_TO_IDX['red'],
                                                                  env.agent_dir])
        full_grid = full_grid[1:-1, 1:-1]

        flattened_grid = full_grid.flatten()

        return flattened_grid

    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view"""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)

if __name__ == "__main__":
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env.reset()

    wrapped_env = FlatObsWrapper(env)
    action = wrapped_env.actions.forward
    wrapped_obs, reward, done, info = wrapped_env.step(action)

    img = wrapped_env.render('rgb_array')
