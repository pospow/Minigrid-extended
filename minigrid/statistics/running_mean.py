import numpy as np
from collections import deque


class RunningMean():
    '''Computes and stores the running average of the values'''

    def __init__(self, size=None, is_state=False, episode_steps=None):
        self.is_state = is_state
        self.episode_steps = episode_steps
        self.reset()
        if size is None:
            self.size = 10
        else:
            self.size = size

    def reset(self):
        self.values = deque()
        self.avg = 0.0
        self.prev_avg = 0.0
        self.rms = 0.0
        self.prev_rms = 0.0
        self.sum = 0.0
        self.var = 0.0
        self.std = 0.0
        self.count = 0
        if self.is_state:
            assert self.episode_steps is not None, 'Please specify the total number of steps for the episode'
            self.obs_arr = np.zeros(
                self.episode_steps)  # For states, we need to record every data for later use (find max, plotting graphs)
            self.sqsum_arr = np.zeros(self.episode_steps)  # For calculation of rms values
        else:
            self.obs_arr = None
            self.sqsum_arr = None
        self.obs_max = 0.0

    def update(self, new_val):
        self.count += 1
        if self.is_state:  # record data in array for every time step
            if self.obs_max < abs(new_val):
                self.obs_max = new_val
            self.obs_arr[self.count - 1] = new_val
            if self.count == 1:
                self.sqsum_arr[self.count - 1] = new_val * new_val
            else:
                self.sqsum_arr[self.count - 1] = self.sqsum_arr[self.count - 2] + new_val * new_val
            self.prev_rms = self.rms
            self.rms = self._calc_rms()

        self.prev_avg = self.avg

        self.values.append(new_val)
        if self.count > self.size:
            self.sum -= self.values.popleft()
            n = self.size
        else:
            n = self.count
        self.sum += new_val
        self.avg = np.mean(self.values)
        self.var = np.sum(np.square(self.values - self.avg)) / n
        self.std = np.sqrt(self.var)

    def get_current(self):
        return self.obs_arr[self.count - 1]

    def get_normalized_by_rms(self):
        return self.obs_arr[self.count - self.size: self.count] / self.rms

    def get_normalized_by_mean(self):
        return self.obs_arr[self.count - self.size: self.count] / self.avg

    def get_normalized_by_gaussian(self):
        return (self.obs_arr[self.count - self.size: self.count] - self.avg) / self.std

    def _calc_rms(self):
        if self.count < self.size:
            return np.sqrt(np.sum(self.sqsum_arr[:self.count]) / self.count)
        else:
            return np.sqrt(np.sum(self.sqsum_arr[self.count - self.size: self.count]) / self.size)
