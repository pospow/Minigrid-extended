import pytest
import numpy as np
from minigrid.statistics.running_mean import RunningMean


def test_running_mean():
    test_rm = RunningMean(size=10, is_state=True, episode_steps=200)
    for i in range(10):
        test_rm.update(np.random.rand())
    assert test_rm.count == 10

# if __name__ == '__main__':
#     test_rm = RunningMean(size=10, is_state=True, episode_steps=200)
#     for i in range(10):
#         test_rm.update(np.random.rand())
#     print(a)