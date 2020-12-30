import glob
import io
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import os

np.set_printoptions(precision=2, suppress=True)

import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def plot_grid_std_learning_curves(d, num_it):
    for i, key in enumerate(d):
        ax = plt.subplot(2, 2, i + 1)
        rewards, success_rates = d[key]
        plot_std_learning_curves(rewards, success_rates, num_it, no_show=True)
        ax.set_title(key)
    plt.show()


def plot_std_learning_curves(rewards, success_rates, num_it, dir_path, file_name, no_show=False):
    r, sr = np.asarray(rewards), np.asarray(success_rates)
    df = pd.DataFrame(r).melt()
    sns.lineplot(x="variable", y="value", data=df, label='reward/eps')
    df = pd.DataFrame(sr).melt()
    sns.lineplot(x="variable", y="value", data=df, label='success rate')
    plt.xlabel("Training iterations")
    plt.ylabel("")
    plt.xlim([0, num_it])
    plt.ylim([0, 1])
    plt.legend()
    plt.grid('on')
    if not no_show:
        # plt.show()
        file_name = os.path.join(dir_path, file_name)
        plt.savefig(file_name)


def plot_learning_curve(rewards, success_rates, num_it, dir_path, file_name, plot_std=False):
    if plot_std:
        plot_std_learning_curves(rewards, success_rates, num_it, dir_path, file_name)
    else:
        plt.plot(rewards, label='reward/eps')
        if success_rates:
            plt.plot(success_rates, label='success rate')
            plt.legend()
        else:
            plt.ylabel('return / eps')
        plt.ylim([0, 1])
        plt.xlim([0, num_it - 1])
        plt.xlabel('train iter')
        plt.grid('on')

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = os.path.join(dir_path, file_name)

        plt.savefig(file_name)
        plt.close()


class ParamDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getsstate__(self):
        return self

    def __setstate__(self, d):
        self = d
