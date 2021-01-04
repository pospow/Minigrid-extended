import tensorflow as tf
import numpy as np
import gym
from minigrid.statistics.running_mean import RunningMean, RunningMeanTF
from minigrid.util.tf_util import *


class Policy():
    def __init__(self, name, ob, ob_length, hid_size, num_hid_layers, num_subpolicies, gaussian_fixed_var=True):
        self.hid_size = hid_size
        self.num_hid_layers = num_hid_layers
        self.num_subpolicies = num_subpolicies
        self.gaussian_fixed_var = gaussian_fixed_var

        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            with tf.variable_scope('obfilter'):
                self.ob_rms = RunningMeanTF(size=ob_length, shape=ob.shape[1])

            for i in range(ob.shape[1]):
                number = np.random.rand()
                self.ob_rms.update(number)
            obz = tf.clip_by_value(self.ob_rms.get_normalized_by_gaussian(), -5.0, 5.0)

            # value function
            last_out = tf.expand_dims(tf.to_float(obz), axis=0)
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(dense(last_out, hid_size, 'vffc%i' % (i + 1), weight_init=normc_initializer(1.0)))
            self.vpred = dense(last_out, 1, 'vffinal', weight_init=normc_initializer(1.0))[:, 0]

    def run_output(self):
        with tf.Session() as sess:
            print(sess.run(self.vpred, feed_dict={}))


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ob_space = env.observation_space
    inputs = tf.placeholder(name='ob', dtype=tf.float32, shape=[None, ob_space.shape[0]])
    test_policy = Policy(name='policy', ob=inputs, ob_length=10, hid_size=32, num_hid_layers=2, num_subpolicies=3)
