import numpy as np
import tensorflow as tf
import gym
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


class RunningMeanTF():
    def __init__(self, size, shape):
        self.size = size
        self.shape = shape
        self._queue = tf.queue.FIFOQueue(capacity=self.size, dtypes=tf.float32, shapes=self.shape)
        self._sum = tf.get_variable(dtype=tf.float32, shape=self.shape, initializer=tf.constant_initializer(0.0),
                                    name='runningsum',
                                    trainable=False)
        self._sumsq = tf.get_variable(dtype=tf.float32, shape=self.shape, initializer=tf.constant_initializer(0.0),
                                     name='runningsumsq',
                                     trainable=False)
        self.count = 1
        if self.count < self.size:
            self.mean = self._sum / self.count
            self.std = tf.sqrt(tf.maximum((self._sumsq / self.count) - tf.square(self.mean), 1e-2))
        else:
            self.mean = self._sum / self.size
            tf.sqrt(tf.maximum((self._sumsq / self.size) - tf.square(self.mean), 1e-2))
        # self.mean = tf.cond(self.count == 0, 0.0, self._sum / self.count)
        # self.std = tf.cond(self.count == 0, 0.0, tf.sqrt(tf.maximum((self._sumsq / self.count) - tf.square(self.mean), 1e-2)))

        self.new_val = tf.placeholder(shape=self.shape, dtype=tf.float32, name='new_val')
        self.new_valsq = tf.placeholder(shape=self.shape, dtype=tf.float32, name='new_val_sq')
        self.old_val = tf.placeholder(shape=self.shape, dtype=tf.float32, name='old_val')
        self.old_valsq = tf.placeholder(shape=self.shape, dtype=tf.float32, name='old_val_sq')

        self.new_sum = tf.assign_sub(tf.assign_add(self._sum, self.new_val), self.old_val)
        self.new_sumsq = tf.assign_sub(tf.assign_add(self._sumsq, self.new_valsq), self.old_valsq)
        self.enque = self._queue.enqueue(self.new_val)
        # self.deque = self._queue.dequeue()

    def update(self, x, old_x=None):
        if old_x is None:
            tf.get_default_session().run([self.new_sum, self.new_sumsq, self.enque], feed_dict={self.new_val: x,
                                                                                                self.new_valsq: np.square(x),
                                                                                                self.old_val: np.zeros_like(x),
                                                                                                self.old_valsq: np.zeros_like(x)})
        else:
            tf.get_default_session().run([self.new_sum, self.new_sumsq, self.enque],
                                         feed_dict={self.new_val: x,
                                                    self.new_valsq: np.square(x),
                                                    self.old_val: old_x,
                                                    self.old_valsq: np.square(old_x)})

    def deque(self):
        return tf.get_default_session().run(self._queue.dequeue())

    def deque_all(self):
        return tf.get_default_session().run(self._queue.dequeue_many(self.size))

    def qlength(self):
        return tf.get_default_session().run(self._queue.size())

    def stats(self):
        return tf.get_default_session().run([self.mean, self.std])

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    ob_space = env.observation_space
    init_obs = env.reset()
    inputs = tf.placeholder(name='ob', dtype=tf.float32, shape=[1, ob_space.shape[0]])
    # test_outputs = tf.get_variable(dtype=tf.float32, shape=[1, inputs.shape[1]], name='test_output')
    ob_rms = RunningMeanTF(size=10, shape=inputs.get_shape())
    # acc_output = tf.assign_add(test_outputs, inputs)
    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        # print(sess.run(test_outputs, feed_dict={inputs: init_obs}))
        sess.run(init_op)
        # print('initialized outputs: ', sess.run(test_outputs))
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # ob_rms.update(obs.reshape(inputs.get_shape()))

            print('current obs at {}'.format(i))
            print(obs)
            if ob_rms.count < ob_rms.size:
                ob_rms.update(obs.reshape(inputs.get_shape()))
            else:
                old_val = ob_rms.deque()
                ob_rms.update(obs.reshape(inputs.get_shape()), old_x=old_val)

            print('current queue size {}'.format(ob_rms.qlength()))
            print('current queue ')
            # acc_result = sess.run(acc_output, feed_dict={inputs: obs.reshape(inputs.shape)})



        # print('current obs: ', obs)
        # print(sess.run(acc_output, feed_dict={inputs: init_obs.reshape(inputs.shape)}))

    # test_policy = Policy(name='policy', ob=inputs, ob_length=10, hid_size=32, num_hid_layers=2, num_subpolicies=3)
