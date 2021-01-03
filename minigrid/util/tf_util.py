import numpy as np
import tensorflow as tf
import copy
import os
import collections


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + '/w', [x.shape[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + '/b', [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret
