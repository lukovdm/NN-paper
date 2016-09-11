from utility import *

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("MINI_BATCH_SIZE", 100, "size of the mini-batch")
flags.DEFINE_integer("TEST_STEPS", 10, "amount of steps before testing")


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="data")
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("conv1"):
    conv1 = conv_layer(x, [5, 5, 1, 50], [1, 1, 1, 1], "VALID")

with tf.name_scope("conv2"):
    conv2 = conv_layer(conv1, [5, 5, 50, 80], [1, 1, 1, 1], "VALID")

with tf.name_scope("fully_connected"):
    conv2_reshapen = tf.reshape(conv2, [None, 720])  # TODO: dubble check 720
    fully_connected1 = nn_layer(conv2_reshapen, 720, 500)
    dropout1 = tf.nn.dropout(fully_connected1, keep_prob)

with tf.name_scope("output_layer"):
    y = nn_layer(dropout1, 720, 10, act=tf.nn.softmax)
