from utility import *

import tensorflow as tf
import numpy as np
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("MINI_BATCH_SIZE", 16, "size of the mini-batch")
flags.DEFINE_integer("TEST_STEPS", 100, "amount of steps before testing")
flags.DEFINE_string("summaries_dir", "/tmp/NN-summaries", "location of summaries")
flags.DEFINE_integer("max_steps", 10000, "the maximum amount of steps taken")
flags.DEFINE_float("dropout", 0.5, "the dropout")


if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

keep_prob = tf.placeholder(tf.float32, name="keep_probability")

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="data")

with tf.name_scope("conv1"):
    conv1 = conv_layer(x, [5, 5, 1, 128], [1, 1, 1, 1], "VALID", "conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    pool1_reshaped = tf.reshape(pool1, (-1, 12, 12, 1))
    tf.image_summary("conv1", pool1_reshaped, max_images=10)

with tf.name_scope("conv2"):
    conv2 = conv_layer(pool1, [5, 5, 128, 256], [1, 1, 1, 1], "VALID", "conv2")
    conv2_reshaped = tf.reshape(conv2, (-1, 8, 8, 1))
    tf.image_summary("conv2", conv2_reshaped, max_images=10)

with tf.name_scope("conv3"):
    conv3 = conv_layer(conv2, [3, 3, 256, 256], [1, 1, 1, 1], "VALID", "conv3")
    conv3_img = tf.reshape(pool1, (-1, 6, 6, 1))
    tf.image_summary("conv3", conv3_img, max_images=10)

with tf.name_scope("fc_1"):
    conv3_reshaped = tf.reshape(conv3, [-1, 6 * 6 * 256])
    fc_1 = nn_layer(conv3_reshaped, 6 * 6 * 256, 1024, "fc_1")
    fc_1_dropout = tf.nn.dropout(fc_1, keep_prob)

with tf.name_scope("output"):
    y = nn_layer(fc_1_dropout, 1024, 10, "output", act=tf.nn.softmax)

saver = tf.train.Saver()


with tf.Session() as sess:
    saver.restore(sess, "MNIS2.ckpt")
    print("model restored")

    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/run', sess.graph)

    image = np.array(Image.open("Data/4.png").convert('L')).reshape(1, 28, 28, 1)
    out, summary = sess.run([y, merged], feed_dict={x: image, keep_prob: 1.0})
    summary_writer.add_summary(summary)
    print(out)
    PIL_img = np.squeeze(image)
    #Image.fromarray(PIL_img, 'L').show()

    summary_writer.close()




