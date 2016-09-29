from utility import *

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("MINI_BATCH_SIZE", 16, "size of the mini-batch")
flags.DEFINE_integer("TEST_STEPS", 100, "amount of steps before testing")
flags.DEFINE_string("summaries_dir", "/tmp/NN-summaries", "location of summaries")
flags.DEFINE_integer("max_steps", 100, "the maximum amount of steps taken")


if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="data")
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    tf.image_summary('input', x_reshaped, max_images=10)
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
    keep_prob = tf.placeholder(tf.float32)

    mnist_data_set = mnist.read_data_sets("Data", one_hot=True)

with tf.name_scope("conv1"):
    conv1 = conv_layer(x_reshaped, [5, 5, 1, 64], [1, 1, 1, 1], "VALID", "conv1")

with tf.name_scope("conv2"):
    conv2 = conv_layer(conv1, [5, 5, 64, 128], [1, 1, 1, 1], "VALID", "conv2")

with tf.name_scope("fully_connected"):
    conv2_reshaped = tf.reshape(conv2, [-1, 20 * 20 * 128])
    fc_1 = nn_layer(conv2_reshaped, 20 * 20 * 128, 1024, "fc_1")
    fc_1_dropout = tf.nn.dropout(fc_1, keep_prob)

with tf.name_scope("output_layer"):
    y = nn_layer(fc_1_dropout, 1024, 10, "output", act=tf.nn.softmax)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.scalar_summary("loss", loss)

with tf.name_scope("train"):
    trainer = tf.train.AdamOptimizer()
    train_step = trainer.minimize(loss)

with tf.name_scope("accuracy"):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)


def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist_data_set.train.next_batch(100)
        k = FLAGS.dropout
    else:
        xs, ys = mnist_data_set.test.images, mnist_data_set.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')

    sess.run(tf.initialize_all_variables())

    for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                print('ran step ' + str(i))
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
