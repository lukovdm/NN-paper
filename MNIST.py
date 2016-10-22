from utility import *

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("MINI_BATCH_SIZE", 64, "size of the mini-batch")
flags.DEFINE_integer("TEST_STEPS", 100, "amount of steps before testing")
flags.DEFINE_string("summaries_dir", "/tmp/NN-graph", "location of summaries")
flags.DEFINE_integer("max_steps", 5000, "the maximum amount of steps taken")
flags.DEFINE_float("dropout", 0.5, "the dropout")


if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
tf.gfile.MakeDirs(FLAGS.summaries_dir)

y_ = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
keep_prob = tf.placeholder(tf.float32, name="keep_probability")

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 784], name="data")
    x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
    tf.image_summary('input', x_reshaped, max_images=10)
    mnist_data_set = mnist.read_data_sets("Data", one_hot=True)

with tf.name_scope("conv1"):
    conv1 = conv_layer(x_reshaped, [5, 5, 1, 128], [1, 1, 1, 1], "VALID", "conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

with tf.name_scope("conv2"):
    conv2 = conv_layer(pool1, [5, 5, 128, 256], [1, 1, 1, 1], "VALID", "conv2")

with tf.name_scope("conv3"):
    conv3 = conv_layer(conv2, [3, 3, 256, 256], [1, 1, 1, 1], "VALID", "conv3")

with tf.name_scope("fc_1"):
    conv3_reshaped = tf.reshape(conv3, [-1, 6 * 6 * 256])
    fc_1 = nn_layer(conv3_reshaped, 6 * 6 * 256, 1024, "fc_1")
    fc_1_dropout = tf.nn.dropout(fc_1, keep_prob)

with tf.name_scope("output"):
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

with tf.name_scope("saving"):
    saver = tf.train.Saver()


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
                print('running step ' + str(i))
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

        test_writer.flush()
        train_writer.flush()

    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, FLAGS.max_steps+1)
    print('Accuracy at step %s: %s' % (FLAGS.max_steps+1, acc))

    save_path = saver.save(sess, "MNIST.ckpt")
    print("Model saved to %s" % save_path)

    test_writer.flush()

    train_writer.close()
    test_writer.close()
