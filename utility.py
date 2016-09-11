import tensorflow as tf


def conv_layer(input_tensor, filter_shape, strides, padding):
    """
    a convolutional layer initializer

    :param input_tensor: shape [batch, in_height, in_width, in_channels]
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param strides: list of ints defining the stride
    :param padding: "SAME" / "VALID"
    :return: shape [batch, in_height, in_width, in_channels]
    """

    with tf.name_scope('kernels'):
        kernels = tf.get_variable(
            name="Xavier_initializer",
            shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            trainable=True
        )
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(1.0, shape=filter_shape[3]))

    with tf.name_scope('convolution'):
        # http://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
        pre_bias_activation = tf.nn.conv2d(input_tensor, kernels, strides, padding)
        activation = tf.nn.bias_add(pre_bias_activation, biases)

    return activation


def nn_layer(input_tensor, input_dim, output_dim, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """

    with tf.name_scope('weights'):
        weights = tf.get_variable(
            name="Xavier_initializer",
            shape=[input_dim, output_dim],
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.constant(0.1, shape=output_dim))
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
    activations = act(preactivate, 'activation')
    return activations
