import tensorflow as tf


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def conv_layer(input_tensor, filter_shape, strides, padding, layer_name, act=tf.nn.relu):
    """
    a convolutional layer initializer

    :param input_tensor: shape [batch, in_height, in_width, in_channels]
    :param filter_shape: [filter_height, filter_width, in_channels, out_channels]
    :param strides: list of ints defining the stride
    :param padding: "SAME" / "VALID"
    :param layer_name: name of the layer
    :param act: the activation function used
    :return: shape [batch, in_height, in_width, in_channels]
    """

    with tf.name_scope('kernels'):
        kernels = tf.get_variable(
            name="Xavier_initializer",
            shape=filter_shape,
            initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            trainable=True
        )
        tf.image_summary(layer_name + "/kernels")
    with tf.name_scope('biases'):
        biases = tf.constant(1.0, shape=filter_shape[3])
        variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('convolution'):
        # http://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
        preactivation = tf.nn.conv2d(input_tensor, kernels, strides, padding)
        tf.image_summary(layer_name + "/preactivations")
        activation = act(preactivation) + biases
        tf.image_summary(layer_name + "/activations")

    return activation


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
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
        variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
        biases = tf.constant(0.1, shape=output_dim)
        variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations
