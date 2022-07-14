
"""This is a TensorFlow implementation of VGG16.

Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
(https://arxiv.org/abs/1409.1556)

Explanation on VGGNet can be found in my blog post:
https://mohitjain.me/2018/06/07/vggnet/

@author: Mohit Jain (contact: mohitjain1999(at)yahoo.com)
"""

import tensorflow as tf
import numpy as np

class VGG16(object):

    """ Implementation of VGG16 network """

    def __init__(self, num_classes):

        """Create the graph of the AlexNet model.

         Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
        """

        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes
        self.DROP_RATE = 0.0

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):

        """Create the network graph."""

        with tf.variable_scope('vgg16') as vs:

            self.conv1_1_W, self.conv1_1_b = var_conv_layer(3, 64, 'conv1_1')
            self.conv1_2_W, self.conv1_2_b = var_conv_layer(64, 64, 'conv1_2')

            self.conv2_1_W, self.conv2_1_b = var_conv_layer(64, 128, 'conv2_1')
            self.conv2_2_W, self.conv2_2_b = var_conv_layer(128, 128, 'conv2_2')

            self.conv3_1_W, self.conv3_1_b = var_conv_layer(128, 256, 'conv3_1')
            self.conv3_2_W, self.conv3_2_b = var_conv_layer(256, 256, 'conv3_2')
            self.conv3_3_W, self.conv3_3_b = var_conv_layer(256, 256, 'conv3_3')

            self.conv4_1_W, self.conv4_1_b = var_conv_layer(256, 512, 'conv4_1')
            self.conv4_2_W, self.conv4_2_b = var_conv_layer(512, 512, 'conv4_2')
            self.conv4_3_W, self.conv4_3_b = var_conv_layer(512, 512, 'conv4_3')

            self.conv5_1_W, self.conv5_1_b = var_conv_layer(512, 512, 'conv5_1')
            self.conv5_2_W, self.conv5_2_b = var_conv_layer(512, 512, 'conv5_2')
            self.conv5_3_W, self.conv5_3_b = var_conv_layer(512, 512, 'conv5_3')

            self.fc6_W, self.fc6_b = var_fc_layer(2 * 2 * 512, 4096, name = 'fc6')
            self.fc7_W, self.fc7_b = var_fc_layer(4096, 4096, name = 'fc7')
            self.fc8_W, self.fc8_b = var_fc_layer(4096, self.NUM_CLASSES, name = 'fc8')

    def model(self, inputs):

        with tf.variable_scope('vgg16') as vs:

            conv1_1 = conv_layer(inputs, self.conv1_1_W, self.conv1_1_b)
            conv1_2 = conv_layer(conv1_1, self.conv1_2_W, self.conv1_2_b)
            pool1 = max_pool(conv1_2, 'pool1')

            conv2_1 = conv_layer(pool1, self.conv2_1_W, self.conv2_1_b)
            conv2_2 = conv_layer(conv2_1, self.conv2_2_W, self.conv2_2_b)
            pool2 = max_pool(conv2_2, 'pool2')

            conv3_1 = conv_layer(pool2, self.conv3_1_W, self.conv3_1_b)
            conv3_2 = conv_layer(conv3_1, self.conv3_2_W, self.conv3_2_b)
            conv3_3 = conv_layer(conv3_2, self.conv3_3_W, self.conv3_3_b)
            pool3 = max_pool(conv3_3, 'pool3')

            conv4_1 = conv_layer(pool3, self.conv4_1_W, self.conv4_1_b)
            conv4_2 = conv_layer(conv4_1, self.conv4_2_W, self.conv4_2_b)
            conv4_3 = conv_layer(conv4_2, self.conv4_3_W, self.conv4_3_b)
            pool4 = max_pool(conv4_3, 'pool4')

            conv5_1 = conv_layer(pool4, self.conv5_1_W, self.conv5_1_b)
            conv5_2 = conv_layer(conv5_1, self.conv5_2_W, self.conv5_2_b)
            conv5_3 = conv_layer(conv5_2, self.conv5_3_W, self.conv5_3_b)
            pool5 = max_pool(conv5_3, 'pool5')

            flattened = tf.reshape(pool5, [-1, 2 * 2 * 512])
            fc6 = fc_layer(flattened, self.fc6_W, self.fc6_b)
            dropout6 = dropout(fc6, self.DROP_RATE)

            fc7 = fc_layer(dropout6, self.fc7_W, self.fc7_b)
            dropout7 = dropout(fc7, self.DROP_RATE)

            fc8 = fc_layer(dropout7, self.fc8_W, self.fc8_b, relu = False)

        return fc8

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'vgg16' in var.name]


def var_conv_layer(input_channels, num_filters, name, filter_height = 3, filter_width = 3):

    """Create a convolution layer."""

    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases of the conv layer
        W = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters], initializer = tf.contrib.layers.xavier_initializer())

        b = tf.get_variable('biases', shape = [num_filters], initializer = tf.constant_initializer(0.0))

    return W, b

def conv_layer(x, W, b, stride = 1, padding = 'SAME'):

    # Perform convolution.
    conv = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding)
    # Add the biases.
    z = tf.nn.bias_add(conv, b)
    # Apply ReLu non linearity.
    a = tf.nn.relu(z)

    return a

def var_fc_layer(input_size, output_size, name):

    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases.
        W = tf.get_variable('weights', shape = [input_size, output_size], initializer = tf.contrib.layers.xavier_initializer())

        b = tf.get_variable('biases', shape = [output_size], initializer = tf.constant_initializer(0.0))

    return W, b

def fc_layer(x, W, b, relu = True):
    # Matrix multiply weights and inputs and add biases.
    z = tf.nn.bias_add(tf.matmul(x, W), b)

    if relu:
        # Apply ReLu non linearity.
        a = tf.nn.relu(z)
        return a
    else:
        return z

def max_pool(x, name, filter_height = 2, filter_width = 2,
    stride = 2, padding = 'VALID'):

    """Create a max pooling layer."""

    return tf.nn.max_pool(x, ksize = [1, filter_height, filter_width, 1],
            strides = [1, stride, stride, 1], padding = padding, name = name)

def dropout(x, drop_rate):

    """Create a dropout layer."""

    return tf.nn.dropout(x, rate = drop_rate)
