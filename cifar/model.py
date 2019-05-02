"""Cifar model."""
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple(
    "HParams", "batch_size, num_classes, min_lrn_rate, lrn_rate, num_units, weight_decay_rate, relu_leakiness, optimizer"
)


class Net(object):
    """CNN model with skip connection."""

    def __init__(self, hps, images, labels, mode):
        """Model constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
        self.hps = hps
        self._images = images
        self.labels = labels
        self.mode = mode

        self._extra_train_ops = []

    def build_graph(self):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self._build_model()
        if self.mode == "train":
            self._build_train_op()
        self.summaries = tf.summary.merge_all()

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        """Build the core model within the graph."""
        with tf.variable_scope("init"):
            x = self._images
            x = self._conv("init_conv", x, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        skip_connection_func = self._skip_connection
        filters = [16, 64, 128, 256]

        with tf.variable_scope("unit_1_0"):
            x = skip_connection_func(x, filters[0], filters[1], self._stride_arr(strides[0]), True)
        for i in six.moves.range(1, self.hps.num_units):
            with tf.variable_scope("unit_1_%d" % i):
                x = skip_connection_func(x, filters[1], filters[1], self._stride_arr(1), False)

        with tf.variable_scope("unit_2_0"):
            x = skip_connection_func(x, filters[1], filters[2], self._stride_arr(strides[1]), False)
        for i in six.moves.range(1, self.hps.num_units):
            with tf.variable_scope("unit_2_%d" % i):
                x = skip_connection_func(x, filters[2], filters[2], self._stride_arr(1), False)

        with tf.variable_scope("unit_3_0"):
            x = skip_connection_func(x, filters[2], filters[3], self._stride_arr(strides[2]), False)
        for i in six.moves.range(1, self.hps.num_units):
            with tf.variable_scope("unit_3_%d" % i):
                x = skip_connection_func(x, filters[3], filters[3], self._stride_arr(1), False)

        with tf.variable_scope("unit_last"):
            x = self._batch_norm("final_bn", x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._global_avg_pool(x)

        with tf.variable_scope("logit"):
            logits = self._fully_connected(x, self.hps.num_classes)
            normalized_logits = tf.nn.sigmoid(logits)
            self.predictions = tf.nn.sigmoid(logits)

        with tf.variable_scope("label_or_not"):
            labeled_examples = tf.greater(tf.reduce_max(self.labels, axis=1), tf.zeros([self.hps.batch_size, 1]))
            labeled_examples = tf.cast(labeled_examples, tf.float32)
            unlabeled_examples = tf.ones([self.hps.batch_size, 1]) - labeled_examples

        with tf.variable_scope("costs"):
            xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.labels)
            xent = tf.reduce_mean(xent, axis=1)
            self.cross_entropy = tf.reduce_mean(tf.multiply(labeled_examples, xent), name="xent")

            wmc = tf.zeros([self.hps.batch_size, 1])
            for i in range(self.hps.num_classes):
                one_situation = tf.concat(
                    [
                        tf.concat([tf.ones([self.hps.batch_size, i]), tf.zeros([self.hps.batch_size, 1])], axis=1),
                        tf.ones([self.hps.batch_size, self.hps.num_classes - i - 1]),
                    ],
                    axis=1,
                )
                wmc += tf.reduce_prod(one_situation - normalized_logits, axis=1)
            wmc = tf.abs(wmc)
            self.wmc = tf.reduce_mean(wmc)
            log_wmc = tf.log(wmc)

            self.cost = tf.multiply(labeled_examples, xent) - 0.05 * tf.multiply(unlabeled_examples, log_wmc)
            self.cost = tf.reduce_mean(self.cost)
            self.cost += self._decay()
            tf.summary.scalar("cost", self.cost)

    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar("learning_rate", self.lrn_rate)

        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)

        if self.hps.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
        elif self.hps.optimizer == "mom":
            optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)

        apply_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step, name="train_step")

        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable("beta", params_shape, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable("gamma", params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == "train":
                mean, variance = tf.nn.moments(x, [0, 1, 2], name="moments")

                moving_mean = tf.get_variable(
                    "moving_mean",
                    params_shape,
                    tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False,
                )
                moving_variance = tf.get_variable(
                    "moving_variance",
                    params_shape,
                    tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False,
                )

                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                    "moving_mean",
                    params_shape,
                    tf.float32,
                    initializer=tf.constant_initializer(0.0, tf.float32),
                    trainable=False,
                )
                variance = tf.get_variable(
                    "moving_variance",
                    params_shape,
                    tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32),
                    trainable=False,
                )
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(variance.op.name, variance)
            # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())
            return y

    def _skip_connection(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Skip connection unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope("common_bn_relu"):
                x = self._batch_norm("init_bn", x)
                x = self._relu(x, self.hps.relu_leakiness)
                orig_x = x
        else:
            with tf.variable_scope("residual_bn_relu"):
                orig_x = x
                x = self._batch_norm("init_bn", x)
                x = self._relu(x, self.hps.relu_leakiness)

        with tf.variable_scope("sub1"):
            x = self._conv("conv1", x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope("sub2"):
            x = self._batch_norm("bn2", x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv("conv2", x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope("sub3"):
            x = self._batch_norm("bn3", x)
            x = self._relu(x, self.hps.relu_leakiness)
            x = self._conv("conv3", x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope("sub_add"):
            if in_filter != out_filter:
                orig_x = self._conv("project", orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info("image after unit %s", x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r"DW") > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.summary.histogram(var.op.name, var)

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                "DW",
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
            )
            return tf.nn.conv2d(x, kernel, strides, padding="SAME")

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name="leaky_relu")

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        x = tf.reshape(x, [self.hps.batch_size, -1])
        w = tf.get_variable("DW", [x.get_shape()[1], out_dim], initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable("biases", [out_dim], initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
