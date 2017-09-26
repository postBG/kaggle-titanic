import tensorflow as tf

from .layers import fully_connected, dropout


class SimpleFeedForwardNetwork:
    def __init__(self, inputs, configs):
        self.inputs = inputs
        self.labels = tf.placeholder(tf.int32, name='labels')

        self.drop_rate = tf.placeholder(tf.float32, name="dropout_rate")
        self.n_units = configs.n_units
        self.configs = configs

        self.build_graph(self.inputs)

    def build_graph(self, inputs):
        with tf.name_scope('hidden_layer1'):
            self.layer1 = dropout(
                fully_connected(self.inputs, self.n_units,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.05)),
                self.drop_rate)

        with tf.name_scope('output_layer'):
            self.logits = fully_connected(self.layer1, 2)

        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.labels, self.logits))
