import tensorflow as tf


def fully_connected(inputs, n_units, activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(), name=None, **kwargs):
    return tf.layers.dense(inputs, n_units,
                           activation=activation, kernel_initializer=kernel_initializer, name=name, **kwargs)


def dropout(inputs, drop_rate):
    return tf.layers.dropout(inputs, rate=drop_rate)
