import tensorflow as tf

from .layers import fully_connected, dropout


class SimpleFeedForwardNetwork:
    def __init__(self, config, inputs):
        self.inputs = inputs

        self.drop_rate = config.drop_rate
        self.bottleneck_n_units = config.bottleneck_n_units
        self.config = config

    def build_graph(self, inputs):
        self.bottleneck_layer = fully_connected(inputs, self.bottleneck_n_units, name="bottleneck")
