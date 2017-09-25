import tensorflow as tf

from models import Model


class SimpleFeedFoward(Model):
    def __init__(self, config, inputs):
        self.inputs = inputs

        self.drop_rate = config.drop_rate
        self.config = config

    def build_graph(self, inputs):
        pass
