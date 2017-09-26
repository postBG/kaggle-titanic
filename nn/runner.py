import tensorflow as tf

from nn.model import SimpleFeedForwardNetwork
from nn.trainer import Trainer
from nn.data_manager import TitanicBatchManager
from data import TitanicData

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../data',
                           """Directory path that contains titanic data as train.csv and test.csv""")
tf.app.flags.DEFINE_string('preprocess_dir', '../preprocess',
                           """Directory path that saves preprocessed data as csv""")
tf.app.flags.DEFINE_float('drop_rate', 0.5,
                          """drop rate 0.1 means 10% of units will be dropped""")
tf.app.flags.DEFINE_integer('n_units', 128,
                            """number of hidden units in hidden layers""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size""")
tf.app.flags.DEFINE_integer('epochs', 100,
                            """number of epochs""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                            """number of epochs""")


def main(argv=None):
    titanic = TitanicData(FLAGS.data_dir, FLAGS.preprocess_dir)
    titanic.preprocess_and_save()
    batch_manager = TitanicBatchManager(titanic, FLAGS.batch_size)

    inputs = tf.placeholder(tf.float32, shape=[None, len(titanic.processed_test.columns)])
    model = SimpleFeedForwardNetwork(inputs, FLAGS)
    trainer = Trainer(model, batch_manager, FLAGS)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainer.run(sess)

if __name__ == '__main__':
    tf.app.run()
