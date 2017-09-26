import tensorflow as tf
import pandas as pd

from nn.model import SimpleFeedForwardNetwork
from nn.trainer import Trainer
from nn.data_manager import TitanicBatchManager
from printer import DataFramePrinter
from data import TitanicData

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../data',
                           """Directory path that contains titanic data as train.csv and test.csv""")
tf.app.flags.DEFINE_string('preprocess_dir', '../preprocess',
                           """Directory path that saves preprocessed data as csv""")
tf.app.flags.DEFINE_float('drop_rate', 0.5,
                          """drop rate 0.1 means 10% of units will be dropped""")
tf.app.flags.DEFINE_integer('n_units', 16,  # 16
                            """number of hidden units in hidden layers""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """batch size""")
tf.app.flags.DEFINE_integer('epochs', 300,  # 300
                            """number of epochs""")
tf.app.flags.DEFINE_float('lr', 0.0001,
                          """number of epochs""")


def create_data_frame(passenger_ids, pred):
    df = pd.DataFrame(index=range(len(passenger_ids.index)))
    df['PassengerId'] = passenger_ids.copy()
    df['Survived'] = pd.Series(data=pred)

    return df


def main(argv=None):
    titanic = TitanicData(FLAGS.data_dir, FLAGS.preprocess_dir)
    titanic.preprocess_and_save()
    batch_manager = TitanicBatchManager(titanic, FLAGS.batch_size)

    inputs = tf.placeholder(tf.float32, shape=[None, len(titanic.processed_test.columns)])
    model = SimpleFeedForwardNetwork(inputs, FLAGS)
    trainer = Trainer(model, batch_manager, FLAGS)

    printer = DataFramePrinter()
    printer.answer_dir = '../answer'

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainer.run(sess)

        pred_labels = tf.argmax(model.logits, axis=1)
        pred = sess.run(pred_labels, feed_dict={
            model.inputs: titanic.processed_test,
            model.drop_rate: 1.0
        })

        printer.to_csv(create_data_frame(titanic.processed_test['PassengerId'].copy(), pred), 'output_nn.csv')


if __name__ == '__main__':
    tf.app.run()
