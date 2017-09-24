import tensorflow as tf
import pandas as pd

from data import TitanicData
import models
from printer import DataFramePrinter

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory path that contains titanic data as train.csv and test.csv""")
tf.app.flags.DEFINE_string('preprocess_dir', 'preprocess',
                           """Directory path that saves preprocessed data as csv""")


def main(argv=None):
    titanic = TitanicData(FLAGS.data_dir, FLAGS.preprocess_dir)
    titanic.preprocess_and_save()
    printer = DataFramePrinter()

    model = models.MyDecisionTreeClassifier(max_depth=3)
    model.fit(pd.concat([titanic.processed_validation, titanic.processed_train]))
    printer.to_csv(model.predict(titanic.processed_test), 'output_decision.csv')


if __name__ == '__main__':
    tf.app.run()
