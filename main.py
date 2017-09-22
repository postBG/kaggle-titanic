import tensorflow as tf

from data import TitanicData
from models import SimpleGenderModel, MyDecisionTreeClassifier
from printer import DataFramePrinter

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory path that contains titanic data as train.csv and test.csv""")
tf.app.flags.DEFINE_string('preprocess_dir', 'preprocess',
                           """Directory path that saves preprocessed data as csv""")
tf.app.flags.DEFINE_string('simple_output', 'output_simple.csv',
                           """Directory path that saves simple gender model output as csv""")
tf.app.flags.DEFINE_string('decision_output', 'output_decision.csv',
                           """Directory path that saves simple gender model output as csv""")


def main(argv=None):
    titanic = TitanicData(FLAGS.data_dir, FLAGS.preprocess_dir)
    titanic.preprocess_and_save()
    printer = DataFramePrinter()

    model = MyDecisionTreeClassifier(max_depth=5)
    model.fit(titanic.processed_train)
    printer.to_csv(model.predict(titanic.processed_test), FLAGS.decision_output)


if __name__ == '__main__':
    tf.app.run()
