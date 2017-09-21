import tensorflow as tf

from data import TitanicPreprocessor

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', 'data',
                           """Directory path that contains titanic data as train.csv and test.csv""")
tf.app.flags.DEFINE_string('preprocess_dir', 'preprocess',
                           """Directory path that saves preprocessed data as csv""")


def main(argv=None):
    titanic = TitanicPreprocessor()
    titanic.preprocess_and_save()


if __name__ == '__main__':
    tf.app.run()
