import tensorflow as tf


def _optimizer(model, lr):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(model.loss)


def _correct_pred(model):
    return tf.equal(tf.argmax(model.logits, axis=1), tf.argmax(model.labels, axis=1))


def _accuracy(model):
    return tf.reduce_mean(tf.cast(_correct_pred(model), tf.float32), name="accuracy")


class Trainer:
    def __init__(self, model, batch_manager, configs):
        self.model = model
        self.batch_manager = batch_manager
        self.epochs = configs.epochs
        self.dropout_rate = configs.drop_rate

        self.validation_features, self.validation_labels = self.batch_manager.validation_data

        self.optimizer = _optimizer(self.model, configs.lr)
        self.correct_pred = _correct_pred(self.model)
        self.accuracy = _accuracy(model)

    def run(self, session):
        for epoch in range(self.epochs):
            self.train_one_epoch(session, epoch)

    def train_one_epoch(self, session, epoch):
        for features, labels in self.batch_manager.batch_data():
            session.run(self.optimizer, feed_dict={
                self.model.inputs: features,
                self.model.labels: labels,
                self.model.drop_rate: self.dropout_rate
            })

        loss, accur, train_accur = self.validate_stats(session, features, labels)
        self._print_stats(loss, accur, train_accur, epoch=epoch)

    def validate_stats(self, session, train_features, train_labels):
        loss = session.run(self.model.loss, feed_dict={
            self.model.inputs: self.validation_features,
            self.model.labels: self.validation_labels,
            self.model.drop_rate: 1.0
        })

        accur = session.run(self.accuracy, feed_dict={
            self.model.inputs: self.validation_features,
            self.model.labels: self.validation_labels,
            self.model.drop_rate: 1.0
        })

        train_accur = session.run(self.accuracy, feed_dict={
            self.model.inputs: train_features,
            self.model.labels: train_labels,
            self.model.drop_rate: 1.0
        })

        return loss, accur, train_accur

    def _print_stats(self, loss, accuracy, train_accuracy, **kwargs):
        print('Epoch {:>2}:  '.format(kwargs.get('epoch', 0), end=''))
        print('Traning Loss: {:>10.4f}, Train Accuracy: {:.6f}, Validation Accuracy: {:.6f}'
              .format(loss, train_accuracy, accuracy))
