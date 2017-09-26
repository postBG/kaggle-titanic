from data import TitanicData, separate_features_and_labels


def one_hot_encoding_labels(labels):
    return [[0, 1] if label == 1 else [1, 0] for label in labels]


class TitanicBatchManager:
    def __init__(self, titanic, batch_size=80):
        self.titanic = titanic
        self.batch_size = batch_size

        self.validation_features, self.validation_labels = \
            separate_features_and_labels(self.titanic.processed_validation)

    def batch_data(self):
        train_data_size = len(self.titanic.processed_train)

        train_features, train_labels = separate_features_and_labels(self.titanic.processed_train)
        for start in range(0, train_data_size, self.batch_size):
            end = min(start + self.batch_size - 1, train_data_size)

            if end == train_data_size:
                return
            yield train_features.loc[start:end].values, one_hot_encoding_labels(train_labels.loc[start:end].values)

    @property
    def validation_data(self):
        return self.validation_features.values, one_hot_encoding_labels(self.validation_labels.values)
