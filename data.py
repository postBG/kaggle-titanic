import os
import pandas as pd


class TitanicPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir

        self.train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

    def preprocess_and_save(self, preprocess_dir='preprocess'):
        self.processed_train = TitanicPreprocessor._preprocess(self.train)
        self.processed_test = TitanicPreprocessor._preprocess(self.test)

        self.preprocess_dir = preprocess_dir

        if not os.path.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

        self.processed_train.to_csv(os.path.join(self.preprocess_dir, 'train.csv'), index=False)
        self.processed_test.to_csv(os.path.join(self.preprocess_dir, 'test.csv'), index=False)

    @staticmethod
    def _preprocess(frame):
        """Preprocess using side effect."""
        frame = TitanicPreprocessor._add_or_modifiy_fields(frame)
        frame = TitanicPreprocessor._remove_fields(frame)

        return frame

    @staticmethod
    def _add_or_modifiy_fields(frame):
        frame['NameLength'] = frame['Name'].apply(len)
        frame['HasCabin'] = frame['Cabin'].apply(lambda cabin: 1 if type(cabin) == str else 0)

        dummy_fields = ['Pclass', 'Sex', 'Embarked']
        for field in dummy_fields:
            dummies = pd.get_dummies(frame[field], prefix=field, drop_first=False)
            frame = pd.concat([frame, dummies], axis=1)

        frame['Age'] = frame['Age'].fillna(frame['Age'].mean())
        frame['Fare'] = frame['Fare'].fillna(frame['Fare'].mean())

        return frame

    @staticmethod
    def _remove_fields(frame):
        remove_fields = ['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        for field in remove_fields:
            frame = frame.drop(field, axis=1)

        return frame