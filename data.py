import os
import pandas as pd


def file_exists(directory, filename):
    return os.path.exists(os.path.join(directory, filename))


class TitanicData:
    def __init__(self, data_dir='data', preprocess_dir='preprocess'):
        self.data_dir = data_dir
        self.preprocess_dir = preprocess_dir

        self.train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        self.test = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

    def load_preprocess_when_exists(self):
        if not os.path.exists(self.preprocess_dir):
            raise FileNotFoundError(self.preprocess_dir)

        self.processed_train = pd.read_csv(os.path.join(self.preprocess_dir, 'train.csv'))
        self.processed_validation = pd.read_csv(os.path.join(self.preprocess_dir, 'validation.csv'))
        self.processed_test = pd.read_csv(os.path.join(self.preprocess_dir, 'test.csv'))

    def preprocess_and_save(self, validation_rate=0.1):
        if file_exists(self.preprocess_dir, 'train.csv') \
                and file_exists(self.preprocess_dir, 'test.csv') \
                and file_exists(self.preprocess_dir, 'validation.csv'):
            self.load_preprocess_when_exists()
            return

        if not os.path.exists(self.preprocess_dir):
            os.makedirs(self.preprocess_dir)

        validation_number = int(len(self.train) * validation_rate)

        processed_train = TitanicData._preprocess(self.train)
        processed_train.iloc[:validation_number].to_csv(os.path.join(self.preprocess_dir, 'validation.csv'), index=False)
        processed_train.iloc[validation_number:].to_csv(os.path.join(self.preprocess_dir, 'train.csv'), index=False)

        processed_test = TitanicData._preprocess(self.test)
        processed_test.to_csv(os.path.join(self.preprocess_dir, 'test.csv'), index=False)

        self.load_preprocess_when_exists()

    @staticmethod
    def _preprocess(frame):
        """Preprocess using side effect."""
        frame = TitanicData._add_or_modifiy_fields(frame)
        frame = TitanicData._remove_fields(frame)

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
