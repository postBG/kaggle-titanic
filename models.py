import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class Model:
    def predict(self, data):
        raise NotImplementedError


def print_stats(predicted, labels):
    correct = sum([p == l for p, l in zip(predicted, labels)])
    print('model accuracy: {:2}'.format(correct / len(labels)))


class SklearnModel(Model):
    def fit(self, train_data):
        features, labels = self._seperate_features_and_labels(train_data)
        self.model.fit(features, labels)

    def predict(self, data):
        ids = data['PassengerId']
        labels = self.model.predict(data)

        return pd.DataFrame(data=[(passenger_id, survive,) for passenger_id, survive in zip(ids, labels)],
                            columns=['PassengerId', 'Survived'])

    def _seperate_features_and_labels(self, train_data):
        train_labels = train_data['Survived']
        train_features = train_data.drop('Survived', axis=1)

        return train_features, train_labels


class SimpleGenderModel(Model):
    def predict(self, data):
        answer = data[['PassengerId', 'Sex_female']].rename(columns={
            'Sex_female': 'Survived'
        })

        return answer


class MyDecisionTreeClassifier(SklearnModel):
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeClassifier(*args, **kwargs)
