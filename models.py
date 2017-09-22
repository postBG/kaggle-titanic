import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Model:
    def _seperate_features_and_labels(self, train_data):
        train_labels = train_data['Survived']
        train_features = train_data.drop('Survived', axis=1)

        return train_features, train_labels

    def predict(self, data):
        raise NotImplementedError


def print_stats(predicted, labels):
    correct = sum([p == l for p, l in zip(predicted, labels)])
    print('model accuracy: {:2}'.format(correct / len(labels)))


class SklearnMixin(Model):
    def fit(self, train_data):
        features, labels = self._seperate_features_and_labels(train_data)
        self.model.fit(features, labels)

    def predict(self, data):
        ids = data['PassengerId']
        labels = self.model.predict(data)

        return pd.DataFrame(data=[(passenger_id, survive,) for passenger_id, survive in zip(ids, labels)],
                            columns=['PassengerId', 'Survived'])


class SimpleGenderModel(Model):
    def predict(self, data):
        answer = data[['PassengerId', 'Sex_female']].rename(columns={
            'Sex_female': 'Survived'
        })

        return answer


class MyDecisionTreeClassifier(SklearnMixin):
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeClassifier(max_depth=9)


class MyLogisticRegression(SklearnMixin):
    def __init__(self, *args, **kwargs):
        self.model = LogisticRegression(*args, **kwargs)


class MySVC(SklearnMixin):
    def __init__(self):
        self.model = SVC(kernel='linear', C=0.025)


class MyRandomForestClassifier(SklearnMixin):
    def __init__(self):
        rf_params = {
            'n_jobs': -1,
            'n_estimators': 500,
            'warm_start': True,
            'max_depth': 6,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'verbose': 0
        }
        self.model = RandomForestClassifier(**rf_params)