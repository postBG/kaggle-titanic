import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from printer import print_stats


class Model:
    def _seperate_features_and_labels(self, train_data):
        train_labels = train_data['Survived']
        train_features = train_data.drop('Survived', axis=1)

        return train_features, train_labels

    def predict(self, data):
        raise NotImplementedError


class SklearnMixin(Model):
    def fit(self, train_data):
        features, labels = self._seperate_features_and_labels(train_data)
        self.model.fit(features, labels)

        print_stats(self.model.predict(features), labels)

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
        self.model = DecisionTreeClassifier(*args, **kwargs)


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


class ApesTogetherStrongModel(Model):
    def __init__(self):
        self.simple_gender_model = SimpleGenderModel()
        self.decision_tree_model = MyDecisionTreeClassifier()
        self.logistic_model = MyLogisticRegression()
        self.svc_model = MySVC()
        self.random_forest_model = MyRandomForestClassifier()

        self.models = [self.simple_gender_model,
                       self.decision_tree_model,
                       self.logistic_model,
                       self.svc_model,
                       self.random_forest_model]

    def fit(self, train_data):
        for model in self.models:
            model.fit(train_data)

    def predict(self, data):
        voting_board = pd.DataFrame()
