import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import separate_features_and_labels
from printer import print_stats


class Model:
    def fit(self, train_data):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError


class SklearnMixin(Model):
    def fit(self, train_data):
        features, labels = separate_features_and_labels(train_data)
        self.model.fit(features, labels)

        print_stats(self.model.predict(features), labels)

    def predict(self, data):
        ids = data['PassengerId']
        labels = self.model.predict(data)

        return pd.DataFrame(data=[(passenger_id, survive,) for passenger_id, survive in zip(ids, labels)],
                            columns=['PassengerId', 'Survived'])


class SimpleGenderModel(Model):
    def fit(self, train_data):
        pass

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


class EnsembleWeakModel(Model):
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
        voting_board = self._create_voting_board(data)
        for model in self.models:
            voting_board['voting'] = voting_board['voting'] + model.predict(data)['Survived']

        majority_threshold = len(self.models) / 2
        voting_board['Survived'] = voting_board['voting'].apply(lambda vote: 1 if vote > majority_threshold else 0)

        return voting_board.filter(['PassengerId', 'Survived'], axis=1)

    def _create_voting_board(self, data):
        voting_board = pd.DataFrame(index=range(len(data.index)))
        voting_board['PassengerId'] = data['PassengerId'].copy()
        voting_board['voting'] = pd.Series([0 for _ in range(len(voting_board.index))])

        return voting_board
