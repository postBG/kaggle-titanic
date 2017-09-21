class Model:
    def predict(self, data):
        raise NotImplementedError


class SimpleGenderModel(Model):
    def predict(self, data):
        answer = data[['PassengerId', 'Sex_female']].rename(columns={
            'Sex_female': 'Survived'
        })

        return answer
