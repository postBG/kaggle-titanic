import os


class Printer:
    answer_dir = 'answer'

    def to_csv(self, data, filename):
        if not os.path.exists(Printer.answer_dir):
            os.makedirs(Printer.answer_dir)

        self._to_csv(data, filename)

    def _to_csv(self, data, filename):
        raise NotImplementedError


class DataFramePrinter(Printer):
    def _to_csv(self, data, filename):
        data.to_csv(os.path.join(self.answer_dir, filename), index=False)
