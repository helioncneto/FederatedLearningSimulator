from sklearn.metrics import confusion_matrix
from typing import Protocol


class Metrics(Protocol):
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        """Implementation of metric building"""


class Accuracy:
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        tn, fp, fn, tp = conf_matrix
        return (tp + tn) / (tp + tn + fp, fn)


class Precision:
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        _, fp, _, tp = conf_matrix
        return tp / (tp + fp)


class Sensitivity:
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        _, _, fn, tp = conf_matrix
        return tp / (tp + fn)


class Specificity:
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        tn, fp, _, _ = conf_matrix
        return tn / (tn + fp)


class F1Score:
    @staticmethod
    def calculate(conf_matrix: tuple) -> float:
        return 2 / ((1 / Precision.calculate(conf_matrix)) + (1 / Sensitivity.calculate(conf_matrix)))


class Evaluator:

    AVAILABLE_METRICS = {'accuracy': Accuracy(),
                         'precision': Precision(),
                         'sensitivity': Sensitivity(),
                         'specificity': Specificity(),
                         'f1score': F1Score()}
    metrics: list

    def __init__(self, *args) -> None:
        if set(args).issubset(set(self.AVAILABLE_METRICS.keys())):
            self.metrics = list(args)
        else:
            raise Exception

    def run_metrics(self, predicted, real) -> dict:
        tn, fp, fn, tp = confusion_matrix(real, predicted).ravel()
        results = {}
        for metric in self.metrics:
            metric = self.AVAILABLE_METRICS[metric]
            result = metric.calculate((tn, fp, fn, tp))
            results[metric] = result
        return results


