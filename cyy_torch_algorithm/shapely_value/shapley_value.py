import math
from itertools import chain, combinations


class ShapleyValue:
    @staticmethod
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def __init__(
        self,
        worker_number: int,
        last_round_metric: float = 0,
    ):
        self.worker_number = worker_number
        self.last_round_metric = last_round_metric
        self.round_number = 0
        self.metric_fun = None
        self.save_fun = None

    def set_metric_function(self, metric_fun):
        self.metric_fun = metric_fun

    def set_save_function(self, save_fun):
        self.save_fun = save_fun

    @staticmethod
    def normalize_shapley_values(shapley_values: dict, marginal_gain: float) -> dict:
        sum_value: float = 0
        if marginal_gain >= 0:
            sum_value = sum([v for v in shapley_values.values() if v >= 0])
            if math.isclose(sum_value, 0):
                sum_value = 1e-9
        else:
            sum_value = sum([v for v in shapley_values.values() if v < 0])
            if math.isclose(sum_value, 0):
                sum_value = -1e-9

        return {k: marginal_gain * v / sum_value for k, v in shapley_values.items()}
