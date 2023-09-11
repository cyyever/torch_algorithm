import math
from itertools import chain, combinations
from typing import Callable, Iterable


class ShapleyValue:
    @classmethod
    def powerset(cls, iterable: Iterable) -> chain:
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s: list = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def __init__(
        self,
        players: list,
        last_round_metric: float = 0,
    ) -> None:
        self.players: tuple = tuple(players)
        self.last_round_metric = last_round_metric
        self.round_number = 0
        self.metric_fun: None | Callable = None
        self.save_fun: None | Callable = None

    @property
    def player_number(self) -> int:
        return len(self.players)

    def set_metric_function(self, metric_fun) -> None:
        self.metric_fun = lambda subset: metric_fun(self.__get_players(subset))

    def set_save_function(self, save_fun) -> None:
        def new_save_fun(sv1, sv2) -> None:
            save_fun(
                {self.players[k]: v for k, v in sv1.items()},
                {self.players[k]: v for k, v in sv2.items()},
            )

        self.save_fun = new_save_fun

    def __get_players(self, indices) -> tuple:
        return tuple(self.players[i] for i in indices)

    @property
    def complete_player_indices(self) -> tuple:
        return tuple(range(len(self.players)))

    @staticmethod
    def normalize_shapley_values(shapley_values: dict, marginal_gain: float) -> dict:
        sum_value: float = 0
        if marginal_gain >= 0:
            sum_value = sum(v for v in shapley_values.values() if v >= 0)
            if math.isclose(sum_value, 0):
                sum_value = 1e-9
        else:
            sum_value = sum(v for v in shapley_values.values() if v < 0)
            if math.isclose(sum_value, 0):
                sum_value = -1e-9

        return {k: marginal_gain * v / sum_value for k, v in shapley_values.items()}
