import math
from itertools import chain, combinations
from typing import Any, Callable, Iterable

from cyy_naive_lib.log import log_info


class ShapleyValue:
    def __init__(self, players: list) -> None:
        self.players: tuple = tuple(players)
        self.metric_fun: None | Callable = None
        self.batch_metric_fun: None | Callable = None

    @property
    def player_number(self) -> int:
        return len(self.players)

    @property
    def complete_player_indices(self) -> tuple:
        return tuple(range(len(self.players)))

    def set_metric_function(self, metric_fun) -> None:
        self.metric_fun = lambda subset: metric_fun(self.__get_players(subset))
        assert self.batch_metric_fun is None
        self.batch_metric_fun = lambda subsets: {
            subset: metric_fun(self.__get_players(subset)) for subset in subsets
        }

    def set_batch_metric_function(self, metric_fun) -> None:
        self.batch_metric_fun = lambda subsets: metric_fun(
            tuple(self.__get_players(subset) for subset in subsets)
        )
        assert self.metric_fun is None
        self.metric_fun = lambda subset: list(self.batch_metric_fun([subset]).values())[
            0
        ]

    def get_shapely_values(self) -> Any:
        return None

    @classmethod
    def powerset(cls, iterable: Iterable) -> chain:
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s: list = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    @classmethod
    def normalize_shapley_values(
        cls, shapley_values: dict, marginal_gain: float
    ) -> dict:
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

    def __get_players(self, indices) -> tuple:
        return tuple(self.players[i] for i in indices)


class RoundBasedShapleyValue(ShapleyValue):
    def __init__(self, initial_metric: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initial_metric = initial_metric
        self.round_trunc_threshold: float | None = None
        self.round_metrics: dict[int, float] = {}

    def set_round_truncation_threshold(self, threshold: float) -> None:
        self.round_trunc_threshold = threshold

    def get_last_round_metric(self, round_number: int) -> float:
        last_round_metric = self.initial_metric
        previous_rounds = tuple(k for k in self.round_metrics if k < round_number)
        if previous_rounds:
            last_round_metric = self.round_metrics[max(previous_rounds)]
        return last_round_metric

    def compute(self, round_number: int) -> None:
        assert self.metric_fun is not None
        self.round_metrics[round_number] = self.metric_fun(self.complete_player_indices)
        if self.round_trunc_threshold is not None and (
            abs(
                self.round_metrics[round_number]
                - self.get_last_round_metric(round_number=round_number)
            )
            <= self.round_trunc_threshold
        ):
            log_info(
                "skip round %s, current_round_metric %s last_round_metric %s round_trunc_threshold %s",
                round_number,
                self.round_metrics[round_number],
                self.get_last_round_metric(round_number=round_number),
                self.round_trunc_threshold,
            )
            return None
        self._compute_impl(round_number=round_number)
        return None

    def get_result(self) -> Any:
        return None

    def exit(self) -> None:
        pass

    def _compute_impl(self, round_number: int) -> None:
        raise NotImplementedError()
