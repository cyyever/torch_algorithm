import math
from collections.abc import Callable, Iterable
from itertools import chain, combinations
from typing import Any

from cyy_naive_lib.log import log_info


class ShapleyValue:
    def __init__(self, players: Iterable, **kwargs: Any) -> None:
        self.players: tuple = ()
        self.set_players(players)
        self.metric_fun: None | Callable = None
        self.batch_metric_fun: None | Callable = None

    def set_players(self, players: Iterable) -> None:
        self.players = tuple(players)

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["batch_metric_fun"] = None
        state["metric_fun"] = None
        return state

    @property
    def player_number(self) -> int:
        return len(self.players)

    @property
    def complete_player_indices(self) -> tuple:
        return tuple(range(len(self.players)))

    def set_metric_function(self, metric_fun: Callable) -> None:
        assert self.metric_fun is None
        self.metric_fun = lambda subset: metric_fun(self.get_players(subset))
        assert self.batch_metric_fun is None
        self.batch_metric_fun = lambda subsets: {
            subset: metric_fun(self.get_players(subset)) for subset in subsets
        }

    def set_batch_metric_function(self, metric_fun: Callable) -> None:
        assert self.batch_metric_fun is None
        assert self.metric_fun is None
        self.batch_metric_fun = metric_fun
        assert self.batch_metric_fun is not None
        self.metric_fun = lambda subset: list(metric_fun([subset]).values())[0]

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

    def get_players(self, indices: Iterable | int) -> tuple | Any:
        if isinstance(indices, int):
            return self.players[indices]
        return tuple(self.players[i] for i in indices)


class RoundBasedShapleyValue(ShapleyValue):
    def __init__(self, initial_metric: float = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initial_metric = initial_metric
        self.round_trunc_threshold: float | None = None
        self.round_metrics: dict[int, float] = {}

    def set_round_truncation_threshold(self, threshold: float) -> None:
        self.round_trunc_threshold = threshold

    def get_last_round_metric(self, round_index: int) -> float:
        last_round_metric = self.initial_metric
        previous_rounds = tuple(k for k in self.round_metrics if k < round_index)
        if previous_rounds:
            last_round_metric = self.round_metrics[max(previous_rounds)]
        return last_round_metric

    def compute(self, round_index: int) -> None:
        assert self.metric_fun is not None
        self.round_metrics[round_index] = self.metric_fun(self.complete_player_indices)
        if self.round_trunc_threshold is not None and (
            abs(
                self.round_metrics[round_index]
                - self.get_last_round_metric(round_index=round_index)
            )
            <= self.round_trunc_threshold
        ):
            log_info(
                "skip round %s, current_round_metric %s last_round_metric %s round_trunc_threshold %s",
                round_index,
                self.round_metrics[round_index],
                self.get_last_round_metric(round_index=round_index),
                self.round_trunc_threshold,
            )
            return None
        self._compute_impl(round_index=round_index)
        return None

    def get_best_players(self, round_index: int) -> set | None:
        return None

    def get_result(self) -> Any:
        return None

    def exit(self) -> None:
        pass

    def _compute_impl(self, round_index: int) -> None:
        raise NotImplementedError()
