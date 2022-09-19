import copy
import math

from cyy_naive_lib.log import get_logger

from .shapley_value import ShapleyValue


class MultiRoundShapleyValue(ShapleyValue):
    def __init__(
        self,
        worker_number: int,
        last_round_metric: float = 0,
        round_trunc_threshold: float | None = None,
    ):
        super().__init__(
            worker_number=worker_number, last_round_metric=last_round_metric
        )
        self.shapley_values: dict = {}
        self.shapley_values_S: dict = {}
        self.round_trunc_threshold = round_trunc_threshold

    def compute(self):
        assert self.metric_fun is not None
        self.round_number += 1
        metrics = {}
        this_round_metric = self.metric_fun(
            self.round_number, set(range(self.worker_number))
        )
        if this_round_metric is None:
            get_logger().warning("force stop")
            return
        metrics[()] = self.last_round_metric
        metrics[tuple(range(self.worker_number))] = this_round_metric
        if self.round_trunc_threshold is not None:
            if (
                abs(this_round_metric - self.last_round_metric)
                <= self.round_trunc_threshold
            ):
                self.shapley_values[self.round_number] = {
                    i: 0 for i in range(self.worker_number)
                }
                self.shapley_values_S[self.round_number] = {
                    i: 0 for i in range(self.worker_number)
                }
                if self.save_fun is not None:
                    self.save_fun(
                        self.round_number,
                        self.shapley_values[self.round_number],
                        self.shapley_values_S[self.round_number],
                    )
                get_logger().info(
                    "skip round %s, this_round_metric %s last_round_metric %s round_trunc_threshold %s",
                    self.round_number,
                    this_round_metric,
                    self.last_round_metric,
                    self.round_trunc_threshold,
                )
                self.last_round_metric = this_round_metric
                return

        for subset in ShapleyValue.powerset(range(self.worker_number)):
            key = tuple(sorted(subset))
            if key not in metrics:
                if not subset:
                    metric = self.last_round_metric
                else:
                    metric = self.metric_fun(self.round_number, subset)
                    if metric is None:
                        get_logger().warning("force stop")
                        return
                metrics[key] = metric
                get_logger().info(
                    "round %s subset %s metric %s", self.round_number, key, metric
                )

        # best subset in metrics
        subset_rank = sorted(
            metrics.items(), key=lambda x: (x[1], -len(x[0])), reverse=True
        )
        if subset_rank[0][0]:
            best_S = subset_rank[0][0]
        else:
            best_S = subset_rank[1][0]

        # calculating best subset SV
        N_S = len(best_S)
        metrics_S = {k: v for k, v in metrics.items() if set(k).issubset(set(best_S))}
        round_SV_S = {}
        for subset, metric in metrics_S.items():
            if not subset:
                continue
            for client_id in subset:
                marginal_contribution = (
                    metric
                    - metrics_S[tuple(sorted(i for i in subset if i != client_id))]
                )
                if client_id not in round_SV_S:
                    round_SV_S[client_id] = 0
                round_SV_S[client_id] += marginal_contribution / (
                    (math.comb(N_S - 1, len(subset) - 1)) * N_S
                )
        round_marginal_gain_S = metrics_S[best_S] - self.last_round_metric

        self.shapley_values_S[
            self.round_number
        ] = ShapleyValue.normalize_shapley_values(round_SV_S, round_marginal_gain_S)

        # calculating fullset SV
        if set(best_S) == set(range(self.worker_number)):
            self.shapley_values[self.round_number] = copy.deepcopy(
                self.shapley_values_S[self.round_number]
            )
        else:
            round_shapley_values = {}
            for subset, metric in metrics.items():
                if not subset:
                    continue
                for client_id in subset:
                    marginal_contribution = (
                        metric
                        - metrics[tuple(sorted(i for i in subset if i != client_id))]
                    )
                    if client_id not in round_shapley_values:
                        round_shapley_values[client_id] = 0
                    round_shapley_values[client_id] += marginal_contribution / (
                        (math.comb(self.worker_number - 1, len(subset) - 1))
                        * self.worker_number
                    )

            round_marginal_gain = this_round_metric - self.last_round_metric
            self.shapley_values[
                self.round_number
            ] = ShapleyValue.normalize_shapley_values(
                round_shapley_values, round_marginal_gain
            )

        if self.save_fun is not None:
            self.save_fun(
                self.round_number,
                self.shapley_values[self.round_number],
                self.shapley_values_S[self.round_number],
            )
        self.last_round_metric = this_round_metric
        get_logger().info("shapley_value %s", self.shapley_values[self.round_number])
        get_logger().info(
            "shapley_value_best_set %s", self.shapley_values_S[self.round_number]
        )
