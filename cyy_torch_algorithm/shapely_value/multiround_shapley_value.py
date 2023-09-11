import copy
import math

from cyy_naive_lib.log import get_logger

from .shapley_value import ShapleyValue


class MultiRoundShapleyValue(ShapleyValue):
    def __init__(
        self,
        players: list,
        last_round_metric: float = 0,
        round_trunc_threshold: float | None = None,
    ) -> None:
        super().__init__(players=players, last_round_metric=last_round_metric)
        self.shapley_values: dict = {}
        self.shapley_values_S: dict = {}
        self.round_trunc_threshold = round_trunc_threshold

    def compute(self, round_number: int) -> None:
        self.shapley_values.clear()
        self.shapley_values_S.clear()
        assert self.metric_fun is not None
        metrics: dict = {tuple(): self.last_round_metric}
        this_round_metric = self.metric_fun(self.complete_player_indices)
        if this_round_metric is None:
            get_logger().warning("force stop")
            return
        metrics[self.complete_player_indices] = this_round_metric
        if self.round_trunc_threshold is not None and (
            abs(this_round_metric - self.last_round_metric)
            <= self.round_trunc_threshold
        ):
            self.shapley_values = {i: 0 for i in self.complete_player_indices}
            self.shapley_values_S = {i: 0 for i in self.complete_player_indices}
            if self.save_fun is not None:
                self.save_fun(
                    self.shapley_values,
                    self.shapley_values_S,
                )
            get_logger().info(
                "skip round %s, this_round_metric %s last_round_metric %s round_trunc_threshold %s",
                round_number,
                this_round_metric,
                self.last_round_metric,
                self.round_trunc_threshold,
            )
            self.last_round_metric = this_round_metric
            return

        for subset in self.powerset(self.complete_player_indices):
            key = tuple(sorted(subset))
            if key not in metrics:
                if not subset:
                    metric = self.last_round_metric
                else:
                    metric = self.metric_fun(subset)
                    if metric is None:
                        get_logger().warning("force stop")
                        return
                metrics[key] = metric
                get_logger().info(
                    "round %s subset %s metric %s", round_number, key, metric
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

        self.shapley_values_S = ShapleyValue.normalize_shapley_values(
            round_SV_S, round_marginal_gain_S
        )

        # calculating fullset SV
        if set(best_S) == set(self.complete_player_indices):
            self.shapley_values = copy.deepcopy(self.shapley_values_S)
        else:
            round_shapley_values = {}
            for subset, metric in metrics.items():
                if not subset:
                    continue
                for player_id in subset:
                    marginal_contribution = (
                        metric
                        - metrics[tuple(sorted(i for i in subset if i != player_id))]
                    )
                    if player_id not in round_shapley_values:
                        round_shapley_values[player_id] = 0
                    round_shapley_values[player_id] += marginal_contribution / (
                        (math.comb(self.player_number - 1, len(subset) - 1))
                        * self.player_number
                    )

            round_marginal_gain = this_round_metric - self.last_round_metric
            self.shapley_values = ShapleyValue.normalize_shapley_values(
                round_shapley_values, round_marginal_gain
            )

        if self.save_fun is not None:
            self.save_fun(
                self.shapley_values,
                self.shapley_values_S,
            )
        self.last_round_metric = this_round_metric
        get_logger().info("shapley_value %s", self.shapley_values)
        get_logger().info("shapley_value_best_set %s", self.shapley_values_S)
