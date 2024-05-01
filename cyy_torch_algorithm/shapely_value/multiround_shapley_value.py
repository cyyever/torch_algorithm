import copy
import math

from cyy_naive_lib.log import log_info

from .shapley_value import RoundBasedShapleyValue


class MultiRoundShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shapley_values: dict[int, dict] = {}
        self.shapley_values_S: dict[int, dict] = {}

    def _compute_impl(self, round_number: int) -> None:
        self.shapley_values[round_number] = {}
        self.shapley_values_S[round_number] = {}
        assert self.metric_fun is not None
        last_round_metric = self.get_last_round_metric(round_number=round_number)
        metrics: dict = {
            (): last_round_metric,
            self.complete_player_indices: self.round_metrics[round_number],
        }

        subsets = set()
        for subset in self.powerset(self.complete_player_indices):
            sorted_subset = tuple(sorted(subset))
            if sorted_subset not in metrics:
                subsets.add(sorted_subset)

        assert self.batch_metric_fun is not None
        resulting_metrics = self.batch_metric_fun(subsets)
        for subset, metric in resulting_metrics.items():
            log_info("round %s subset %s metric %s", round_number, subset, metric)
        metrics |= resulting_metrics

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
        round_marginal_gain_S = metrics_S[best_S] - last_round_metric

        self.shapley_values_S[round_number] = self.normalize_shapley_values(
            round_SV_S, round_marginal_gain_S
        )

        # calculating fullset SV
        if set(best_S) == set(self.complete_player_indices):
            self.shapley_values[round_number] = copy.deepcopy(
                self.shapley_values_S[round_number]
            )
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

            round_marginal_gain = self.round_metrics[round_number] - last_round_metric
            self.shapley_values[round_number] = self.normalize_shapley_values(
                round_shapley_values, round_marginal_gain
            )

        log_info("shapley_value %s", self.shapley_values[round_number])
        log_info("shapley_value_best_set %s", self.shapley_values_S[round_number])

    def get_best_players(self, round_number: int) -> set | None:
        return set(self.get_players(self.shapley_values_S[round_number].keys()))

    def get_result(self) -> dict:
        return {
            "round_shapley_values": self.shapley_values,
            "round_shapley_values_approximated": self.shapley_values_S,
        }
