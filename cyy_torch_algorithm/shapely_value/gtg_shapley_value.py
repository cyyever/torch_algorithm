import copy

import numpy as np
from cyy_naive_lib.log import log_debug, log_info, log_warning

from .shapley_value import RoundBasedShapleyValue


class GTGShapleyValue(RoundBasedShapleyValue):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.shapley_values: dict[int, dict] = {}
        self.shapley_values_S: dict[int, dict] = {}

        self.eps = 0.001
        self.round_trunc_threshold = 0.001

        self.converge_min = max(30, self.player_number)
        self.last_k = 10
        self.converge_criteria = 0.05

        self.max_percentage = 0.8
        self.max_number = min(
            2**self.player_number,
            max(
                self.converge_min,
                self.max_percentage * (2**self.player_number)
                + np.random.randint(-5, +5),
            ),
        )
        log_info("max_number %s", self.max_number)

    def _compute_impl(self, round_index: int) -> None:
        self.shapley_values[round_index] = {}
        self.shapley_values_S[round_index] = {}
        assert self.metric_fun is not None
        this_round_metric = self.metric_fun(self.complete_player_indices)
        metrics: dict = {}

        # for best_S
        perm_records = {}

        index = 0
        contribution_records: list = []
        last_round_metric = self.get_last_round_metric(round_index=round_index)
        while not metrics or self.not_convergent(index, contribution_records):
            for player_id in self.complete_player_indices:
                index += 1
                v: list = [0] * (self.player_number + 1)
                v[0] = last_round_metric
                marginal_contribution = [0] * self.player_number
                perturbed_indices = np.concatenate(
                    (
                        np.array([player_id]),
                        np.random.permutation(
                            [i for i in self.complete_player_indices if i != player_id]
                        ),
                    )
                ).astype(int)

                for j in self.complete_player_indices:
                    subset = tuple(sorted(perturbed_indices[: (j + 1)].tolist()))
                    # truncation
                    if abs(this_round_metric - v[j]) >= self.eps:
                        if subset not in metrics:
                            if not subset:
                                metric = last_round_metric
                            else:
                                metric = self.metric_fun(subset)
                                if metric is None:
                                    log_warning("force stop")
                                    return
                            log_info(
                                "round %s subset %s metric %s",
                                round_index,
                                self.get_players(subset),
                                metric,
                            )
                            metrics[subset] = metric
                        v[j + 1] = metrics[subset]
                    else:
                        v[j + 1] = v[j]

                    # update SV
                    marginal_contribution[perturbed_indices[j]] = v[j + 1] - v[j]
                contribution_records.append(marginal_contribution)
                # for best_S
                perm_records[tuple(perturbed_indices.tolist())] = marginal_contribution

        # for best_S
        subset_rank = sorted(
            metrics.items(), key=lambda x: (x[1], -len(x[0])), reverse=True
        )
        if subset_rank[0][0]:
            best_S: tuple = subset_rank[0][0]
        else:
            best_S = subset_rank[1][0]

        contrib_S = [
            v for k, v in perm_records.items() if set(k[: len(best_S)]) == set(best_S)
        ]
        SV_calc_temp = np.sum(contrib_S, 0) / len(contrib_S)
        round_marginal_gain_S = metrics[best_S] - last_round_metric
        round_SV_S: dict = {}
        for client_id in best_S:
            round_SV_S[client_id] = float(SV_calc_temp[client_id])

        self.shapley_values_S[round_index] = self.normalize_shapley_values(
            round_SV_S, round_marginal_gain_S
        )

        # calculating fullset SV
        # shapley value calculation
        if set(best_S) == set(self.complete_player_indices):
            self.shapley_values[round_index] = copy.deepcopy(
                self.shapley_values_S[round_index]
            )
        else:
            round_shapley_values = np.sum(contribution_records, 0) / len(
                contribution_records
            )
            assert len(round_shapley_values) == self.player_number

            round_marginal_gain = this_round_metric - last_round_metric
            round_shapley_value_dict = {}
            for idx, value in enumerate(round_shapley_values):
                round_shapley_value_dict[idx] = float(value)

            self.shapley_values[round_index] = self.normalize_shapley_values(
                round_shapley_value_dict, round_marginal_gain
            )

        self.shapley_values[round_index] = {
            self.get_players(k): v for k, v in self.shapley_values[round_index].items()
        }
        self.shapley_values_S[round_index] = {
            self.get_players(k): v
            for k, v in self.shapley_values_S[round_index].items()
        }
        log_info("shapley_value %s", self.shapley_values[round_index])
        log_info("shapley_value_S %s", self.shapley_values_S[round_index])

    def get_best_players(self, round_index: int) -> set | None:
        return set(self.shapley_values_S[round_index].keys())

    def get_result(self) -> dict:
        return {
            "round_shapley_values": self.shapley_values,
            "round_shapley_values_approximated": self.shapley_values_S,
        }

    def not_convergent(self, index: int, contribution_records: list) -> bool:
        if index >= self.max_number:
            log_info("convergent for max_number %s", self.max_number)
            return False
        if index <= self.converge_min:
            return True
        all_vals = (
            np.cumsum(contribution_records, 0)
            / np.reshape(np.arange(1, len(contribution_records) + 1), (-1, 1))
        )[-self.last_k :]
        errors = np.mean(
            np.abs(all_vals[-self.last_k :] - all_vals[-1:])
            / (np.abs(all_vals[-1:]) + 1e-12),
            -1,
        )
        if np.max(errors) > self.converge_criteria:
            return True
        log_debug(
            "convergent in index %s and min index for convergent %s max error %s error threshold %s",
            index,
            self.converge_min,
            np.max(errors),
            self.converge_criteria,
        )
        return False
