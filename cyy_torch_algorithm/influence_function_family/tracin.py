import json
import os

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import (
    SampleGradientHook, get_sample_gradient_dict)
from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class TracInHook(Hook):
    def __init__(
        self, test_sample_indices: set | None = None, check_point_gap: int | None = None
    ):
        super().__init__(stripable=True)
        self._sample_grad_hook: SampleGradientHook = SampleGradientHook()

        self.__check_point_gap = check_point_gap
        self.__test_sample_indices = test_sample_indices
        self.__test_sample_grad_dict = SyncedTensorDict.create()

        self.__tracked_indices: None | set = None
        self.__influence_values: dict = {}

    def set_tracked_indices(self, tracked_indices: set) -> None:
        self.__tracked_indices = set(tracked_indices)
        get_logger().info("track %s indices", len(self.__tracked_indices))

    @property
    def influence_values(self) -> dict:
        return self.__influence_values

    def get_save_dir(self, trainer):
        save_dir = os.path.join(trainer.save_dir, "TracIn")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    @torch.no_grad()
    def _before_execute(self, model_executor, **kwargs):
        self.__influence_values = {}
        if self.__tracked_indices is not None:
            self._sample_grad_hook.set_computed_indices(self.__tracked_indices)
        self.__compute_test_sample_gradients(model_executor=model_executor)

    def __compute_test_sample_gradients(self, model_executor):
        inferencer = model_executor.get_inferencer(phase=MachineLearningPhase.Test)
        inferencer.disable_logger()
        inferencer.disable_performance_metric_logger()
        if self.__test_sample_indices is None:
            self.__test_sample_grad_dict[-1] = inferencer.get_gradient()
        else:

            def collect_result(res_dict):
                for k, v in res_dict.items():
                    self.__test_sample_grad_dict[k] = v

            get_sample_gradient_dict(
                inferencer=inferencer,
                computed_indices=self.__test_sample_indices,
                result_collection_fun=collect_result,
            )

    @torch.no_grad()
    def _before_batch(self, model_executor, batch_index, **kwargs):
        if self.__check_point_gap is None or batch_index % self.__check_point_gap == 0:
            self.__compute_test_sample_gradients(model_executor=model_executor)

    def _after_optimizer_step(self, model_executor, step_skipped, batch_size, **kwargs):
        if step_skipped:
            return
        assert self.__test_sample_grad_dict
        trainer = model_executor
        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")
        lr = optimizer.param_groups[0]["lr"]
        # momentum = optimizer.param_groups[0]["momentum"]
        # assert momentum == 0
        # lr = optimizer.param_groups[0]["lr"]
        # weight_decay = optimizer.param_groups[0]["weight_decay"]
        # assert weight_decay == 0

        assert self.__test_sample_grad_dict
        for k, test_grad in self.__test_sample_grad_dict.items():
            if k not in self.__influence_values:
                self.__influence_values[k] = {}
            for k2, sample_grad in self._sample_grad_hook.result_dict.items():
                if k2 not in self.__influence_values[k]:
                    self.__influence_values[k][k2] = 0
                self.__influence_values[k][k2] += (
                    test_grad.cpu().dot(sample_grad.cpu()).item() * lr / batch_size
                )
        self._sample_grad_hook.reset_result()

    def _after_execute(self, model_executor, **kwargs):
        if -1 in self.__influence_values:
            assert len(self.__influence_values) == 1
            self.__influence_values = self.__influence_values[-1]
        with open(
            os.path.join(
                self.get_save_dir(model_executor),
                "influence_value.json",
            ),
            mode="wt",
            encoding="utf-8",
        ) as f:
            json.dump(self.__influence_values, f)
        self._sample_grad_hook.release_queue()
