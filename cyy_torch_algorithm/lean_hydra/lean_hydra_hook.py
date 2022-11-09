import json
import os

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.hook import Hook


class LeanHyDRAHook(Hook):
    def __init__(self, test_gradient):
        super().__init__(stripable=True)
        self.sample_gradient_hook = SampleGradientHook()
        self.__save_dir = None
        self._computed_indices = None
        self.__test_gradient = test_gradient.cpu()
        self._contributions: torch.Tensor | None = None
        self._training_set_size = None
        self.sample_gradient_hook.set_result_transform(self._gradient_dot_product)

    def _gradient_dot_product(self, result, **kwargs):
        return self.__test_gradient.dot(result.cpu()).item()

    @property
    def contributions(self):
        return self._contributions

    def __get_save_dir(self, trainer):
        if self.__save_dir is None:
            self.__save_dir = os.path.join(trainer.save_dir, "lean_HyDRA")
            os.makedirs(self.__save_dir, exist_ok=True)
        return self.__save_dir

    def _before_execute(self, model_executor, **kwargs):
        trainer = model_executor
        self._training_set_size = len(trainer.dataset_util)

        if not self._computed_indices:
            self._computed_indices = set(range(self._training_set_size))
        else:
            get_logger().info("only compute %s indices", len(self._computed_indices))
        self._contributions = torch.zeros(self._training_set_size).to(
            trainer.device, non_blocking=True
        )

    def set_computed_indices(self, computed_indices):
        self._computed_indices = set(computed_indices)
        self.sample_gradient_hook.set_computed_indices(computed_indices)

    def _after_execute(self, model_executor, **kwargs):
        trainer = model_executor
        assert self._contributions.shape[0] == self._training_set_size

        with open(
            os.path.join(self.__get_save_dir(trainer), "lean_hydra_contribution.json"),
            mode="wt",
            encoding="utf-8",
        ) as f:
            contributions = self._contributions.cpu().tolist()
            json.dump({idx: contributions[idx] for idx in self._computed_indices}, f)
        self.sample_gradient_hook.release_queue()
