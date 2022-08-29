#!/usr/bin/env python3

import torch.optim
from cyy_torch_algorithm.retraining import DeterministicTraining
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase

from .lean_hydra_adam_hook import LeanHyDRAAdamHook
from .lean_hydra_sgd_hook import LeanHyDRASGDHook


class LeanHyDRAConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.deterministic_training = DeterministicTraining(self)

    def create_deterministic_trainer(self):
        return self.deterministic_training.create_deterministic_trainer()

    def recreate_trainer_and_hook(self):
        tester = self.deterministic_training.last_trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=False
        )
        tester.disable_logger()
        test_gradient = tester.get_gradient()
        del tester
        optimizer = self.deterministic_training.last_trainer.get_optimizer()
        if isinstance(optimizer, torch.optim.SGD):
            hydra_hook = LeanHyDRASGDHook(test_gradient=test_gradient)
        elif isinstance(optimizer, torch.optim.Adam):
            hydra_hook = LeanHyDRAAdamHook(test_gradient=test_gradient)
        else:
            raise NotImplementedError(
                f"Unsupported optimizer {self.deterministic_training.last_trainer.hyper_parameter.optimizer_name}"
            )
        trainer = self.deterministic_training.recreate_trainer()
        trainer.append_hook(hydra_hook)

        return trainer, hydra_hook, test_gradient
