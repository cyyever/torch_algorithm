import math
from typing import Any, Callable

from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import ExecutorHookPoint
from cyy_torch_toolbox.reproducible_env import global_reproducible_env
from cyy_torch_toolbox.trainer import Trainer


class DeterministicTraining:
    def __init__(self, config: DefaultConfig) -> None:
        self.config = config
        self.trainer_fun: Callable = self.config.create_trainer
        self.last_trainer: None | Trainer = None
        self.seed_path: None | str = None

    def create_deterministic_trainer(
        self, trainer_fun: None | Callable = None
    ) -> Trainer:
        self.config.reproducible_env_config.make_reproducible_env = True
        self.config.apply_global_config()
        if trainer_fun is not None:
            self.trainer_fun = trainer_fun
        self.last_trainer = self.trainer_fun()
        self.seed_path = global_reproducible_env.last_seed_path
        return self.last_trainer

    def recreate_trainer(self) -> Trainer:
        assert self.last_trainer is not None
        previous_training_loss = {
            epoch: self.last_trainer.performance_metric.get_loss(epoch)
            for epoch in range(1, self.last_trainer.hyper_parameter.epoch + 1)
        }

        global_reproducible_env.disable()
        global_reproducible_env.load(self.seed_path)
        global_reproducible_env.enable()
        trainer = self.trainer_fun()
        trainer.set_device(self.last_trainer.device)

        def validate_reproducibility(**kwargs: Any) -> None:
            epoch = kwargs["epoch"]
            if (
                math.fabs(
                    previous_training_loss[epoch]
                    - trainer.performance_metric.get_loss(epoch)
                )
                >= 1e-6
            ):
                raise RuntimeError("not in reproducible training")

        trainer.append_named_hook(
            ExecutorHookPoint.AFTER_EPOCH,
            "validate_reproducibility",
            validate_reproducibility,
            stripable=True,
        )
        return trainer
