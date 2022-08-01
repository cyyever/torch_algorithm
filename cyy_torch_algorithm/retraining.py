import math

from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from cyy_torch_toolbox.reproducible_env import global_reproducible_env


class DeterministicTraining:
    def __init__(self, config: DefaultConfig):
        self.config = config
        self.last_trainer = None
        self.seed_path = None

    def create_deterministic_trainer(self):
        self.config.make_reproducible_env = True
        self.config.apply_global_config()
        self.last_trainer = self.config.create_trainer()
        self.seed_path = global_reproducible_env.last_seed_path
        return self.last_trainer

    def recreate_trainer(self):
        previous_training_loss = {
            epoch: self.last_trainer.performance_metric.get_loss(epoch).cpu()
            for epoch in range(1, self.last_trainer.hyper_parameter.epoch + 1)
        }

        global_reproducible_env.disable()
        global_reproducible_env.load(self.seed_path)
        global_reproducible_env.enable()
        trainer = self.config.create_trainer()
        trainer.set_device(self.last_trainer.device)

        def validate_reproducibility(epoch, **_):
            if (
                math.fabs(
                    previous_training_loss[epoch]
                    - trainer.performance_metric.get_loss(epoch)
                )
                >= 1e-6
            ):
                raise RuntimeError("not in reproducible training")

        trainer.append_named_hook(
            ModelExecutorHookPoint.AFTER_EPOCH,
            "validate_reproducibility",
            validate_reproducibility,
            stripable=True,
        )
        return trainer
