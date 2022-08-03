#!/usr/bin/env python3


import torch.optim
from cyy_torch_algorithm.hydra.hydra_adam_hook import HyDRAAdamHook
from cyy_torch_algorithm.hydra.hydra_sgd_hook import HyDRASGDHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import MachineLearningPhase


class HyDRAConfig(DefaultConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_size: int = 128
        self.tracking_percentage: float = None
        self.__tracking_indices = None
        self.use_hessian: bool = False
        self.use_approximation: bool = True

    def create_trainer(self, return_hydra_hook=False, **kwargs):
        trainer = super().create_trainer(**kwargs)

        optimizer = trainer.get_optimizer()
        if isinstance(optimizer, torch.optim.SGD):
            hydra_hook = HyDRASGDHook(
                cache_size=self.cache_size,
                use_hessian=self.use_hessian,
                use_approximation=self.use_approximation,
            )
        elif isinstance(optimizer, torch.optim.Adam):
            hydra_hook = HyDRAAdamHook(
                cache_size=self.cache_size,
                use_hessian=self.use_hessian,
                use_approximation=self.use_approximation,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer {type(optimizer)}")
        trainer.remove_optimizer()
        trainer.append_hook(hydra_hook)

        if self.tracking_percentage is not None:
            subset_dict = trainer.dataset_collection.get_dataset_util(
                phase=MachineLearningPhase.Training
            ).iid_sample(self.tracking_percentage)
            self.__tracking_indices = sum(subset_dict.values(), [])
        if self.__tracking_indices:
            hydra_hook.set_computed_indices(self.__tracking_indices)
        if return_hydra_hook:
            return trainer, hydra_hook
        return trainer
