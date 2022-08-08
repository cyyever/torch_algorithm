from typing import Callable

import torch
import torch.nn.functional
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.influence_function import \
    compute_perturbation_gradient_difference


def compute_perturbation_grad_cos(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    test_gradient: torch.Tensor | None = None,
) -> tuple:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()

    diff = compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
    )
    res: dict = {}
    test_gradient = test_gradient.cpu()
    for (perturbation_idx, v) in diff.iterate():
        res[perturbation_idx] = torch.nn.functional.cosine_similarity(
            v.cpu(), test_gradient, dim=0
        ).item()
    return res, diff
