import functools
from typing import Callable

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.influence_function import \
    compute_perturbation_gradient_difference
from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import \
    sample_dot_product


def compute_perturbation_grad_dot(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    test_gradient: torch.Tensor | None = None,
    grad_diff=None,
) -> dict:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()

    if grad_diff is not None:
        test_gradient = test_gradient.cpu()
        res = {}
        for (perturbation_idx, v) in grad_diff.iterate():
            res[perturbation_idx] = v.dot(test_gradient).item()
        return res

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(sample_dot_product, vector=test_gradient),
    )
