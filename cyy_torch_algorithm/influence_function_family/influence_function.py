import functools
from typing import Callable

import torch
from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import (
    get_sample_gradient_product_dict, sample_dot_product)
from cyy_torch_algorithm.influence_function_family.inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from cyy_torch_algorithm.influence_function_family.util import \
    compute_perturbation_gradient_difference
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer


def get_default_inverse_hvp_arguments() -> dict:
    return {"dampling_term": 0.01, "scale": 100000, "epsilon": 0.03, "repeated_num": 3}


def compute_influence_function(
    trainer: Trainer,
    computed_indices: set | None,
    test_gradient: torch.Tensor | None = None,
    inverse_hvp_arguments: None | dict = None,
) -> dict:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()
        del inferencer

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()
    product = (
        stochastic_inverse_hessian_vector_product(
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()

    return get_sample_gradient_product_dict(
        inferencer=inferencer, vector=product, computed_indices=computed_indices
    )


def compute_perturbation_influence_function(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    test_gradient: torch.Tensor | None = None,
    inverse_hvp_arguments: None | dict = None,
    grad_diff=None,
) -> dict:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()

    product = (
        -stochastic_inverse_hessian_vector_product(
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()
    if grad_diff is not None:
        res = {}
        for (perturbation_idx, v) in grad_diff.items():
            res[perturbation_idx] = v.dot(product).item()
        return res

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(sample_dot_product, vector=product),
    )
