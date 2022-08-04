import functools
from typing import Callable

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import (
    get_sample_gradient_dict, get_sample_gradient_product_dict,
    sample_dot_product)

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product


def __get_inverse_hvp_arguments() -> dict:
    return {"dampling_term": 0.01, "scale": 1000, "epsilon": 0.03, "repeated_num": 3}


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

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = __get_inverse_hvp_arguments()
    product = (
        stochastic_inverse_hessian_vector_product(
            inferencer, test_gradient, **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )

    return get_sample_gradient_product_dict(
        inferencer=inferencer, vector=product, computed_indices=computed_indices
    )


def compute_perturbation_gradient_difference(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    result_transform: Callable | None = None,
) -> dict:
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )

    perturbation_idx_dict: dict = {}

    def sample_selector(sample_index, sample_input):
        nonlocal perturbation_idx_dict

        res = perturbation_idx_fun(sample_index=sample_index, sample_input=sample_input)
        if res:
            for perturbation_idx in res:
                if perturbation_idx not in perturbation_idx_dict:
                    perturbation_idx_dict[perturbation_idx] = set()
                perturbation_idx_dict[perturbation_idx].add(sample_index)
            return True
        return False

    tmp_dict = get_sample_gradient_dict(
        inferencer=inferencer,
        sample_selector=sample_selector,
        result_transform=result_transform,
    )
    sample_dict: dict = {}
    for perturbation_idx, sample_indices in perturbation_idx_dict.items():
        assert sample_indices
        for sample_index in sample_indices:
            v = tmp_dict[sample_index]
            if perturbation_idx not in sample_dict:
                sample_dict[perturbation_idx] = v
            else:
                sample_dict[perturbation_idx] = sample_dict[perturbation_idx] + v

    tmp_dict = get_sample_gradient_dict(
        inferencer=inferencer,
        input_transform=perturbation_fun,
        result_transform=result_transform,
    )
    perturbation_dict: dict = {}
    for k, v in tmp_dict.items():
        sample_index, component_index = k
        if component_index not in perturbation_dict:
            perturbation_dict[component_index] = v
        else:
            perturbation_dict[component_index] = perturbation_dict[component_index] + v

    result: dict = {}
    for perturbation_idx in sample_dict:
        result[perturbation_idx] = (
            sample_dict[perturbation_idx] - perturbation_dict[perturbation_idx]
        )

    return result


def compute_perturbation_influence_function(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    test_gradient: torch.Tensor | None = None,
    inverse_hvp_arguments: None | dict = None,
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
        inverse_hvp_arguments = __get_inverse_hvp_arguments()

    product = (
        -stochastic_inverse_hessian_vector_product(
            inferencer, test_gradient, **inverse_hvp_arguments
        )
        / trainer.dataset_size
    ).cpu()

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(sample_dot_product, vector=product),
    )
