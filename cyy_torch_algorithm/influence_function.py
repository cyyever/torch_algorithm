from typing import Callable

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.sample_gradient_product.sample_gradient_product_hook import \
    get_sample_gradient_product_dict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product


def __get_inverse_hvp_arguments() -> dict:
    return {"dampling_term": 0.01, "scale": 1000, "epsilon": 0.03, "repeated_num": 1}


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
    product = stochastic_inverse_hessian_vector_product(
        inferencer, test_gradient, **inverse_hvp_arguments
    ) / len(trainer.dataset)

    return get_sample_gradient_product_dict(inferencer, product, computed_indices)


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
    product = -stochastic_inverse_hessian_vector_product(
        inferencer, test_gradient, **inverse_hvp_arguments
    ) / len(trainer.dataset)

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

    tmp_dict = get_sample_gradient_product_dict(
        inferencer=inferencer, vector=product, sample_selector=sample_selector
    )
    sample_product_dict: dict = {}
    for perturbation_idx, sample_indices in perturbation_idx_dict.items():
        assert sample_indices
        for sample_index in sample_indices:
            if perturbation_idx not in sample_product_dict:
                sample_product_dict[perturbation_idx] = tmp_dict[sample_index]
            else:
                sample_product_dict[perturbation_idx] = (
                    sample_product_dict[perturbation_idx] + tmp_dict[sample_index]
                )

    tmp_dict = get_sample_gradient_product_dict(
        inferencer=inferencer,
        vector=product,
        input_transform=perturbation_fun,
    )
    perturbation_product_dict: dict = {}
    for k, v in tmp_dict.items():
        sample_index, component_index = k
        if component_index not in perturbation_product_dict:
            perturbation_product_dict[component_index] = v
        else:
            perturbation_product_dict[component_index] = (
                perturbation_product_dict[component_index] + v
            )

    result: dict = {}
    for perturbation_idx in sample_product_dict:
        result[perturbation_idx] = (
            sample_product_dict[perturbation_idx]
            - perturbation_product_dict[perturbation_idx]
        )

    return result
