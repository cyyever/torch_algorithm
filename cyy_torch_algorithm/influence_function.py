import functools
from typing import Callable

import torch
from cyy_torch_toolbox.device import put_data_to_device
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_algorithm.inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import (
    get_sample_gradient_dict, get_sample_gradient_product_dict,
    sample_dot_product)


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

    sample_to_perturbations: dict = {}

    def sample_selector(sample_index, sample_input):
        nonlocal sample_to_perturbations

        res = perturbation_idx_fun(sample_index=sample_index, sample_input=sample_input)
        if res:
            assert sample_index not in sample_to_perturbations
            sample_to_perturbations[sample_index] = res
            return True
        return False

    if result_transform is None:
        sample_dict = SyncedTensorDict.create(cache_size=128)
    else:
        sample_dict: dict = {}

    def collect_result(result_dict):
        nonlocal sample_dict
        nonlocal sample_to_perturbations
        for sample_idx, grad in result_dict.items():
            for perturbation_idx in sample_to_perturbations[sample_idx]:
                if perturbation_idx not in sample_dict:
                    sample_dict[perturbation_idx] = grad
                else:
                    grad = put_data_to_device(
                        grad,
                        device=sample_dict[perturbation_idx].device,
                        non_blocking=True,
                    )
                    sample_dict[perturbation_idx] = sample_dict[perturbation_idx] + grad

    get_sample_gradient_dict(
        inferencer=inferencer,
        sample_selector=sample_selector,
        result_transform=result_transform,
        result_collection_fun=collect_result,
    )
    if result_transform is None:
        perturbation_dict = SyncedTensorDict.create(cache_size=128)
    else:
        perturbation_dict: dict = {}

    def collect_result2(result_dict):
        nonlocal perturbation_dict
        for k, v in result_dict.items():
            sample_index, component_index = k
            if component_index not in perturbation_dict:
                perturbation_dict[component_index] = v
            else:
                perturbation_dict[component_index] = (
                    perturbation_dict[component_index] + v
                )

    get_sample_gradient_dict(
        inferencer=inferencer,
        input_transform=perturbation_fun,
        result_transform=result_transform,
        result_collection_fun=collect_result2,
    )
    if result_transform is None:
        result = SyncedTensorDict.create(cache_size=128)
    else:
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
    )

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(sample_dot_product, vector=product),
    )
