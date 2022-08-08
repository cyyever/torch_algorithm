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
    return {"dampling_term": 0.01, "scale": 10000, "epsilon": 0.03, "repeated_num": 3}


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
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()

    return get_sample_gradient_product_dict(
        inferencer=inferencer, vector=product, computed_indices=computed_indices
    )


def compute_perturbation_gradient_difference(
    trainer: Trainer,
    perturbation_idx_fun: Callable,
    perturbation_fun: Callable,
    result_transform: Callable | None = None,
):
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
        nonlocal inferencer
        for sample_idx, v in result_dict.items():
            v = put_data_to_device(v, device=inferencer.device, non_blocking=True)
            for perturbation_idx in sample_to_perturbations[sample_idx]:
                if perturbation_idx not in sample_dict:
                    sample_dict[perturbation_idx] = v
                else:
                    sample_dict[perturbation_idx] = sample_dict[perturbation_idx] + v

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
        nonlocal inferencer
        for k, v in result_dict.items():
            sample_index, perturbation_index = k
            v = put_data_to_device(v, device=inferencer.device, non_blocking=True)
            if perturbation_index not in perturbation_dict:
                perturbation_dict[perturbation_index] = v
            else:
                perturbation_dict[perturbation_index] = (
                    perturbation_dict[perturbation_index] + v
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
    assert len(sample_dict) == len(perturbation_dict)
    for perturbation_idx in sample_dict.keys():
        tmp = sample_dict[perturbation_idx] - perturbation_dict[perturbation_idx]
        if isinstance(tmp, torch.Tensor):
            tmp = tmp.cpu()
        result[perturbation_idx] = tmp
    if result_transform is None:
        sample_dict.release()
        perturbation_dict.release()
    return result


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
        inverse_hvp_arguments = __get_inverse_hvp_arguments()

    product = (
        -stochastic_inverse_hessian_vector_product(
            inferencer, vectors=[test_gradient], **inverse_hvp_arguments
        )
        / trainer.dataset_size
    )[0].cpu()
    if grad_diff is not None:
        res = {}
        for (perturbation_idx, v) in grad_diff.iterate():
            res[perturbation_idx] = v.dot(product).item()
        return res

    return compute_perturbation_gradient_difference(
        trainer=trainer,
        perturbation_idx_fun=perturbation_idx_fun,
        perturbation_fun=perturbation_fun,
        result_transform=functools.partial(sample_dot_product, vector=product),
    )
