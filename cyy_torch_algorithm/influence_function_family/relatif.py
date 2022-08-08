from typing import Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.influence_function_family.influence_function import \
    get_default_inverse_hvp_arguments
from cyy_torch_algorithm.influence_function_family.inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product
from cyy_torch_algorithm.influence_function_family.util import \
    compute_perturbation_gradient_difference
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer


def compute_perturbation_relatif(
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
    test_gradient = test_gradient.cpu()

    if grad_diff is None:
        grad_diff = compute_perturbation_gradient_difference(
            trainer=trainer,
            perturbation_idx_fun=perturbation_idx_fun,
            perturbation_fun=perturbation_fun,
        )

    if inverse_hvp_arguments is None:
        inverse_hvp_arguments = get_default_inverse_hvp_arguments()
        inverse_hvp_arguments["repeated_num"] = 1

    res: dict = {}
    accumulated_indices = []
    accumulated_vectors = []
    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    torch.cuda.empty_cache()
    batch_size = 1
    for (perturbation_idx, v) in grad_diff.iterate():
        get_logger().error("v norm is %s", torch.linalg.vector_norm(v))
        accumulated_indices.append(perturbation_idx)
        accumulated_vectors.append(v)
        if len(accumulated_indices) != batch_size:
            continue
        products = stochastic_inverse_hessian_vector_product(
            inferencer, vectors=accumulated_vectors, **inverse_hvp_arguments
        )
        for idx, product in zip(accumulated_indices, products):
            res[idx] = (
                -test_gradient.dot(product) / torch.linalg.vector_norm(product)
            ).cpu()
        accumulated_indices = []
        accumulated_vectors = []
    if accumulated_indices:
        products = stochastic_inverse_hessian_vector_product(
            inferencer, vectors=accumulated_vectors, **inverse_hvp_arguments
        )
        for idx, product in zip(accumulated_indices, products):
            res[idx] = (
                -test_gradient.dot(product) / torch.linalg.vector_norm(product)
            ).item()
    assert len(res) == len(grad_diff)
    return res
