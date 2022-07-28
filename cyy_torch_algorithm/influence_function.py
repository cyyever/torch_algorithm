import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.sample_gradient_product.sample_gradient_product_hook import \
    get_sample_gradient_product_dict

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
    product = stochastic_inverse_hessian_vector_product(
        inferencer, test_gradient, **inverse_hvp_arguments
    ) / len(trainer.dataset)

    return get_sample_gradient_product_dict(inferencer, product, computed_indices)


def compute_perturbation_influence_function(
    trainer: Trainer,
    computed_indices: set | None,
    perturbation_fun,
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

    return get_sample_gradient_product_dict(
        inferencer=inferencer,
        vector=product,
        computed_indices=computed_indices,
        input_transform=perturbation_fun,
    )
