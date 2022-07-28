import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.sample_gradient_product.sample_gradient_product_hook import \
    get_sample_gradient_product_dict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product


def compute_influence_function(
    trainer: Trainer,
    computed_indices: set | None,
    test_gradient: torch.Tensor | None = None,
    dampling_term: float = 0.01,
    scale: float = 1000,
    epsilon: float = 0.03,
) -> dict:
    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()

    inferencer = trainer.get_inferencer(
        phase=MachineLearningPhase.Training, copy_model=True
    )
    product = stochastic_inverse_hessian_vector_product(
        inferencer,
        test_gradient,
        repeated_num=3,
        dampling_term=dampling_term,
        scale=scale,
        epsilon=epsilon,
    ) / len(trainer.dataset)

    return get_sample_gradient_product_dict(inferencer, product, computed_indices)
