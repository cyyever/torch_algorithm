from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.trainer import Trainer

from cyy_torch_algorithm.sample_gradient.sample_gradient_util import \
    get_sample_gradient_dict

from .inverse_hessian_vector_product import (
    stochastic_inverse_hessian_vector_product,
    stochastic_inverse_hessian_vector_product_new)


def compute_influence_function(
    trainer: Trainer,
    computed_indices,
    test_gradient=None,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
) -> dict:

    if test_gradient is None:
        inferencer = trainer.get_inferencer(phase=MachineLearningPhase.Test)
        test_gradient = inferencer.get_gradient()

    product = stochastic_inverse_hessian_vector_product(
        trainer.dataset,
        trainer.copy_model_with_loss(True),
        test_gradient,
        repeated_num=3,
        max_iteration=None,
        batch_size=trainer.hyper_parameter.batch_size,
        dampling_term=dampling_term,
        scale=scale,
        epsilon=epsilon,
    ) / len(trainer.dataset)

    inferencer = trainer.get_inferencer(phase=MachineLearningPhase.Training)
    training_sample_gradient_dict = get_sample_gradient_dict(
        inferencer, computed_indices
    )
    return {
        sample_index: (product @ sample_gradient).data.item()
        for (sample_index, sample_gradient) in training_sample_gradient_dict.iterate()
    }


def compute_influence_function_new(
    trainer: Trainer,
    computed_indices,
    test_gradient=None,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
) -> dict:

    if test_gradient is None:
        inferencer = trainer.get_inferencer(
            phase=MachineLearningPhase.Test, copy_model=True
        )
        test_gradient = inferencer.get_gradient()

    product = stochastic_inverse_hessian_vector_product_new(
        trainer.get_inferencer(phase=MachineLearningPhase.Training, copy_model=True),
        test_gradient,
        repeated_num=3,
        dampling_term=dampling_term,
        scale=scale,
        epsilon=epsilon,
    ) / len(trainer.dataset)

    inferencer = trainer.get_inferencer(phase=MachineLearningPhase.Training)
    training_sample_gradient_dict = get_sample_gradient_dict(
        inferencer, computed_indices
    )
    return {
        sample_index: (product @ sample_gradient).data.item()
        for sample_index, sample_gradient in training_sample_gradient_dict.items()
    }
