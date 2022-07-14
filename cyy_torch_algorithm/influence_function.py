from cyy_torch_cpp_extension.data_structure import SyncedTensorDict

from .inverse_hessian_vector_product import \
    stochastic_inverse_hessian_vector_product


def compute_influence_function(
    trainer,
    test_gradient,
    training_sample_gradient_dict: SyncedTensorDict,
    batch_size=None,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
) -> dict:
    training_dataset_size = len(trainer.dataset)
    if batch_size is None:
        batch_size = trainer.hyper_parameter.batch_size
    product = (
        stochastic_inverse_hessian_vector_product(
            trainer.dataset,
            trainer.copy_model_with_loss(True),
            test_gradient,
            repeated_num=3,
            max_iteration=None,
            batch_size=batch_size,
            dampling_term=dampling_term,
            scale=scale,
            epsilon=epsilon,
        )
        / training_dataset_size
    )

    return {
        sample_index: (product @ sample_gradient).data.item()
        for (sample_index, sample_gradient) in training_sample_gradient_dict.iterate()
    }
