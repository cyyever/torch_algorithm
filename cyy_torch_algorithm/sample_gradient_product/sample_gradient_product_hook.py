import copy
import functools

import torch
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.device import put_data_to_device
from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint
from functorch import grad, vmap


def sample_gradient_product_worker_fun(
    vector,
    model_with_loss,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
    worker_stream,
):
    vector = put_data_to_device(vector, worker_device)
    is_input_feature = input_features[0] is not None
    gradient_lists = vmap(
        grad(
            functools.partial(
                eval_model,
                device=worker_device,
                model_with_loss=model_with_loss,
                is_input_feature=is_input_feature,
            )
        ),
        in_dims=(None, 0, 0),
        randomness="same",
    )(
        model_with_loss.model_util.get_parameter_list(detach=True).to(worker_device),
        torch.stack(input_features) if is_input_feature else torch.stack(inputs),
        torch.stack(targets),
    )
    return {
        idx: vector @ gradient for idx, gradient in zip(sample_indices, gradient_lists)
    }


class SampleGradientProductHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_worker_fun(self):
        return functools.partial(sample_gradient_product_worker_fun, self.__vector)


def get_sample_gradient_product_dict(
    inferencer, vector, computed_indices=None, input_transform=None
) -> dict:
    tmp_inferencer = copy.deepcopy(inferencer)
    hook = SampleGradientProductHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    hook.set_input_transform(input_transform)
    hook.set_vector(vector)
    products: dict = {}
    tmp_inferencer.append_hook(hook)

    def collect_gradients(**kwargs):
        nonlocal products
        products |= hook.sample_result_dict

    tmp_inferencer.append_named_hook(
        hook_point=ModelExecutorHookPoint.AFTER_FORWARD,
        name="collect_gradients",
        fun=collect_gradients,
    )
    tmp_inferencer.inference(use_grad=True)
    assert products
    return products
