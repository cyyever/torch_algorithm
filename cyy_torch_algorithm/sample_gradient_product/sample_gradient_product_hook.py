import copy
import functools

import torch
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.device import put_data_to_device
from functorch import grad, vmap


def sample_gradient_product_worker_fun(
    vector,
    model_with_loss,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
):
    vector = put_data_to_device(vector, worker_device, non_blocking=True)
    gradient_lists = vmap(
        grad(
            functools.partial(
                eval_model,
                device=worker_device,
                model_with_loss=model_with_loss,
            )
        ),
        in_dims=(None, 0, 0),
        randomness="same",
    )(
        model_with_loss.model_util.get_parameter_list(detach=True),
        torch.stack(
            put_data_to_device(inputs, device=worker_device, non_blocking=True)
        ),
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
    inferencer,
    vector,
    computed_indices=None,
    sample_selector=None,
    input_transform=None,
) -> dict:
    tmp_inferencer = copy.deepcopy(inferencer)
    tmp_inferencer.disable_logger()
    hook = SampleGradientProductHook()
    hook.set_vector(vector)
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    if sample_selector is not None:
        hook.set_sample_selector(sample_selector)
    hook.set_input_transform(input_transform)
    tmp_inferencer.append_hook(hook)

    tmp_inferencer.inference(use_grad=False)
    hook.release_queue()
    products = hook.result_dict
    assert products
    return products
