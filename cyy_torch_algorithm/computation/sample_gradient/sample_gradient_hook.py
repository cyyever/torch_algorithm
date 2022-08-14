import copy
import functools

import torch
from cyy_torch_algorithm.computation.evaluation import eval_model
from cyy_torch_algorithm.computation.sample_computation_hook import (
    SampleComputationHook, sample_dot_product)
from functorch import grad, vmap


def sample_gradient_worker_fun(
    model_with_loss,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
):
    gradient_lists = vmap(
        grad(
            functools.partial(
                eval_model, device=worker_device, model_with_loss=model_with_loss
            )
        ),
        in_dims=(None, 0, 0),
        randomness="same",
    )(
        model_with_loss.model_util.get_parameter_list(detach=False),
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, gradient_lists))


class SampleGradientHook(SampleComputationHook):
    def _get_worker_fun(self):
        return sample_gradient_worker_fun


def get_sample_gradient_dict(
    inferencer,
    computed_indices=None,
    sample_selector=None,
    input_transform=None,
    result_transform=None,
    result_collection_fun=None,
) -> dict | None:
    tmp_inferencer = copy.deepcopy(inferencer)
    tmp_inferencer.disable_logger()
    hook = SampleGradientHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    if sample_selector is not None:
        hook.set_sample_selector(sample_selector)
    if input_transform is not None:
        hook.set_input_transform(input_transform)
    if result_transform is not None:
        hook.set_result_transform(result_transform)
    if result_collection_fun is not None:
        hook.set_result_collection_fun(result_collection_fun)
    tmp_inferencer.append_hook(hook)
    tmp_inferencer.inference()
    gradients = None
    if result_collection_fun is None:
        gradients = hook.result_dict
        assert gradients
    hook.release_queue()
    return gradients


def get_sample_gradient_product_dict(vector, **kwargs):
    return get_sample_gradient_dict(
        result_transform=functools.partial(sample_dot_product, vector=vector)
    )
