import copy
import functools

import torch
from cyy_torch_algorithm.computation.evaluation import eval_model
from cyy_torch_algorithm.computation.sample_computation_hook import (
    SampleComputationHook, sample_dot_product)

try:
    from torch.func import grad, vmap
except BaseException:
    from functorch import grad, vmap


def sample_gradient_worker_fun(
    model_with_loss,
    parameter_list,
    parameter_shapes,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
):
    def wrapper(parameter_list, target, *args, input_keys=None):
        nonlocal parameter_shapes
        input_kwargs = {}
        if input_keys is not None:
            input_kwargs = dict(zip(input_keys, args))
        else:
            assert len(args) == 1
            input_kwargs["inputs"] = args[0]

        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_with_loss=model_with_loss,
            parameter_shapes=parameter_shapes,
            non_blocking=True,
            **input_kwargs
        )
        return grad(f, argnums=0)(parameter_list).view(-1)

    match inputs[0]:
        case torch.Tensor():
            gradient_lists = vmap(wrapper, in_dims=(None, 0, 0), randomness="same",)(
                parameter_list,
                torch.stack(targets),
                torch.stack(inputs),
            )
        case dict():
            input_keys = list(inputs[0].keys())
            dict_inputs = []
            for k in input_keys:
                dict_inputs.append(torch.stack([a[k] for a in inputs]))
            in_dims = [0] * (len(dict_inputs) + 2)
            in_dims[0] = None
            gradient_lists = vmap(
                functools.partial(wrapper, input_keys=input_keys),
                in_dims=tuple(in_dims),
                randomness="same",
            )(parameter_list, torch.stack(targets), *dict_inputs)

    return dict(zip(sample_indices, gradient_lists))


class SampleGradientHook(SampleComputationHook):
    def _get_sample_computation_fun(self):
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
    tmp_inferencer.disable_hook("logger")
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
    gradients = hook.result_dict
    if result_collection_fun is None:
        assert gradients
    hook.release_queue()
    return gradients


def get_sample_gradient_product_dict(vector, **kwargs) -> dict | None:
    return get_sample_gradient_dict(
        result_transform=functools.partial(sample_dot_product, vector=vector), **kwargs
    )
