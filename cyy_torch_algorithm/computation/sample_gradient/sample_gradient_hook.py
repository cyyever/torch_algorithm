import copy
import functools

import torch
from cyy_torch_toolbox.typing import TensorDict
from torch.func import grad, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook, dot_product


def sample_gradient_worker_fun(
    model_evaluator,
    sample_indices: list[int],
    inputs,
    targets,
    worker_device: torch.device,
    parameter_dict: TensorDict,
) -> dict[int, TensorDict]:
    def wrapper(parameter_dict, target, *args, input_keys=None):
        if input_keys is not None:
            inputs = dict(zip(input_keys, args))
        else:
            assert len(args) == 1
            inputs = args[0]

        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
            inputs=inputs,
        )
        return grad(f, argnums=0)(parameter_dict)

    match inputs[0]:
        case torch.Tensor():
            gradient_dicts = vmap(
                wrapper,
                in_dims=(None, 0, 0),
                randomness="same",
            )(
                parameter_dict,
                torch.stack(targets),
                torch.stack(inputs),
            )
        case dict():
            input_keys = list(inputs[0].keys())
            dict_inputs = []
            for k in input_keys:
                dict_inputs.append(torch.stack([a[k] for a in inputs]))
            in_dims: list[int | None] = [0] * (len(dict_inputs) + 2)
            in_dims[0] = None
            gradient_dicts = vmap(
                functools.partial(wrapper, input_keys=input_keys),
                in_dims=tuple(in_dims),
                randomness="same",
            )(parameter_dict, torch.stack(targets), *dict_inputs)
        case _:
            raise NotImplementedError(inputs)
    result: dict[int, TensorDict] = {}
    for idx, sample_idx in enumerate(sample_indices):
        result[sample_idx] = {}
        for k, v in gradient_dicts.items():
            result[sample_idx][k] = v[idx]
    return result


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
) -> dict:
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
    hook.reset()
    return gradients


def get_sample_gvp_dict(vector, **kwargs) -> dict:
    return get_sample_gradient_dict(
        result_transform=functools.partial(dot_product, rhs=vector), **kwargs
    )


def get_self_product(vectors, result, sample_index, **kwargs) -> float:
    return dot_product(result=result, rhs=vectors[sample_index])


def get_self_gvp_dict(vectors: dict, **kwargs) -> dict:
    return get_sample_gradient_dict(
        result_transform=functools.partial(get_self_product, vectors),
        computed_indices=set(vectors.keys()),
        **kwargs,
    )
