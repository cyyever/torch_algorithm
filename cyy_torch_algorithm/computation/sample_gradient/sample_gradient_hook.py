import copy
import functools
from collections.abc import Callable
from typing import Any

import torch
from cyy_torch_toolbox import (
    IndicesType,
    Inferencer,
    ModelEvaluator,
    ModelGradient,
    ModelParameter,
    OptionalIndicesType,
    TensorDict,
)
from cyy_torch_toolbox.tensor import dot_product
from torch.func import grad, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gradient_worker_fun(
    model_evaluator: ModelEvaluator,
    sample_indices: IndicesType,
    inputs: list[torch.Tensor] | list[TensorDict],
    targets: list[torch.Tensor],
    worker_device: torch.device,
    parameters: ModelParameter,
) -> dict[int, ModelGradient]:
    def wrapper(parameters, target, *args, input_keys=None):
        if input_keys is not None:
            inputs = dict(zip(input_keys, args, strict=False))
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
        return grad(f, argnums=0)(parameters)

    match inputs[0]:
        case torch.Tensor():
            gradient_dicts = vmap(
                wrapper,
                in_dims=(None, 0, 0),
                randomness="same",
            )(
                parameters,
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
            )(parameters, torch.stack(targets), *dict_inputs)
        case _:
            raise NotImplementedError(inputs)
    result: dict[int, ModelGradient] = {}
    for idx, sample_idx in enumerate(sample_indices):
        result[sample_idx] = {}
        for k, v in gradient_dicts.items():
            result[sample_idx][k] = v[idx]
    return result


class SampleGradientHook(SampleComputationHook):
    def _get_sample_computation_fun(self):
        return sample_gradient_worker_fun


def get_sample_gradients_impl(
    inferencer: Inferencer,
    computed_indices: OptionalIndicesType = None,
    result_transform: None | Callable = None,
) -> dict[int, ModelGradient] | dict[int, Any]:
    tmp_inferencer = copy.deepcopy(inferencer)
    tmp_inferencer.hook_config.use_performance_metric = False
    tmp_inferencer.hook_config.summarize_executor = False
    hook = SampleGradientHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    if result_transform is not None:
        hook.set_result_transform(result_transform)
    tmp_inferencer.append_hook(hook)
    tmp_inferencer.inference()
    gradients = {
        k: {name: tensor.cpu() for name, tensor in v.items()}
        for k, v in hook.result_dict.items()
    }
    assert gradients
    hook.release()
    return gradients


def get_sample_gradients(
    inferencer: Inferencer,
    computed_indices: OptionalIndicesType = None,
) -> dict[int, ModelGradient]:
    return get_sample_gradients_impl(
        inferencer=inferencer, computed_indices=computed_indices
    )


def get_sample_gvps(vector, **kwargs) -> dict[int, float]:
    return get_sample_gradients_impl(
        result_transform=functools.partial(dot_product, b=vector), **kwargs
    )


def __get_self_product(
    vectors: dict[int, TensorDict], result: TensorDict, sample_index: int, **kwargs
) -> float:
    return dot_product(result, vectors[sample_index])


def get_self_gvps(vectors: dict[int, TensorDict], **kwargs) -> dict[int, float]:
    return get_sample_gradients_impl(
        result_transform=functools.partial(__get_self_product, vectors),
        computed_indices=set(vectors.keys()),
        **kwargs,
    )
