import functools
from collections.abc import Callable
from typing import Any

import torch
from cyy_torch_toolbox import (
    ModelEvaluator,
    ModelParameter,
    cat_tensor_dict,
)
from torch.func import grad, vjp, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gvjp_worker_fun(
    vector: torch.Tensor,
    model_evaluator: ModelEvaluator,
    sample_indices: list[int],
    inputs: list[torch.Tensor] | list[dict[str, torch.Tensor]],
    targets: list[torch.Tensor],
    worker_device: torch.device,
    parameters: ModelParameter,
    **kwargs: Any,
) -> dict[int, torch.Tensor]:
    input_list: list[torch.Tensor] = []
    input_keys: list[str] = []
    if isinstance(inputs[0], dict):
        input_key_sets = set(inputs[0].keys())
        for k in input_key_sets.copy():
            if "mask" in k:
                input_keys.append(k)
                input_key_sets.remove(k)
        assert len(input_key_sets) == 1
        input_keys += list(input_key_sets)
        for k in input_keys:
            input_list.append(torch.stack([a[k] for a in inputs]))
    else:
        input_list.append(torch.stack(inputs))

    def vjp_wrapper(parameters, target, *input_tensors):
        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
            input_keys=input_keys,
        )

        def grad_f(input_tensor):
            return cat_tensor_dict(
                grad(f, argnums=0)(
                    parameters, input_tensors=list(input_tensors[0:-1]) + [input_tensor]
                )
            )

        vjpfunc = vjp(grad_f, input_tensors[-1])[1]
        return vjpfunc(vector)[0]

    products = vmap(
        vjp_wrapper,
        in_dims=tuple([None] + [0] * (len(input_list) + 1)),
        randomness="same",
    )(parameters, torch.stack(targets), *input_list)
    return dict(zip(sample_indices, products, strict=True))


class SampleGradientVJPHook(SampleComputationHook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__vector: torch.Tensor | None = None

    def set_vector(self, vector: torch.Tensor) -> None:
        self.__vector = vector

    def _get_sample_computation_fun(self) -> Callable[..., Any]:
        assert self.__vector is not None
        return functools.partial(sample_gvjp_worker_fun, self.__vector)
