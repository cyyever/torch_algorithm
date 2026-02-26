import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.cuda
from cyy_preprocessing_pipeline import cat_tensor_dict
from cyy_torch_toolbox import ModelEvaluator, ModelParameter
from torch.func import grad, jvp, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gjvp_worker_fun(
    vector: torch.Tensor,
    model_evaluator: ModelEvaluator,
    parameters: ModelParameter,
    sample_indices: list[int],
    inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    worker_device: torch.device,
) -> dict[int, torch.Tensor]:
    def jvp_wrapper(parameters, input_tensor, target):
        f = functools.partial(
            eval_model,
            inputs=input_tensor,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
        )

        def grad_f(input_tensor):
            return cat_tensor_dict(grad(f, argnums=0)(parameters))

        return jvp(grad_f, (input_tensor,), (vector,))[1]

    products = vmap(jvp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameters,
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, products, strict=True))


class SampleGradientJVPHook(SampleComputationHook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__vector: torch.Tensor | None = None

    def set_vector(self, vector: torch.Tensor) -> None:
        self.__vector = vector

    def _get_sample_computation_fun(self) -> Callable[..., Any]:
        assert self.__vector is not None
        return functools.partial(sample_gjvp_worker_fun, self.__vector)
