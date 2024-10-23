import functools
from collections.abc import Callable

import torch
import torch.cuda
from cyy_torch_toolbox import ModelParameter, cat_tensor_dict
from torch.func import grad, jvp, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gjvp_worker_fun(
    vector,
    model_evaluator,
    parameters: ModelParameter,
    sample_indices,
    inputs: list[torch.Tensor],
    targets: list[torch.Tensor],
    worker_device,
) -> dict:
    def jvp_wrapper(parameters, input_tensor, target):
        f = functools.partial(
            eval_model,
            inputs=input_tensor,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
            input_shape=inputs[0].shape,
        )

        def grad_f(input_tensor):
            return cat_tensor_dict(grad(f, argnums=0)(parameters))

        return jvp(grad_f, (input_tensor.view(-1),), (vector,))[1]

    products = vmap(jvp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameters,
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, products, strict=False))


class SampleGradientJVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_sample_computation_fun(self) -> Callable:
        return functools.partial(sample_gjvp_worker_fun, self.__vector)
