import functools

import torch
import torch.cuda
from cyy_torch_toolbox.tensor import cat_tensor_dict
from torch.func import grad, jvp, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gjvp_worker_fun(
    vector,
    model_evaluator,
    parameter_dict,
    sample_indices,
    inputs,
    targets,
    worker_device,
    is_input_feature,
    **kwargs
):
    def jvp_wrapper(parameter_dict, input_tensor, target):
        f = functools.partial(
            eval_model,
            inputs=input_tensor,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
            input_shape=inputs[0].shape,
            is_input_feature=is_input_feature,
        )

        def grad_f(input_tensor):
            return cat_tensor_dict(grad(f, argnums=0)(parameter_dict))

        return jvp(grad_f, (input_tensor.view(-1),), (vector,))[1]

    products = vmap(jvp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameter_dict,
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, products))


class SampleGradientJVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_sample_computation_fun(self):
        return functools.partial(sample_gjvp_worker_fun, self.__vector)
