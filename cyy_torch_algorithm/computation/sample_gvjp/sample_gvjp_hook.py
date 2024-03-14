import functools

import torch
from cyy_torch_toolbox.tensor import cat_tensor_dict
from torch.func import grad, vjp, vmap

from ..evaluation import eval_model
from ..sample_computation_hook import SampleComputationHook


def sample_gvjp_worker_fun(
    vector,
    model_evaluator,
    sample_indices,
    inputs,
    targets,
    worker_device,
    parameter_dict,
    **kwargs,
) -> dict:
    hugging_face_batch_encoding: None | dict = None
    if isinstance(inputs[0], dict):
        hugging_face_batch_encoding = inputs[0]
        new_inputs = [i.copy() for i in inputs]
        inputs = [i.pop("inputs_embeds") for i in new_inputs]
    input_shape = inputs[0].shape

    def vjp_wrapper(parameter_dict, input_tensor, target):
        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_evaluator=model_evaluator,
            input_shape=input_shape,
            hugging_face_batch_encoding=hugging_face_batch_encoding,
        )

        def grad_f(input_tensor):
            return cat_tensor_dict(
                grad(f, argnums=0)(parameter_dict, inputs=input_tensor)
            )

        vjpfunc = vjp(grad_f, input_tensor.view(-1))[1]
        return vjpfunc(vector)[0]

    products = vmap(vjp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameter_dict,
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, products))


class SampleGradientVJPHook(SampleComputationHook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_sample_computation_fun(self):
        return functools.partial(sample_gvjp_worker_fun, self.__vector)
