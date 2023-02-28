import functools

import torch
import torch.cuda
from cyy_torch_algorithm.computation.evaluation import eval_model
from cyy_torch_algorithm.computation.sample_computation_hook import \
    SampleComputationHook

try:
    from torch.func import grad, vjp, vmap
except BaseException:
    from functorch import grad, vjp, vmap


def sample_gvjp_worker_fun(
    vector,
    model_with_loss,
    parameter_list,
    parameter_shapes,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
):
    is_input_feature = input_features[0] is not None
    if is_input_feature:
        inputs = input_features

    def vjp_wrapper(parameter_list, input_tensor, target):
        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_with_loss=model_with_loss,
            input_shape=inputs[0].shape,
            is_input_feature=is_input_feature,
            non_blocking=True,
            parameter_shapes=parameter_shapes,
        )

        def grad_f(input_tensor):
            return grad(f, argnums=0)(parameter_list, inputs=input_tensor).view(-1)

        vjpfunc = vjp(grad_f, input_tensor.view(-1))[1]
        return vjpfunc(vector)[0]

    products = vmap(vjp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameter_list,
        torch.stack(inputs),
        torch.stack(targets),
    )
    return dict(zip(sample_indices, products))


class SampleGradientVJPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_sample_computation_fun(self):
        return functools.partial(sample_gvjp_worker_fun, self.__vector)
