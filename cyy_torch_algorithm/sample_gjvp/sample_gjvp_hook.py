import functools

import torch
import torch.cuda
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.device import put_data_to_device
from functorch import grad, jvp, vmap


def sample_gjvp_worker_fun(
    vector,
    model_with_loss,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
):
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    vector = put_data_to_device(vector, device=worker_device, non_blocking=True)
    is_input_feature = input_features[0] is not None
    if is_input_feature:
        inputs = input_features
    inputs = put_data_to_device(inputs, device=worker_device, non_blocking=True)

    def jvp_wrapper(parameter_list, input_tensor, target):
        f = functools.partial(
            eval_model,
            targets=target,
            device=worker_device,
            model_with_loss=model_with_loss,
            input_shape=inputs[0].shape,
            is_input_feature=is_input_feature,
            non_blocking=True,
        )

        def grad_f(input_tensor):
            return grad(f, argnums=0)(parameter_list, input_tensor).view(-1)

        return jvp(grad_f, (input_tensor.view(-1),), (vector,))[1]

    products = vmap(jvp_wrapper, in_dims=(None, 0, 0), randomness="same")(
        parameter_list,
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

    def _get_worker_fun(self):
        return functools.partial(sample_gjvp_worker_fun, self.__vector)
