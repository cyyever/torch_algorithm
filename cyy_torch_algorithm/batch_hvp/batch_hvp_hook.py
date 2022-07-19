import functools

import torch
import torch.cuda
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.device import put_data_to_device
from functorch import grad, jvp, vmap


def batch_hvp_worker_fun(
    vectors,
    model_with_loss,
    sample_indices,
    inputs,
    input_features,
    targets,
    worker_device,
    worker_stream,
):
    model_with_loss.model.to(worker_device)
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    with torch.cuda.stream(worker_stream):
        vectors = put_data_to_device(vectors, device=worker_device, non_blocking=True)
        is_input_feature = input_features[0] is not None
        if is_input_feature:
            inputs = input_features
        inputs = put_data_to_device(
            torch.stack(inputs).squeeze(dim=1), device=worker_device, non_blocking=True
        )
        targets = put_data_to_device(
            torch.stack(targets), device=worker_device, non_blocking=True
        )

        def vjp_wrapper(parameter_list, vector):
            return jvp(
                grad(
                    functools.partial(
                        eval_model,
                        inputs=inputs,
                        targets=targets,
                        device=worker_device,
                        model_with_loss=model_with_loss,
                        is_input_feature=is_input_feature,
                        non_blocking=True,
                    )
                ),
                (parameter_list,),
                (vector,),
            )[1]

        products = vmap(vjp_wrapper, in_dims=(None, 0), randomness="different",)(
            parameter_list,
            torch.stack(vectors),
        )
        return {0: products}


class BatchHVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vectors = None

    def set_vectors(self, vectors):
        self.__vectors = vectors

    def _get_worker_fun(self):
        return functools.partial(batch_hvp_worker_fun, self.__vectors)