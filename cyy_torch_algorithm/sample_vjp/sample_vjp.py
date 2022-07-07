import functools

import torch.cuda
from cyy_torch_algorithm.sample_computation_hook import setup_cuda_device
from cyy_torch_toolbox.device import put_data_to_device
from evaluation import eval_model
from functorch import grad, vjp, vmap


def sample_vjp_worker_fun(
    vector,
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
        vector = put_data_to_device(vector, device=worker_device, non_blocking=True)
        is_input_feature = input_features[0] is not None
        if is_input_feature:
            inputs = input_features
        inputs = put_data_to_device(inputs, device=worker_device, non_blocking=True)

        def vjp_wrapper(parameter_list, input_tensor, target):
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

            vjpfunc = vjp(grad_f, input_tensor.view(-1))[1]
            return vjpfunc(vector)[0]

        products = vmap(vjp_wrapper, in_dims=(None, 0, 0), randomness="different",)(
            parameter_list,
            torch.stack(inputs),
            torch.stack(targets),
        )

        return dict(zip(sample_indices, products))
