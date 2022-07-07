import functools

import torch.cuda
from cyy_torch_algorithm.sample_computation_hook import setup_cuda_device
from cyy_torch_toolbox.device import put_data_to_device
from evaluation import eval_model
from functorch import grad, jvp, vmap


def sample_jvp_worker_fun(product_transform, vector, task, args):
    worker_device, worker_stream = setup_cuda_device(args)
    model_with_loss, (
        sample_indices,
        input_chunk,
        input_feature_chunk,
        target_chunk,
    ) = task
    model_with_loss.model.to(worker_device)
    parameter_list = model_with_loss.model_util.get_parameter_list()
    with torch.cuda.stream(worker_stream):
        vector = put_data_to_device(vector, device=worker_device, non_blocking=True)
        is_input_feature = input_feature_chunk[0] is not None
        raw_input_chunk = input_chunk
        if is_input_feature:
            input_chunk = input_feature_chunk
        input_chunk = put_data_to_device(
            input_chunk, device=worker_device, non_blocking=True
        )

        def jvp_wrapper(parameter_list, input_tensor, target):
            f = functools.partial(
                eval_model,
                targets=target,
                device=worker_device,
                model_with_loss=model_with_loss,
                input_shape=input_chunk[0].shape,
                is_input_feature=is_input_feature,
                non_blocking=True,
            )

            def grad_f(input_tensor):
                return grad(f, argnums=0)(parameter_list, input_tensor).view(-1)

            return jvp(grad_f, (input_tensor.view(-1),), (vector,))[1]

        products = vmap(jvp_wrapper, in_dims=(None, 0, 0), randomness="different",)(
            parameter_list,
            torch.stack(input_feature_chunk)
            if is_input_feature
            else torch.stack(input_chunk),
            torch.stack(target_chunk),
        )

        result = {}
        for index, input_tensor, input_feature_tensor, product in zip(
            sample_indices, raw_input_chunk, input_feature_chunk, products
        ):
            if product_transform is not None:
                product = product_transform(
                    index, input_tensor, input_feature_tensor, product
                )
            result[index] = product
    return result