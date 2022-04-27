#!/usr/bin/env python3
import functools
import threading

import torch.autograd
import torch.cuda
from cyy_torch_toolbox.device import put_data_to_device
from evaluation import eval_model_foreach
from functorch import grad

local_data = threading.local()


def sample_vjp_worker_fun(product_transform, vector, task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    worker_stream = getattr(local_data, "worker_stream", None)
    if worker_stream is None:
        worker_stream = torch.cuda.Stream(device=worker_device)
        local_data.worker_stream = worker_stream
    model_with_loss, (
        sample_indices,
        input_chunk,
        input_feature_chunk,
        target_chunk,
    ) = task
    model_with_loss.model.to(worker_device)
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    with torch.cuda.stream(worker_stream):
        vector = vector.to(worker_device, non_blocking=True)
        is_input_feature = input_feature_chunk[0] is not None
        raw_input_chunk = put_data_to_device(
            input_chunk, device=worker_device, non_blocking=True
        )
        if is_input_feature:
            input_chunk = put_data_to_device(
                input_feature_chunk, device=worker_device, non_blocking=True
            )
        else:
            input_chunk = raw_input_chunk
        f = functools.partial(
            eval_model_foreach,
            targets=target_chunk,
            device=worker_device,
            model_with_loss=model_with_loss,
            input_shape=input_chunk[0].shape,
            model_util=model_with_loss.model_util,
            is_input_feature=is_input_feature,
            non_blocking=True,
        )

        def grad_f(*input_tensors):
            return grad(f, argnums=0)(parameter_list, input_tensors).view(-1)

        products = product = torch.autograd.functional.vjp(
            func=grad_f, inputs=tuple(t.view(-1) for t in input_chunk), v=vector
        )[1]
        worker_stream.synchronize()

    result = {}
    for index, raw_input_tensor, input_tensor, product in zip(
        sample_indices, raw_input_chunk, input_chunk, products
    ):
        if product_transform is not None:
            product = product_transform(index, raw_input_tensor, input_tensor, product)
        result[index] = product
    return result
