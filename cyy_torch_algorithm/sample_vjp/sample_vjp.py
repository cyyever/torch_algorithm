#!/usr/bin/env python3
import functools
import threading

import torch.autograd
from evaluation import eval_model_by_parameter
from functorch import grad

local_data = threading.local()


def sample_vjp_worker_fun(product_transform, vector, task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    model_with_loss, (sample_indices, input_chunk, target_chunk) = task
    model_with_loss.model.to(worker_device)
    vector = vector.to(worker_device)
    forward_embedding = hasattr(model_with_loss.model, "forward_embedding")
    f = functools.partial(
        eval_model_by_parameter,
        device=worker_device,
        model_with_loss=model_with_loss,
        model_util=model_with_loss.model_util,
        forward_embedding=forward_embedding,
    )
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    result = {}
    for index, raw_input_tensor, target in zip(
        sample_indices, input_chunk, target_chunk
    ):
        raw_input_tensor = raw_input_tensor.to(worker_device)
        if forward_embedding:
            input_tensor = model_with_loss.model.get_embedding(
                raw_input_tensor
            ).detach()
        else:
            input_tensor = raw_input_tensor
        input_shape = input_tensor.shape
        input_tensor = input_tensor.view(-1)

        def grad_f(input_tensor):
            return grad(f, argnums=0)(
                parameter_list, input_tensor.view(input_shape), target
            ).view(-1)

        product = torch.autograd.functional.vjp(
            func=grad_f, inputs=input_tensor, v=vector
        )[1]
        if product_transform is not None:
            product = product_transform(index, raw_input_tensor, input_tensor, product)
        result[index] = product
    return result
