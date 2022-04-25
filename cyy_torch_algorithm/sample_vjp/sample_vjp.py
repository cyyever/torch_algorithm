#!/usr/bin/env python3
import functools
import threading

import torch.autograd
from evaluation import eval_model_foreach
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
    raw_input_chunk = [t.to(worker_device) for t in input_chunk]
    if forward_embedding:
        input_chunk = [
            model_with_loss.model.get_embedding(raw_input_tensor).detach()
            for raw_input_tensor in raw_input_chunk
        ]
    f = functools.partial(
        eval_model_foreach,
        targets=target_chunk,
        device=worker_device,
        model_with_loss=model_with_loss,
        input_shape=input_chunk[0].shape,
        model_util=model_with_loss.model_util,
        forward_embedding=forward_embedding,
    )

    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)

    def grad_f(*input_tensors):
        return grad(f, argnums=0)(parameter_list, input_tensors).view(-1)

    products = product = torch.autograd.functional.vjp(
        func=grad_f, inputs=tuple(t.view(-1) for t in input_chunk), v=vector
    )[1]

    result = {}
    for index, raw_input_tensor, input_tensor, product in zip(
        sample_indices, raw_input_chunk, input_chunk, products
    ):
        if product_transform is not None:
            product = product_transform(index, raw_input_tensor, input_tensor, product)
        result[index] = product
    return result
