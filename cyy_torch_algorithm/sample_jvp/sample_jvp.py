#!/usr/bin/env python3
import functools
import threading

import torch.autograd
from evaluation import eval_model_by_parameter
from functorch import grad

local_data = threading.local()


def sample_jvp_worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    model_with_loss, (sample_indices, input_chunk, target_chunk, vector_chunk) = task
    forward_embedding = hasattr(model_with_loss.model, "forward_embedding")
    f = functools.partial(
        eval_model_by_parameter,
        device=worker_device,
        model_with_loss=model_with_loss,
        forward_embedding=forward_embedding,
    )
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    result = {}
    for index, input_tensor, target, vectors in zip(
        sample_indices, input_chunk, target_chunk, vector_chunk
    ):
        input_shape = input_tensor.shape
        print("input_tensor is",input_tensor)

        def grad_f(input_tensor):
            return grad(f, argnums=0)(
                parameter_list, input_tensor.view(input_shape), target
            ).view(-1)

        result[index] = []
        for vector in vectors:
            result[index].append(
                torch.autograd.functional.jvp(
                    func=grad_f, inputs=input_tensor.view(-1), v=vector
                )
            )
    return result
