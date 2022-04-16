#!/usr/bin/env python3
import functools
import threading

import torch.autograd
from cyy_torch_toolbox.tensor import cat_tensors_to_vector
from evaluation import eval_model_by_parameter
from functorch import grad

local_data = threading.local()


def sample_sparse_jvp_worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    (
        model_with_loss,
        (sample_indices, input_chunk, target_chunk, vector_chunk),
        extra_args,
    ) = task
    dot_vector = extra_args.get("dot_vector", None)
    if dot_vector is not None:
        dot_vector = dot_vector.to(worker_device)
    model_with_loss.model.to(worker_device)
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
    for index, input_tensor, target, vectors in zip(
        sample_indices, input_chunk, target_chunk, vector_chunk
    ):
        input_tensor = input_tensor.to(worker_device)
        target = target.to(worker_device)
        input_shape = input_tensor.shape

        result[index] = []
        for part_id, partial_range, partial_vector in vectors:
            i, j = partial_range

            def grad_f(partial_input_tensor):
                nonlocal partial_range
                nonlocal input_tensor
                new_tensor = cat_tensors_to_vector(
                    [
                        input_tensor.view(-1)[:i],
                        partial_input_tensor,
                        input_tensor.view(-1)[j:],
                    ]
                )
                return grad(f, argnums=0)(
                    parameter_list, new_tensor.view(input_shape), target
                ).view(-1)

            jvp_result = torch.autograd.functional.jvp(
                func=grad_f,
                inputs=input_tensor.view(-1)[i:j],
                v=partial_vector.to(worker_device),
                strict=True,
            )[1].detach()
            if dot_vector is not None:
                jvp_result = dot_vector.dot(jvp_result)
            result[index].append((part_id, jvp_result))
    return result
