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
        input_shape = input_tensor.shape
        input_tensor = input_tensor.view(-1)
        target = target.to(worker_device)

        result[index] = []
        for part_id, partial_ranges, partial_vectors in vectors:
            inputs = tuple(input_tensor[i:j] for i, j in partial_ranges)
            v = tuple(p.to(worker_device) for p in partial_vectors)

            def grad_f(*partial_inputs):
                nonlocal input_tensor
                nonlocal partial_ranges
                start_idx = 0
                tensor_list = []
                for partial_range, partial_input in zip(partial_ranges, partial_inputs):
                    i, j = partial_range
                    if i > start_idx:
                        tensor_list.append(input_tensor[start_idx:i])
                    tensor_list.append(partial_input)
                    start_idx = j
                end_tensor = input_tensor[start_idx:]
                if end_tensor.nelement() > 0:
                    tensor_list.append(end_tensor)
                new_tensor = cat_tensors_to_vector(tensor_list)
                return grad(f, argnums=0)(
                    parameter_list, new_tensor.view(input_shape), target
                ).view(-1)

            jvp_result = (
                torch.autograd.functional.jvp(
                    func=grad_f,
                    inputs=inputs,
                    v=v,
                    strict=True,
                )[1]
                .sum()
                .detach()
            )
            if dot_vector is not None:
                jvp_result = dot_vector.dot(jvp_result)
            result[index].append((part_id, jvp_result))
    return result
