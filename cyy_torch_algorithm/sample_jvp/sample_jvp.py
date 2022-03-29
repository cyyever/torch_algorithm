#!/usr/bin/env python3

import functools
import threading

from evaluation import eval_model_by_parameter
from functorch import grad

local_data = threading.local()


def sample_jvp_worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    model_with_loss, (sample_indices, input_chunk, target_chunk, vector_chunk) = task
    f = functools.partial(
        eval_model_by_parameter,
        device=worker_device,
        model_with_loss=model_with_loss,
    )
    parameter_list = (model_with_loss.model_util.get_parameter_list(detach=True),)
    result = {}
    for index, input_tensor, target, vectors in zip(
        sample_indices, input_chunk, target_chunk, vector_chunk
    ):
        result[index] = []
        # We only support sparse vectors
        for vector in vectors:
            elm_idx, elm = vector
            result[index].append(
                grad(grad(f, argnums=0), argnums=3)(
                    parameter_list, input_tensor, target, input_tensor[elm_idx]
                )
                * elm
            )
    return result
