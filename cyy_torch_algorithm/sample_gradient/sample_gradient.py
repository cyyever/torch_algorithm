#!/usr/bin/env python3

import functools
import threading

import torch
from evaluation import eval_model_by_parameter
from functorch import grad, vmap

local_data = threading.local()


def sample_gradient_worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    (index, input_chunk, target_chunk, model_with_loss) = task
    gradient_lists = vmap(
        grad(
            functools.partial(
                eval_model_by_parameter,
                device=worker_device,
                model_with_loss=model_with_loss,
            )
        ),
        in_dims=(None, 0, 0),
        randomness="different",
    )(
        model_with_loss.model_util.get_parameter_list(detach=True),
        torch.stack(input_chunk),
        torch.stack(target_chunk),
    )
    return (index, gradient_lists)
