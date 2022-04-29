#!/usr/bin/env python3

import functools
import threading

import torch
from evaluation import eval_model
from functorch import grad, vmap

local_data = threading.local()


def sample_gradient_worker_fun(gradient_transform, task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    model_with_loss, (
        sample_indices,
        input_chunk,
        input_feature_chunk,
        target_chunk,
    ) = task
    is_input_feature = input_feature_chunk[0] is not None
    gradient_lists = vmap(
        grad(
            functools.partial(
                eval_model,
                device=worker_device,
                model_with_loss=model_with_loss,
                model_util=model_with_loss.model_util,
                input_feature_chunk=input_feature_chunk,
            )
        ),
        in_dims=(None, 0, 0),
        randomness="different",
    )(
        model_with_loss.model_util.get_parameter_list(detach=True),
        torch.stack(input_feature_chunk)
        if is_input_feature
        else torch.stack(input_chunk),
        torch.stack(target_chunk),
    )
    res = dict(zip(sample_indices, gradient_lists))
    if gradient_transform is not None:
        res = {k: gradient_transform(k, v) for k, v in res.items()}
    return res
