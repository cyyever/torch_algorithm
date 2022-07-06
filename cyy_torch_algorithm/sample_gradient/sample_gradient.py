#!/usr/bin/env python3

import functools

import torch
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_algorithm.sample_computation_hook import setup_cuda_device
from functorch import grad, vmap


def sample_gradient_worker_fun(gradient_transform, task, args):
    worker_device, worker_stream = setup_cuda_device(args)
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
                is_input_feature=is_input_feature,
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
