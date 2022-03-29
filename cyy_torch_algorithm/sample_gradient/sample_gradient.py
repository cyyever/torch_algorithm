#!/usr/bin/env python3

import functools
import threading

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from evaluation import eval_model_by_parameter
from functorch import grad, vmap

local_data = threading.local()


def sample_gradient_worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
        if worker_device.index is not None:
            local_data.worker_stream = torch.cuda.Stream(device=worker_device)
    worker_stream = getattr(local_data, "worker_stream", None)
    (index, input_chunk, target_chunk, model_with_loss) = task
    gradient_lists = []
    with torch.cuda.stream(worker_stream):
        loss = None
        for (sample_input, sample_target) in zip(input_chunk, target_chunk):
            model_with_loss.model.zero_grad(set_to_none=True)
            phase = MachineLearningPhase.Training
            loss = model_with_loss(
                sample_input,
                sample_target,
                phase=phase,
                device=worker_device,
                non_blocking=True,
            )["loss"]
            loss.backward()
            gradient_lists.append(model_with_loss.model_util.get_gradient_list().cpu())
        assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)


def sample_gradient_worker_fun2(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    (index, input_chunk, target_chunk, model_with_loss) = task
    model_with_loss.model.zero_grad(set_to_none=True)
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    input_chunk = torch.stack(input_chunk)
    target_chunk = torch.stack(target_chunk)
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
    )(parameter_list, input_chunk, target_chunk)
    return (index, gradient_lists)
