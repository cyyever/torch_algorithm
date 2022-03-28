#!/usr/bin/env python3

import threading

# import functorch
import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
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


def __evaluation_wrapper(parameter_list, model_with_loss, inputs, targets, device):
    model_util = model_with_loss.model_util
    model_util.load_parameter_list(
        parameter_list,
        check_parameter=False,
        as_parameter=False,
    )
    return model_with_loss(
        inputs,
        targets,
        device=device,
        non_blocking=True,
        phase=MachineLearningPhase.Training,
    )["loss"]


def sample_gradient_worker_fun2(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    (index, input_chunk, target_chunk, model_with_loss) = task
    model_with_loss.model.zero_grad(set_to_none=True)
    parameter_list = model_with_loss.model_util.get_model_list()
    gradient_lists = vmap(grad(__evaluation_wrapper), in_dims=(None, 0, 0))(
        parameter_list, model_with_loss, input_chunk, target_chunk, worker_device
    )
    return (index, gradient_lists)
    # inputs = (weights, examples, targets)
    # gradient_lists = []
    # if worker_device.index is not None:
    #     local_data.worker_stream = torch.cuda.Stream(device=worker_device)
    # worker_stream = getattr(local_data, "worker_stream", None)
    # phase = MachineLearningPhase.Training
    # loss = model_with_loss(
    #     sample_input,
    #     sample_target,
    #     phase=phase,
    #     device=worker_device,
    #     non_blocking=True,
    # )["loss"]

    # # with torch.cuda.stream(worker_stream):
    # loss = None

    # for (sample_input, sample_target) in zip(input_chunk, target_chunk):
    #     loss.backward()
    #     gradient_lists.append(model_with_loss.model_util.get_gradient_list().cpu())
    #     # assert len(gradient_lists) == len(input_chunk)
