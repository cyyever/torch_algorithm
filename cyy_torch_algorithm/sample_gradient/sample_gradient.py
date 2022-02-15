#!/usr/bin/env python3

import threading

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase

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
            sample_input.unsqueeze_(0)
            sample_target.unsqueeze_(0)
            # we should set phase to test so that BatchNorm would use running statistics
            loss = model_with_loss(
                sample_input,
                sample_target,
                phase=MachineLearningPhase.Test,
                device=worker_device,
                non_blocking=True,
            )["loss"]
            loss.backward()
            gradient_lists.append(model_with_loss.model_util.get_gradient_list())
        assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)
