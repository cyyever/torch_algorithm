#!/usr/bin/env python3

import threading

import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_util import ModelUtil


def sample_gradient_worker_fun(task, args):
    worker_device = getattr(threading.local(), "worker_device", None)
    worker_stream = None
    if worker_device is None:
        worker_device = args["device"]
        threading.local().worker_device = worker_device
        if worker_device.index is not None:
            worker_stream = torch.cuda.Stream(device=worker_device)
        threading.local().worker_stream = worker_stream
    (index, input_chunk, target_chunk, model_with_loss) = task
    gradient_lists = []
    with torch.cuda.stream(worker_stream):
        loss = None
        model_util = ModelUtil(model_with_loss.model)
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
            )["loss"]
            loss.backward()
            gradient_lists.append(model_util.get_gradient_list())
        assert len(gradient_lists) == len(input_chunk)
    return (index, gradient_lists)
