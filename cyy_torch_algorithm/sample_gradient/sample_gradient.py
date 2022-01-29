#!/usr/bin/env python3


import torch
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_util import ModelUtil

__worker_stream = None
__worker_device = None


def sample_gradient_worker_fun(task, args):
    global __worker_stream
    global __worker_device
    if __worker_stream is None:
        device = args["device"]
        __worker_device = device
        torch.cuda.set_device(device)
        __worker_stream = torch.cuda.Stream(device)
    with torch.cuda.stream(__worker_stream):
        (index, input_chunk, target_chunk, model_with_loss) = task

        loss = None
        gradient_lists = []
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
                device=__worker_device,
            )["loss"]
            loss.backward()
            gradient_lists.append(model_util.get_gradient_list())
        assert len(gradient_lists) == len(input_chunk)
        __worker_stream.synchronize()
        return (index, gradient_lists)
