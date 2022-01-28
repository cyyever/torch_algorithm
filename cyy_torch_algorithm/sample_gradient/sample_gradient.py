#!/usr/bin/env python3

import atexit

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.data_structure.torch_thread_task_queue import \
    TorchThreadTaskQueue
from cyy_torch_toolbox.device import get_cuda_devices
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

__worker_stream = None
__worker_device = None


def __worker_fun(task, args):
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


__task_queue = None


def stop_task_queue():
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_sample_gradient(model_with_loss: ModelWithLoss, inputs, targets):
    global __task_queue
    assert len(inputs) == len(targets)

    model_with_loss.model.zero_grad(set_to_none=True)

    if __task_queue is None:
        devices = get_cuda_devices()
        if len(devices) > 1:
            __task_queue = TorchProcessTaskQueue(worker_fun=__worker_fun)
        else:
            __task_queue = TorchThreadTaskQueue(worker_fun=__worker_fun)
        __task_queue.start()
    input_chunks = list(
        split_list_to_chunks(
            inputs,
            (len(inputs) + __task_queue.worker_num - 1) // __task_queue.worker_num,
        )
    )

    target_chunks = list(
        split_list_to_chunks(
            targets,
            (len(targets) + __task_queue.worker_num - 1) // __task_queue.worker_num,
        )
    )
    for idx, (input_chunk, target_chunk) in enumerate(zip(input_chunks, target_chunks)):
        __task_queue.add_task(
            (
                idx,
                input_chunk,
                target_chunk,
                model_with_loss,
            )
        )

    gradient_dict = {}
    for _ in range(len(input_chunks)):
        idx, gradient_list = __task_queue.get_result()
        gradient_dict[idx] = gradient_list

    gradient_lists = []
    for idx in sorted(gradient_dict.keys()):
        gradient_lists += gradient_dict[idx]
    assert len(gradient_lists) == len(inputs)
    return gradient_lists
