#!/usr/bin/env python3

import atexit
import collections

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.model_with_loss import ModelWithLoss
from torch import autograd


def __get_f(device, inputs, targets, model_with_loss: ModelWithLoss):
    model_util = ModelUtil(model_with_loss.model)

    def f(*args):
        nonlocal inputs, targets, model_with_loss, device, model_util
        total_loss = None
        for parameter_list in args:
            model_util.load_parameter_list(parameter_list, as_parameter=False)
            loss = model_with_loss(inputs, targets, device=device)["loss"]
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss

    return f


def worker_fun(task, args):
    worker_device = args["device"]
    torch.cuda.set_device(worker_device)
    (idx, vector_chunk, model_with_loss, parameter_list, inputs, targets) = task
    for index, vector in enumerate(vector_chunk):
        vector_chunk[index] = vector.to(worker_device)
    inputs = inputs.to(worker_device)
    targets = targets.to(worker_device)
    parameter_list = parameter_list.to(worker_device)
    vector_chunk = tuple(vector_chunk)
    products = autograd.functional.vhp(
        __get_f(
            worker_device,
            inputs,
            targets,
            model_with_loss,
        ),
        tuple([parameter_list] * len(vector_chunk)),
        vector_chunk,
        strict=True,
    )[1]
    return (idx, products)


__task_queue = None


def stop_task_queue():
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_hessian_vector_product_func(
    model_with_loss: ModelWithLoss, batch, main_device=None
):
    devices = get_devices()

    model = model_with_loss.model
    model_util = ModelUtil(model)
    if model_util.is_pruned:
        model_util.merge_and_remove_masks()
    model.zero_grad()
    model.share_memory()

    parameter_list = model_util.get_parameter_list(detach=True)

    def vhp_func(v):
        global __task_queue
        nonlocal main_device
        v_is_tensor = False
        if isinstance(v, collections.abc.Sequence):
            vectors = v
        else:
            v_is_tensor = True
            vectors = [v]

        vector_chunks = tuple(
            split_list_to_chunks(
                vectors, (len(vectors) + len(devices) - 1) // len(devices)
            )
        )
        assert len(vector_chunks) <= len(devices)

        if __task_queue is None:
            __task_queue = TorchProcessTaskQueue(worker_fun)
            __task_queue.start()
        for idx, vector_chunk in enumerate(vector_chunks):
            __task_queue.add_task(
                (idx, vector_chunk, model_with_loss, parameter_list, batch[0], batch[1])
            )

        total_products = {}
        for _ in range(len(vector_chunks)):
            idx, gradient_list = __task_queue.get_result()
            total_products[idx] = gradient_list

        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if main_device is None:
            main_device = devices[0]
        products = [p.to(main_device) for p in products]
        if v_is_tensor:
            return products[0]
        return products

    return vhp_func
