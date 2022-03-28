#!/usr/bin/env python3
import atexit
import collections
import threading

from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.data_structure.torch_thread_task_queue import \
    TorchThreadTaskQueue
from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_with_loss import ModelWithLoss
from cyy_torch_toolbox.tensor import (cat_tensors_to_vector,
                                      split_tensor_to_list)
from functorch import grad, jvp, vjp
from torch import autograd


def __get_f(device, inputs, targets, model_with_loss, model_util):
    def f(*args):
        nonlocal inputs, targets, device, model_util
        model_util.load_parameter_list(
            list(args),
            check_parameter=False,
            as_parameter=False,
        )
        return model_with_loss(
            inputs,
            targets,
            device=device,
            non_blocking=False,
            phase=MachineLearningPhase.Test,
        )["loss"]

    return f


local_data = threading.local()


def worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device

    (idx, vector_chunk, model_with_loss, inputs, targets) = task
    vector_chunk = tuple(vector_chunk)
    model_with_loss.model.to(worker_device)
    model_util = model_with_loss.model_util
    parameter_list = tuple(
        get_mapping_values_by_key_order(model_util.get_parameter_dict(detach=False))
    )
    products = []
    shape_list = [p.shape for p in parameter_list]
    for vector in vector_chunk:
        vector = vector.to(worker_device)
        products.append(
            cat_tensors_to_vector(
                autograd.functional.hvp(
                    __get_f(
                        worker_device,
                        inputs,
                        targets,
                        model_with_loss,
                        model_util,
                    ),
                    parameter_list,
                    tuple(split_tensor_to_list(shape_list, vector)),
                    strict=True,
                )[1]
            )
        )
    return (idx, products)


__task_queue = None


def stop_task_queue():
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_hessian_vector_product_func(
    model_with_loss: ModelWithLoss, batch, main_device=None
):

    model = model_with_loss.model
    model.zero_grad(set_to_none=True)
    model.share_memory()

    devices = get_devices()

    def vhp_func(v):
        global __task_queue
        nonlocal main_device
        nonlocal devices
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
            if len(devices) > 0:
                __task_queue = TorchProcessTaskQueue(worker_fun, move_data_in_cpu=True)
            else:
                __task_queue = TorchThreadTaskQueue(worker_fun)
            __task_queue.start()
        for idx, vector_chunk in enumerate(vector_chunks):
            __task_queue.add_task(
                (idx, vector_chunk, model_with_loss, batch[0], batch[1])
            )

        total_products = {}
        for _ in range(len(vector_chunks)):
            idx, gradient_list = __task_queue.get_result()
            total_products[idx] = gradient_list

        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if main_device is not None:
            products = [p.to(main_device) for p in products]
        if v_is_tensor:
            return products[0]
        return products

    return vhp_func
