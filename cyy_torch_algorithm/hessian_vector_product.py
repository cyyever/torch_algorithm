#!/usr/bin/env python3
import atexit
import collections
import threading
from typing import Callable

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.data_structure.torch_thread_task_queue import \
    TorchThreadTaskQueue
from cyy_torch_toolbox.device import get_devices
from cyy_torch_toolbox.model_executor import ModelExecutor
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

from cyy_torch_algorithm.batch_hvp.batch_hvp_hook import hvp

local_data = threading.local()


def worker_fun(task, args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device

    idx, vectors, model_with_loss, inputs, targets = task
    vectors = tuple(vectors)
    products = hvp(
        model_with_loss=model_with_loss,
        inputs=inputs,
        targets=targets,
        vectors=vectors,
        worker_device=worker_device,
    )
    return (idx, products)


__task_queue = None


def stop_task_queue():
    if __task_queue is not None:
        __task_queue.force_stop()


atexit.register(stop_task_queue)


def get_hessian_vector_product_func(
    model_with_loss: ModelWithLoss, batch: tuple
) -> Callable:

    model = model_with_loss.model
    model.zero_grad(set_to_none=True)
    model.share_memory()

    devices = get_devices()

    def vhp_func(v):
        global __task_queue
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
        _, inputs, targets, *__ = ModelExecutor.decode_batch(batch)
        for idx, vector_chunk in enumerate(vector_chunks):
            __task_queue.add_task((idx, vector_chunk, model_with_loss, inputs, targets))

        total_products = {}
        for _ in range(len(vector_chunks)):
            idx, gradient_list = __task_queue.get_result()
            total_products[idx] = gradient_list

        products = []
        for idx in sorted(total_products.keys()):
            products += total_products[idx]
        assert len(products) == len(vectors)
        if v_is_tensor:
            return products[0]
        return products

    return vhp_func
