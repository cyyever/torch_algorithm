import threading
from typing import Callable

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook


class ComputationHook(Hook):
    __local_data = threading.local()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__result_dict = None
        self.__task_queue = None
        self.__task_size: int = 0
        self._result_transform: Callable | None = None

    def set_result_transform(self, f):
        self._result_transform = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def _reset_result(self) -> None:
        self.__result_dict = None

    @property
    def result_dict(self):
        if self.__result_dict is None:
            self.__result_dict = {}
            for _ in range(self.__task_size):
                self.__result_dict |= self.__task_queue.get_result()
            self.__task_size = 0
        return self.__result_dict

    def _split_data(self, data_list: list) -> list:
        chunk_size = 24
        if self.__task_queue is not None:
            avg_chunk_size = (
                len(data_list) + self.__task_queue.worker_num - 1
            ) // self.__task_queue.worker_num
            chunk_size = max(avg_chunk_size, chunk_size)
        get_logger().debug("chunk_size is %s", chunk_size)
        return zip(
            *(tuple(split_list_to_chunks(data, chunk_size)) for data in data_list)
        )

    def __get_task_queue(self, model_executor, worker_fun) -> TorchProcessTaskQueue:
        if self.__task_queue is None:
            max_needed_cuda_bytes = None
            stats = torch.cuda.memory_stats(device=model_executor.device)
            if stats:
                max_needed_cuda_bytes = stats["allocated_bytes.all.peak"]

            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=worker_fun,
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self.__task_queue.start()
        return self.__task_queue

    def _add_task(self, model_executor, worker_fun, task) -> None:
        self.__task_size += 1
        self.__get_task_queue(model_executor, worker_fun).add_task(task)

    def _after_execute(self, **_):
        self.__result_dict = None

    def __del__(self):
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None

    @classmethod
    def _setup_cuda_device(cls, advised_device):
        worker_device = getattr(cls.__local_data, "worker_device", None)
        if worker_device is None:
            worker_device = advised_device
            cls.__local_data.worker_device = worker_device
        worker_stream = getattr(cls.__local_data, "worker_stream", None)
        if worker_stream is None:
            worker_stream = torch.cuda.Stream(device=worker_device)
            cls.__local_data.worker_stream = worker_stream
        return worker_device, worker_stream
