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
        self.__result_dict = {}
        self.__task_queue = None
        self._result_transform: Callable | None = None
        self.__prev_tasks = []

    def set_result_transform(self, f):
        self._result_transform = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def reset_result(self) -> None:
        # get_logger().error("call reset result")
        self.__result_dict = {}

    @property
    def result_dict(self) -> dict:
        self._fetch_result()
        return self.__result_dict

    def _fetch_result(self):
        for _ in self.__prev_tasks:
            self.__result_dict |= self.__task_queue.get_result()
        self.__prev_tasks = []

    def _split_data(self, data_list: list) -> list:
        chunk_size = 24
        if self.__task_queue is not None:
            avg_chunk_size = (
                len(data_list[0]) + self.__task_queue.worker_num - 1
            ) // self.__task_queue.worker_num
            chunk_size = min(max(avg_chunk_size, chunk_size), 50)
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
                move_data_in_cpu=False,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
                use_manager=False,
            )
            self.__task_queue.start()
        return self.__task_queue

    def _add_task(self, model_executor, worker_fun, task) -> None:
        self.__prev_tasks.append(task)
        self.__get_task_queue(model_executor, worker_fun).add_task(task)

    def _before_execute(self, **_):
        self.reset_result()

    def release_queue(self, keep_result=True):
        if keep_result:
            self._fetch_result()
            for k, v in self.__result_dict.items():
                if isinstance(v, torch.Tensor):
                    self.__result_dict[k] = v.clone
        else:
            self.reset_result()
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
