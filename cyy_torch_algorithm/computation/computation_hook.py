import os
import threading
from typing import Callable

import pynvml
import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.device import get_cuda_devices
from cyy_torch_toolbox.hook import Hook


class ComputationHook(Hook):
    _local_data = threading.local()

    def __init__(self, **kwargs):
        if "stripable" not in kwargs:
            kwargs["stripable"] = True
        super().__init__(**kwargs)
        self.__result_dict = {}
        self.__task_queue = None
        self._result_transform: Callable | None = None
        self.__prev_tasks = []
        self.__result_collection_fun: Callable | None = None
        self.__prevous_chunk = None

    def set_result_transform(self, f: Callable) -> None:
        self._result_transform = f

    def set_result_collection_fun(self, f: Callable) -> None:
        self.__result_collection_fun = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def reset_result(self) -> None:
        self._fetch_result()
        del self.__result_dict
        self.__result_dict = {}

    @property
    def result_dict(self) -> dict:
        results = self._fetch_result()
        self.__result_dict |= results
        return self.__result_dict

    def has_unfetched_result(self):
        return bool(self.__prev_tasks)

    def _fetch_result(self) -> dict:
        results: dict = {}
        for _ in self.__prev_tasks:
            if self.__result_collection_fun is not None:
                self.__result_collection_fun(self.__task_queue.get_result())
            else:
                results |= self.__task_queue.get_result()
        self.__prev_tasks = []
        return results

    def __get_chunk_size(self, data_size):
        match self.__prevous_chunk:
            case None:
                self.__prevous_chunk = (1, False)
                return self.__prevous_chunk[0]
            case [size, fixed]:
                if data_size <= size or fixed:
                    return size
                pynvml.nvmlInit()
                free_resource_ratios = []
                for device in get_cuda_devices():
                    h = pynvml.nvmlDeviceGetHandleByIndex(device.index)
                    info = pynvml.nvmlDeviceGetMemoryInfo(h)
                    free_resource_ratios.append(info.free / info.total)
                pynvml.nvmlShutdown()
                if all(r >= 0.2 for r in free_resource_ratios):
                    self.__prevous_chunk = (size + 1, False)
                else:
                    self.__prevous_chunk = (size, True)
                return self.__prevous_chunk[0]
            case _:
                raise NotImplementedError()

    def _split_data(self, data_list: list) -> list:
        chunk_size = self.__get_chunk_size(len(data_list[0]))
        return list(
            zip(*(tuple(split_list_to_chunks(data, chunk_size)) for data in data_list))
        )

    def __get_task_queue(
        self, worker_fun: None | Callable = None
    ) -> TorchProcessTaskQueue:
        if self.__task_queue is None:
            worker_num: int | None | str = os.getenv("cuda_device_num", None)
            if worker_num is not None:
                worker_num = int(worker_num)
            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=worker_fun,
                send_tensor_in_cpu=False,
                use_manager=False,
                worker_num=worker_num,
            )
            self.__task_queue.start()
            torch.cuda.empty_cache()
        return self.__task_queue

    def _add_task(self, task, worker_fun: None | Callable = None) -> None:
        self.__prev_tasks.append(task)
        self.__get_task_queue(worker_fun).add_task(task)

    def _before_execute(self, **_):
        self.reset_result()

    def _after_execute(self, **_):
        self.reset_result()

    def __del__(self):
        self.release_queue()

    def release_queue(self) -> None:
        assert not self.__prev_tasks
        self.reset_result()
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None

    @classmethod
    def _setup_cuda_device(cls, advised_device):
        worker_device = getattr(cls._local_data, "worker_device", None)
        if worker_device is None:
            worker_device = advised_device
            cls._local_data.worker_device = worker_device
        worker_stream = getattr(cls._local_data, "worker_stream", None)
        if worker_stream is None:
            worker_stream = torch.cuda.Stream(device=worker_device)
            cls._local_data.worker_stream = worker_stream
        return worker_device, worker_stream
