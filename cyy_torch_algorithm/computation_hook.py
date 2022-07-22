import threading
from typing import Callable

import torch
from cyy_torch_toolbox.hook import Hook


class ComputationHook(Hook):
    __local_data = threading.local()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result_dict = None
        self._task_queue = None
        self._task_size: int = 0
        self._result_transform: Callable | None = None

    def set_result_transform(self, f):
        self._result_transform = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    @property
    def result_dict(self):
        if self._result_dict is None:
            self._result_dict = {}
            for _ in range(self._task_size):
                self._result_dict |= self._task_queue.get_result()
            self._task_size = 0
        return self._result_dict

    def _after_execute(self, **_):
        self._result_dict = None

    def __del__(self):
        if self._task_queue is not None:
            self._task_queue.release()
            self._task_queue = None

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
