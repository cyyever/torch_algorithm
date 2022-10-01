import os
import threading
from typing import Any, Callable

import pynvml
import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.device import get_cuda_devices
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.tensor import tensor_to


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

    def _get_task_queue(self) -> TorchProcessTaskQueue:
        if self.__task_queue is None:
            worker_num: int | None | str = os.getenv("cuda_device_num", None)
            if worker_num is not None:
                worker_num = int(worker_num)
            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=self._get_worker_fun(),
                send_tensor_in_cpu=False,
                use_manager=False,
                worker_num=worker_num,
                use_worker_queue=True,
            )
            self.__task_queue.start()
            torch.cuda.empty_cache()
        return self.__task_queue

    def _add_task(self, task: Any) -> None:
        self.__prev_tasks.append(task)
        self._get_task_queue().add_task(task)

    def _broadcast_one_shot_data(
        self, batch_index: int, model_executor, **kwargs
    ) -> None:
        model_with_loss = model_executor.model_with_loss
        if model_with_loss.model.training:
            model_with_loss = model_executor.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.requires_grad_(requires_grad=False)
        model_with_loss.model.share_memory()
        task_queue = self._get_task_queue()
        for worker_id in range(task_queue.worker_num):
            worker_queue = task_queue.get_worker_queue(worker_id=worker_id)
            worker_queue.put(
                (batch_index, kwargs | {"model_with_loss": model_with_loss})
            )

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
        torch.cuda.set_device(worker_device)
        return worker_device, worker_stream

    @classmethod
    def get_cached_one_shot_data(cls, batch_index, worker_device, worker_queue) -> dict:
        if (
            not hasattr(ComputationHook._local_data, "batch_index")
            or ComputationHook._local_data.batch_index != batch_index
        ):
            while True:
                res = worker_queue.get()
                if res[0] == batch_index:
                    data = res[1]
                    assert isinstance(data, dict)
                    break
            if "model_with_loss" in data:
                data["model_with_loss"].to(device=worker_device, non_blocking=True)
                data["parameter_list"] = data[
                    "model_with_loss"
                ].model_util.get_parameter_list(detach=True)
                data["parameter_shapes"] = data[
                    "model_with_loss"
                ].model_util.get_parameter_shapes()
            if "inputs" in data:
                data["inputs"] = tensor_to(
                    data["inputs"], device=worker_device, non_blocking=True
                )
            if "targets" in data:
                data["targets"] = tensor_to(
                    data["targets"], device=worker_device, non_blocking=True
                )
            setattr(ComputationHook._local_data, "batch_index", batch_index)
            setattr(ComputationHook._local_data, "data", data)
        else:
            data = getattr(ComputationHook._local_data, "data")
        return data
