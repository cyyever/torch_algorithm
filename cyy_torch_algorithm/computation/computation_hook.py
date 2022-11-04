import copy
import os
import threading
from typing import Any, Callable

import torch
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
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
        self.__pending_task_cnt: int = 0
        self.__prev_tasks = []
        self.__result_collection_fun: Callable | None = None

    def set_result_transform(self, f: Callable) -> None:
        self._result_transform = f

    def set_result_collection_fun(self, f: Callable) -> None:
        self.__result_collection_fun = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def reset_result(self) -> None:
        self.__fetch_result()
        del self.__result_dict
        self.__result_dict = {}

    @property
    def result_dict(self) -> dict:
        results = self.__fetch_result()
        self.__result_dict |= results
        return self.__result_dict

    def has_unfetched_result(self):
        return self.__pending_task_cnt != 0

    def _drop_result(self) -> None:
        self.__fetch_result(drop=True)

    def __fetch_result(self, drop: bool = False) -> dict:
        results: dict = {}
        while self.has_unfetched_result():
            res = self.__task_queue.get_data()
            self.__pending_task_cnt -= res[0]
            assert self.__pending_task_cnt >= 0
            if not drop:
                if self.__result_collection_fun is not None:
                    self.__result_collection_fun(res[1])
                else:
                    results |= res[1]
        self.__prev_tasks = []
        return results

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
                batch_process=True,
            )
            self.__task_queue.start()
            torch.cuda.empty_cache()
        return self.__task_queue

    def _add_task(self, task: Any) -> None:
        self.__prev_tasks.append(task)
        self.__pending_task_cnt += 1
        self._get_task_queue().add_task(task)

    def _broadcast_one_shot_data(
        self, batch_index: int, model_with_loss, **kwargs
    ) -> None:
        if model_with_loss.model.training:
            model_with_loss = copy.deepcopy(model_with_loss)
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

    def __del__(self):
        self.release_queue()

    def release_queue(self) -> None:
        assert not self.has_unfetched_result()
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
    def get_cached_function(cls, name: str, fun: Callable, worker_device) -> Callable:
        if not hasattr(cls._local_data, name):
            fun = tensor_to(
                fun,
                device=worker_device,
                non_blocking=True,
            )
            setattr(cls._local_data, name, fun)
            return fun
        else:
            return getattr(cls._local_data, name)

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
