import copy
import functools
import os
import threading
from typing import Any, Callable

import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model.evaluator import ModelEvaluator
from cyy_torch_toolbox.tensor import tensor_to


class ComputationHook(Hook):
    def __init__(self, **kwargs) -> None:
        super().__init__(stripable=True, **kwargs)
        self.__local_data = threading.local()
        self.__result_dict: dict = {}
        self.__task_queue: TorchProcessTaskQueue | None = None
        self.__model_queue: TorchProcessTaskQueue | None = None
        self._result_transform: Callable | None = None
        self.__pending_task_cnt: int = 0
        self.__prev_tasks: list = []
        self.__result_collection_fun: Callable | None = None
        self.__shared_models: dict = {}

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_ComputationHook__local_data"] = None
        return state

    def set_result_transform(self, f: Callable) -> None:
        self._result_transform = f
        self._remove_cached_item("result_transform")

    def set_result_collection_fun(self, f: Callable) -> None:
        self.__result_collection_fun = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def _model_worker_fun(self, task, *args, **kwargs) -> Any:
        batch_index, need_model_evaluator = task
        res = self.__shared_models[batch_index]
        if need_model_evaluator:
            res = res | {"model_evaluator": self.__shared_models[0]["model_evaluator"]}
        return res

    def reset_result(self) -> None:
        self._drop_result()
        del self.__result_dict
        self.__result_dict = {}

    @property
    def result_dict(self) -> dict:
        return self.__fetch_result()

    def has_unfetched_result(self):
        return self.__pending_task_cnt != 0

    def _drop_result(self) -> None:
        self.__fetch_result(drop=True)

    def __fetch_result(self, drop: bool = False) -> dict:
        results: dict = {}
        assert self.__pending_task_cnt >= 0
        while self.has_unfetched_result():
            assert self.__task_queue is not None
            res = self.__task_queue.get_data()
            assert res is not None
            res = res[0]
            self.__pending_task_cnt -= res[0]
            assert self.__pending_task_cnt >= 0
            if not drop:
                if self.__result_collection_fun is not None:
                    self.__result_collection_fun(res[1])
                else:
                    results |= res[1]
            else:
                del res
        self.__prev_tasks = []
        self.__result_dict |= results
        return self.__result_dict

    def _get_task_queue(self) -> TorchProcessTaskQueue:
        if self.__task_queue is None:
            worker_num: int | None | str = os.getenv("cuda_device_num", None)
            if worker_num is not None:
                worker_num = int(worker_num)
            self.__task_queue = TorchProcessTaskQueue(
                worker_num=worker_num,
                batch_process=True,
            )
            self.__task_queue.start(
                worker_fun=functools.partial(
                    self._get_worker_fun(),
                    model_queue=self.__get_model_queue(),
                )
            )
        return self.__task_queue

    def __get_model_queue(self) -> TorchProcessTaskQueue:
        if self.__model_queue is None:
            self.__model_queue = TorchProcessTaskQueue(
                worker_num=1,
            )
            self.__model_queue.start(worker_fun=self._model_worker_fun, use_thread=True)
        return self.__model_queue

    def _add_task(self, task: Any) -> None:
        self.__prev_tasks.append(task)
        self.__pending_task_cnt += 1
        self._get_task_queue().add_task(task)

    def _broadcast_one_shot_data(
        self, batch_index: int, model_evaluator: ModelEvaluator, **kwargs
    ) -> None:
        with TimeCounter() as cnt:
            if self.__shared_models:
                old_model_evaluator = self.__shared_models[0]
                self.__shared_models.clear()
                self.__shared_models[0] = old_model_evaluator
            assert batch_index >= 0
            data: dict = dict(kwargs)
            if batch_index == 0:
                data["model_evaluator"] = copy.deepcopy(model_evaluator)
                data["model_evaluator"].model.cpu()
                data["model_evaluator"].model.zero_grad(set_to_none=True)
                data["model_evaluator"].model.requires_grad_(False)
                data["model_evaluator"].model.share_memory()
            else:
                data["parameter_dict"] = model_evaluator.model_util.get_parameter_dict(
                    detach=True
                )
                for v in data["parameter_dict"].values():
                    v.grad = None
                    v.requires_grad_(False)
                    v.share_memory_()
            self.__shared_models[batch_index] = data
            get_logger().debug(
                "_broadcast_one_shot_data use %s", cnt.elapsed_milliseconds()
            )

    def _before_execute(self, **_):
        self.reset()

    def __del__(self):
        self.reset()

    def release_queue(self):
        self.reset()

    def reset(self) -> None:
        assert not self.has_unfetched_result()
        self.reset_result()
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None
        if self.__model_queue is not None:
            self.__model_queue.release()
            self.__model_queue = None
        self.__shared_models.clear()

    def _setup_device(self, advised_device) -> tuple:
        worker_device = self.get_cached_item("worker_device", advised_device)
        if not torch.cuda.is_available():
            return worker_device, None
        worker_stream = getattr(self.__local_data, "worker_stream", None)
        if worker_stream is None:
            worker_stream = self.get_cached_item(
                "worker_stream", torch.cuda.Stream(device=worker_device)
            )
        torch.cuda.set_device(worker_device)
        return worker_device, worker_stream

    def _remove_cached_item(self, name: str) -> None:
        if self.__local_data is not None and hasattr(self.__local_data, name):
            delattr(self.__local_data, name)

    def get_cached_item(self, name: str, value: Any, worker_device=None) -> Any:
        if not hasattr(self.__local_data, name):
            if worker_device is not None:
                value = tensor_to(
                    value,
                    device=worker_device,
                    non_blocking=True,
                )
            if self.__local_data is None:
                self.__local_data = threading.local()
            setattr(self.__local_data, name, value)
            return value
        return getattr(self.__local_data, name)

    def get_cached_one_shot_data(
        self,
        batch_index: int,
        worker_device: torch.device,
        model_queue: TorchProcessTaskQueue,
    ) -> dict:
        data = getattr(self.__local_data, "data", {})
        if (
            hasattr(self.__local_data, "batch_index")
            and self.__local_data.batch_index == batch_index
        ):
            return data
        model_queue.add_task((batch_index, "model_evaluator" not in data))
        new_data: dict = model_queue.get_data()[0]

        self.__local_data.batch_index = batch_index
        if "model_evaluator" in new_data:
            new_data["model_evaluator"] = copy.deepcopy(new_data["model_evaluator"])
            new_data["model_evaluator"].model_util.to_device(
                device=worker_device, non_blocking=True
            )
        if "parameter_dict" in new_data:
            new_data["parameter_dict"] = tensor_to(
                new_data["parameter_dict"], device=worker_device, non_blocking=True
            )
        else:
            new_data["parameter_dict"] = new_data[
                "model_evaluator"
            ].model_util.get_parameter_dict(detach=False)
        new_data = tensor_to(new_data, device=worker_device, non_blocking=True)
        data.update(new_data)

        setattr(self.__local_data, "data", data)
        assert "model_evaluator" in data
        return data
