import copy
import functools
import os
import threading
from collections.abc import Callable
from typing import Any

import torch
from cyy_naive_lib.log import log_debug
from cyy_naive_lib.time_counter import TimeCounter
from cyy_preprocessing_pipeline import tensor_to
from cyy_torch_toolbox import Hook, ModelEvaluator, TorchProcessTaskQueue


class ComputationHook(Hook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(stripable=True, **kwargs)
        self.__local_data: threading.local | None = threading.local()
        self.__result_dict: dict[Any, Any] = {}
        self.__task_queue: TorchProcessTaskQueue | None = None
        self.__model_queue: TorchProcessTaskQueue | None = None
        self._result_transform: Callable[..., Any] | None = None
        self.__pending_task_cnt: int = 0
        self.__prev_tasks: list[Any] = []
        self.__result_collection_fun: Callable[..., None] | None = None
        self.__shared_models: dict[int, dict[str, Any]] = {}

    def __getstate__(self) -> dict[str, Any]:
        # capture what is normally pickled
        state = self.__dict__.copy()
        state["_ComputationHook__local_data"] = None
        return state

    def set_result_transform(self, f: Callable[..., Any]) -> None:
        self._result_transform = f
        self._remove_cached_item("result_transform")

    def set_result_collection_fun(self, f: Callable[..., None]) -> None:
        self.__result_collection_fun = f

    def _get_worker_fun(self) -> Callable[..., Any]:
        raise NotImplementedError()

    def _model_worker_fun(self, task, **kwargs: Any) -> Any:
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
    def result_dict(self) -> dict[Any, Any]:
        return self.__fetch_result()

    def has_unfetched_result(self) -> bool:
        return self.__pending_task_cnt != 0

    def _drop_result(self) -> None:
        self.__fetch_result(drop=True)

    def __fetch_result(self, drop: bool = False) -> dict[Any, Any]:
        results: dict[Any, Any] = {}
        assert self.__pending_task_cnt >= 0
        while self.has_unfetched_result():
            assert self.__task_queue is not None
            res = self.__task_queue.get_data().value()
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
            worker_num: int | None | str = os.getenv("CUDA_DEVICE_NUM", None)
            if worker_num is not None:
                worker_num = int(worker_num)
            self.__task_queue = TorchProcessTaskQueue(
                worker_num=worker_num,
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
        self, batch_index: int, model_evaluator: ModelEvaluator, **kwargs: Any
    ) -> None:
        with TimeCounter() as cnt:
            if self.__shared_models:
                old_model_evaluator = self.__shared_models[0]
                self.__shared_models.clear()
                self.__shared_models[0] = old_model_evaluator
            assert batch_index >= 0
            data: dict[str, Any] = dict(kwargs)
            if batch_index == 0:
                data["model_evaluator"] = copy.deepcopy(model_evaluator)
                data["model_evaluator"].model.cpu()
                data["model_evaluator"].model.zero_grad(set_to_none=True)
                data["model_evaluator"].model.requires_grad_(False)
                data["model_evaluator"].model.share_memory()
            else:
                data["parameters"] = model_evaluator.model_util.get_parameters(
                    detach=True
                )
                for v in data["parameters"].values():
                    v.grad = None
                    v.requires_grad_(False)
                    v.share_memory_()
            self.__shared_models[batch_index] = data
            log_debug("_broadcast_one_shot_data use %s", cnt.elapsed_milliseconds())

    def _before_execute(self, **_) -> None:
        self.reset()

    def release(self) -> None:
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

    def _setup_device(
        self, advised_device: torch.device
    ) -> tuple[torch.device, torch.cuda.Stream | None]:
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

    def get_cached_item(
        self, name: str, value: Any, worker_device: torch.device | None = None
    ) -> Any:
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
    ) -> dict[str, Any]:
        data: dict[str, Any] = getattr(self.__local_data, "data", {})
        if (
            hasattr(self.__local_data, "batch_index")
            and self.__local_data.batch_index == batch_index
        ):
            return data
        model_queue.add_task((batch_index, "model_evaluator" not in data))
        new_data: dict[str, Any] = model_queue.get_data().value()

        self.__local_data.batch_index = batch_index
        if "model_evaluator" in new_data:
            new_data["model_evaluator"] = copy.deepcopy(new_data["model_evaluator"])
            new_data["model_evaluator"].model_util.to_device(
                device=worker_device, non_blocking=True
            )
        if "parameters" in new_data:
            new_data["parameters"] = tensor_to(
                new_data["parameters"], device=worker_device, non_blocking=True
            )
        else:
            new_data["parameters"] = new_data[
                "model_evaluator"
            ].model_util.get_parameters(detach=False)
        new_data = tensor_to(new_data, device=worker_device, non_blocking=True)
        data.update(new_data)

        self.__local_data.data = data
        assert "model_evaluator" in data
        return data
