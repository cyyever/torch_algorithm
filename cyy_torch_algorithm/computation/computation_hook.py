import copy
import os
import threading
from typing import Any, Callable

import torch
# from cyy_naive_lib.profiling import Profile
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.tensor import tensor_to


class ComputationHook(Hook):
    __local_data = threading.local()

    def __init__(self, **kwargs):
        super().__init__(stripable=True, **kwargs)
        self.__result_dict = {}
        self.__task_queue: TorchProcessTaskQueue | None = None
        self._result_transform: Callable | None = None
        self.__pending_task_cnt: int = 0
        self.__prev_tasks = []
        self.__result_collection_fun: Callable | None = None
        self.__last_model_id = None
        self.__shared_model = None

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
        with TimeCounter() as cnt:
            task_queue = self._get_task_queue()
            new_kwargs = kwargs
            print("prepare use ", cnt.elapsed_milliseconds())
            if self.__last_model_id is None or self.__last_model_id != id(
                model_with_loss
            ):
                self.__shared_model = copy.deepcopy(model_with_loss)
                self.__shared_model.model.zero_grad(set_to_none=True)
                self.__shared_model.model.share_memory()
                self.__last_model_id = id(model_with_loss)
                new_kwargs |= {"model_with_loss": self.__shared_model}
            else:
                shared_state_dict = self.__shared_model.model.state_dict()
                for k, v in model_with_loss.model.state_dict().items():
                    shared_state_dict[k].copy_(v)

            if not new_kwargs:
                return
            for worker_id in range(task_queue.worker_num):
                worker_queue = task_queue.get_worker_queue(worker_id=worker_id)
                worker_queue.put((batch_index, new_kwargs))
            print("_broadcast_one_shot_data use ", cnt.elapsed_milliseconds())

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
    def _setup_device(cls, advised_device) -> tuple:
        worker_device = getattr(cls.__local_data, "worker_device", None)
        if worker_device is None:
            worker_device = advised_device
            cls.__local_data.worker_device = worker_device
        if not torch.cuda.is_available():
            return worker_device, None
        worker_stream = getattr(cls.__local_data, "worker_stream", None)
        if worker_stream is None:
            worker_stream = torch.cuda.Stream(device=worker_device)
            cls.__local_data.worker_stream = worker_stream
        torch.cuda.set_device(worker_device)
        return worker_device, worker_stream

    @classmethod
    def get_cached_item(cls, name: str, value: Any, worker_device) -> Any:
        if not hasattr(cls.__local_data, name):
            value = tensor_to(
                value,
                device=worker_device,
                non_blocking=True,
            )
            setattr(cls.__local_data, name, value)
            return value
        return getattr(cls.__local_data, name)

    def _after_optimizer_step(self, step_skipped: bool, **kwargs) -> None:
        if step_skipped:
            self._drop_result()

    @classmethod
    def get_cached_one_shot_data(cls, batch_index, worker_device, worker_queue) -> dict:
        cnt = TimeCounter()
        data = getattr(ComputationHook.__local_data, "data", {})
        if (
            not hasattr(ComputationHook.__local_data, "batch_index")
            or ComputationHook.__local_data.batch_index != batch_index
        ):
            new_data = {}
            model_with_loss = None
            while not worker_queue.empty():
                res = worker_queue.get()
                new_data = res[1]
                if "model_with_loss" in new_data:
                    model_with_loss = new_data.pop("model_with_loss")
                if res[0] < batch_index:
                    continue
                break
            setattr(ComputationHook.__local_data, "batch_index", batch_index)
            if model_with_loss is not None:
                setattr(cls.__local_data, "shared_model_with_loss", model_with_loss)
            assert next(
                iter(cls.__local_data.shared_model_with_loss.model.parameters())
            ).is_shared()
            data["model_with_loss"] = copy.deepcopy(
                cls.__local_data.shared_model_with_loss
            )
            data["model_with_loss"].model.requires_grad_(requires_grad=False)
            data["model_with_loss"].to(device=worker_device)
            print("copy model_with_loss", cnt.elapsed_milliseconds())
            # with Profile() as c:
            if "parameter_shapes" not in data:
                data[
                    "parameter_shapes"
                ] = model_with_loss.model_util.get_parameter_shapes()

            if "parameter_list" not in data:
                data["parameter_list"] = data[
                    "model_with_loss"
                ].model_util.get_parameter_list(detach=False)
            else:
                a = data["parameter_list"]
                bias = 0
                for parameter in data["model_with_loss"].model_util.get_parameter_seq(
                    detach=False
                ):
                    param_element_num = parameter.numel()
                    a[bias: bias + param_element_num] = parameter.view(-1)
                    bias += param_element_num

            print("parameter list", cnt.elapsed_milliseconds())
            if new_data:
                data |= tensor_to(new_data, device=worker_device, non_blocking=True)

            if data:
                setattr(ComputationHook.__local_data, "data", data)
        return data
