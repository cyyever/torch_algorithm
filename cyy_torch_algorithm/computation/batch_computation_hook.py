import functools
from collections.abc import Callable
from typing import Any

import torch
from cyy_torch_toolbox import tensor_to

from .computation_hook import ComputationHook


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__data_fun: Callable | None = None
        self.__data: Any | None = None

    def set_data(self, data: Any) -> None:
        self.__data = data

    @property
    def data(self) -> Any:
        if self.__data is None:
            assert self.__data_fun is not None
            return self.__data_fun()
        return self.__data

    def set_data_fun(self, data_fun: Callable) -> None:
        self.__data_fun = data_fun

    def _before_batch(
        self, executor, inputs, targets, batch_index: int, **kwargs: Any
    ) -> None:
        data = self.data
        if data is None:
            assert self.__data_fun is not None
            data = self.__data_fun()
        assert data is not None
        self.add_task(
            executor=executor,
            inputs=inputs,
            targets=targets,
            data=data,
            batch_index=batch_index,
        )

    def _get_batch_computation_fun(self) -> Callable:
        raise NotImplementedError()

    def _get_worker_fun(self) -> Callable:
        return functools.partial(
            self.common_worker_fun,
            self._get_batch_computation_fun(),
        )

    def add_task(self, executor, inputs, targets, data, batch_index: int) -> None:
        assert not self.has_unfetched_result()
        self._broadcast_one_shot_data(
            batch_index=batch_index,
            model_evaluator=executor.model_evaluator,
            inputs=inputs,
            targets=targets,
        )
        for data_idx, data_piece in enumerate(data):
            self._add_task(
                task=(batch_index, data_idx, data_piece),
            )

    def common_worker_fun(
        self,
        worker_fun: Callable,
        tasks: list,
        device: torch.device,
        model_queue,
        **kwargs: Any,
    ) -> tuple:
        batch_size = len(tasks)
        worker_device, worker_stream = self._setup_device(device)
        batch_index = tasks[0][0]
        data_indices = [task[1] for task in tasks]
        data = [task[2] for task in tasks]
        with torch.cuda.stream(worker_stream):
            one_shot_data = self.get_cached_one_shot_data(
                batch_index=batch_index,
                worker_device=worker_device,
                model_queue=model_queue,
            )
            data = tensor_to(data, device=worker_device, non_blocking=True)
            worker_fun = self.get_cached_item(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(data=data, worker_device=worker_device, **one_shot_data)

            result_transform = self.get_cached_item(
                "result_transform", self._result_transform, worker_device=worker_device
            )
            if result_transform is not None:
                new_res: dict = {
                    data_index: result_transform(
                        data_index=data_index, result=v, data=data
                    )
                    for data_index, v, data in zip(
                        data_indices, res, data, strict=False
                    )
                }
            else:
                new_res = dict(zip(data_indices, res, strict=False))
            return batch_size, new_res
