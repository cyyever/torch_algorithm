import functools
from collections.abc import Callable
from typing import Any

import torch
from cyy_preprocessing_pipeline import tensor_to

from .computation_hook import ComputationHook


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.__data_fun: Callable[[], Any] | None = None
        self.__data: Any = None

    def set_data(self, data: Any) -> None:
        self.__data = data

    @property
    def data(self) -> Any:
        if self.__data is None:
            assert self.__data_fun is not None
            return self.__data_fun()
        return self.__data

    def set_data_fun(self, data_fun: Callable[[], Any]) -> None:
        self.__data_fun = data_fun

    def _before_batch(
        self, executor: Any, inputs: Any, targets: Any, batch_index: int, **kwargs: Any
    ) -> None:
        data = self.data
        assert data is not None
        self.add_task(
            executor=executor,
            inputs=inputs,
            targets=targets,
            data=data,
            batch_index=batch_index,
        )

    def _get_batch_computation_fun(self) -> Callable[..., Any]:
        raise NotImplementedError()

    def _get_worker_fun(self) -> Callable[..., Any]:
        return functools.partial(
            self.common_worker_fun,
            self._get_batch_computation_fun(),
        )

    def add_task(
        self, executor: Any, inputs: Any, targets: Any, data: Any, batch_index: int
    ) -> None:
        assert not self.has_unfetched_result()
        self._broadcast_one_shot_data(
            batch_index=batch_index,
            model_evaluator=executor.model_evaluator,
            inputs=inputs,
            targets=targets,
        )
        data_indices = list(range(len(data)))
        self._add_task(
            task=(batch_index, data_indices, data),
        )

    def common_worker_fun(
        self,
        worker_fun: Callable[..., Any],
        tasks: list[tuple[int, list[int], Any]],
        device: torch.device,
        model_queue: Any,
        **kwargs: Any,
    ) -> list[tuple[int, dict[int, Any]]]:
        task = tasks[0]
        batch_index = task[0]
        data_indices = task[1]
        data = task[2]
        worker_device, worker_stream = self._setup_device(device)
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
                new_res: dict[int, Any] = {
                    data_index: result_transform(
                        data_index=data_index, result=v, data=data_piece
                    )
                    for data_index, v, data_piece in zip(
                        data_indices, res, data, strict=True
                    )
                }
            else:
                new_res = dict(zip(data_indices, res, strict=True))
            return [(1, new_res)]
