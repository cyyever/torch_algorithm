import functools
from typing import Callable

import torch
from cyy_torch_algorithm.computation.computation_hook import ComputationHook
from cyy_torch_toolbox.tensor import tensor_to


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__data_fun: Callable | None = None

    def set_data_fun(self, data_fun):
        self.__data_fun = data_fun

    def _after_forward(self, model_executor, inputs, targets, batch_index, **kwargs):
        assert self.__data_fun is not None
        data = self.__data_fun()
        if data is None:
            return
        self.add_task(
            model_executor=model_executor,
            inputs=inputs,
            targets=targets,
            data=data,
            batch_index=batch_index,
        )

    def _get_batch_computation_fun(self):
        raise NotImplementedError()

    def _get_worker_fun(self):
        return functools.partial(
            BatchComputationHook.common_worker_fun,
            self._result_transform,
            self._get_batch_computation_fun(),
        )

    def add_task(self, model_executor, inputs, targets, data, batch_index):
        assert not self.has_unfetched_result()
        self._broadcast_one_shot_data(
            batch_index=batch_index,
            model_with_loss=model_executor.model_with_loss,
            inputs=inputs,
            targets=targets,
        )
        for data_idx, data_piece in enumerate(data):
            self._add_task(
                task=(batch_index, data_idx, data_piece),
            )

    @classmethod
    def common_worker_fun(
        cls, result_transform, worker_fun, tasks, device, worker_queue, **kwargs
    ):
        batch_size = len(tasks)
        worker_device, worker_stream = ComputationHook._setup_device(device)
        batch_index = tasks[0][0]
        data_indices = [task[1] for task in tasks]
        data = [task[2] for task in tasks]
        with torch.cuda.stream(worker_stream):
            one_shot_data = cls.get_cached_one_shot_data(
                batch_index=batch_index,
                worker_device=worker_device,
                worker_queue=worker_queue,
            )
            data = tensor_to(data, device=worker_device, non_blocking=True)
            worker_fun = ComputationHook.get_cached_item(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(data=data, worker_device=worker_device, **one_shot_data)
            assert result_transform is None
            return batch_size, dict(zip(data_indices, res))
