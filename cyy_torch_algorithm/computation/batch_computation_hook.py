import functools
from typing import Callable

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.computation_hook import ComputationHook
from cyy_torch_toolbox.tensor import tensor_to


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__data_fun: Callable | None = None

    def set_data_fun(self, data_fun):
        self.__data_fun = data_fun

    def _fetch_result(self) -> dict:
        res = super()._fetch_result()
        if not res:
            return {}
        result_list = list(get_mapping_values_by_key_order(res))
        return {
            0: torch.concat(
                [a.to(device="cuda:0", non_blocking=True) for a in result_list]
            )
        }

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
        for data_idx, data_piece in enumerate(self._split_data([data])):
            self._add_task(
                task=(batch_index, inputs, targets, data_idx, *data_piece),
            )
        self._broadcast_model(model_executor=model_executor, batch_index=batch_index)

    @classmethod
    def common_worker_fun(
        cls, result_transform, worker_fun, task, device, worker_queue, **kwargs
    ):
        worker_device, worker_stream = ComputationHook._setup_cuda_device(device)
        batch_index, inputs, targets, data_idx, data = task
        with torch.cuda.stream(worker_stream):
            model_with_loss, parameter_list, parameter_shapes = cls.get_cached_model(
                batch_index=batch_index,
                worker_device=worker_device,
                worker_queue=worker_queue,
            )
            inputs = tensor_to(inputs, device=worker_device, non_blocking=True)
            targets = tensor_to(targets, device=worker_device, non_blocking=True)
            data = tensor_to(data, device=worker_device, non_blocking=True)
            res = worker_fun(
                model_with_loss=model_with_loss,
                inputs=inputs,
                targets=targets,
                data_idx=data_idx,
                data=data,
                worker_device=worker_device,
                parameter_list=parameter_list,
                parameter_shapes=parameter_shapes,
            )
            assert result_transform is None
            return res
