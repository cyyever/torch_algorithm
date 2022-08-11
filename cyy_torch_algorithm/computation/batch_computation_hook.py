import functools
from typing import Callable

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.computation.computation_hook import ComputationHook
from cyy_torch_toolbox.device import put_data_to_device


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

    def _after_forward(self, model_executor, inputs, targets, **kwargs):
        assert self.__data_fun is not None
        data = self.__data_fun()
        if data is None:
            return
        self.add_task(
            model_executor=model_executor, inputs=inputs, targets=targets, data=data
        )

    def add_task(self, model_executor, inputs, targets, data):
        assert not self.has_unfetched_result()
        model_with_loss = model_executor.model_with_loss
        if model_with_loss.model.training:
            model_with_loss = model_executor.copy_model_with_loss(deepcopy=True)
            model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.requires_grad_(requires_grad=False)
        model_with_loss.model.share_memory()
        worker_fun = functools.partial(
            BatchComputationHook.common_worker_fun,
            self._result_transform,
            self._get_worker_fun(),
        )
        for data_idx, data_piece in enumerate(self._split_data([data])):
            self._add_task(
                model_executor=model_executor,
                worker_fun=worker_fun,
                task=(model_with_loss, inputs, targets, data_idx, *data_piece),
            )

    @classmethod
    def common_worker_fun(cls, result_transform, worker_fun, task, args):
        worker_device, worker_stream = ComputationHook._setup_cuda_device(
            args["device"]
        )
        model_with_loss, inputs, targets, data_idx, data = task
        with torch.cuda.stream(worker_stream):
            model_with_loss.to(device=worker_device, non_blocking=True)
            inputs = put_data_to_device(inputs, device=worker_device, non_blocking=True)
            targets = put_data_to_device(
                targets, device=worker_device, non_blocking=True
            )
            data = put_data_to_device(data, device=worker_device, non_blocking=True)
            res = worker_fun(
                model_with_loss=model_with_loss,
                inputs=inputs,
                targets=targets,
                data_idx=data_idx,
                data=data,
                worker_device=worker_device,
            )
            assert result_transform is None
            return res
