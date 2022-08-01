import functools
from typing import Callable

from cyy_torch_algorithm.computation_hook import ComputationHook


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__data_fun: Callable | None = None

    def set_data_fun(self, data_fun):
        self.__data_fun = data_fun

    def _after_forward(self, model_executor, inputs, targets, **kwargs):
        assert self.__data_fun is not None
        self._reset_result()
        data = self.__data_fun()
        if not data:
            return

        model_with_loss = model_executor.model_with_loss
        if model_with_loss.model.training:
            model_with_loss = model_executor.copy_model_with_loss(deepcopy=True)
            model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.share_memory()
        worker_fun = functools.partial(
            BatchComputationHook.common_worker_fun,
            self._result_transform,
            self._get_worker_fun(),
        )
        self._fetch_result()
        for data_piece in self._split_data([data]):
            task = (model_with_loss, inputs, targets, *data_piece)
            self._add_task(
                model_executor=model_executor, worker_fun=worker_fun, task=task
            )

    @classmethod
    def common_worker_fun(cls, result_transform, worker_fun, task, args):
        worker_device, worker_stream = ComputationHook._setup_cuda_device(
            args["device"]
        )
        model_with_loss, inputs, targets, data = task
        res = worker_fun(
            model_with_loss=model_with_loss,
            inputs=inputs,
            targets=targets,
            data=data,
            worker_device=worker_device,
            worker_stream=worker_stream,
        )
        assert result_transform is None
        return res
