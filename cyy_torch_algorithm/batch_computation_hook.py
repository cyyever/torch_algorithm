import functools
from typing import Any, Callable

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue

from cyy_torch_algorithm.computation_hook import ComputationHook


class BatchComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__data_fun: Callable | None = None

    def set_data_fun(self, data_fun):
        self.__data_fun = data_fun

    def _after_forward(self, model_executor, inputs, targets, **kwargs):
        assert self.__data_fun is not None
        self._result_dict = None
        self._task_size = None
        data = self.__data_fun()
        if not data:
            return

        self.__schedule_computation(
            trainer=model_executor, inputs=inputs, targets=targets, data=data
        )

    def __schedule_computation(
        self, trainer: Any, inputs: list, targets: list, data: list
    ) -> None:
        model_with_loss = trainer.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.cpu()
        if self._task_queue is None:
            max_needed_cuda_bytes = None
            stats = torch.cuda.memory_stats(device=trainer.device)
            if stats:
                max_needed_cuda_bytes = stats["allocated_bytes.all.peak"]

            self._task_queue = TorchProcessTaskQueue(
                worker_fun=functools.partial(
                    BatchComputationHook.common_worker_fun,
                    self._result_transform,
                    self._get_worker_fun(),
                ),
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self._task_queue.start()
        self._task_size = 0
        for data_piece in split_list_to_chunks(
            data,
            (len(data) + self._task_queue.worker_num - 1)
            // self._task_queue.worker_num,
        ):
            task = (model_with_loss, inputs, targets, data_piece)
            self._task_size += 1
            self._task_queue.add_task(task)

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
