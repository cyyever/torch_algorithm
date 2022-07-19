import functools
import threading
from typing import Any, Callable

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import DatasetType

local_data = threading.local()


def setup_cuda_device(args):
    worker_device = getattr(local_data, "worker_device", None)
    if worker_device is None:
        worker_device = args["device"]
        local_data.worker_device = worker_device
    worker_stream = getattr(local_data, "worker_stream", None)
    if worker_stream is None:
        worker_stream = torch.cuda.Stream(device=worker_device)
        local_data.worker_stream = worker_stream
    return worker_device, worker_stream


def common_worker_fun(result_transform, worker_fun, task, args):
    worker_device, worker_stream = setup_cuda_device(args)
    model_with_loss, (sample_indices, inputs, input_features, targets, batch_dim) = task
    res = worker_fun(
        model_with_loss=model_with_loss,
        sample_indices=sample_indices,
        inputs=inputs,
        input_features=input_features,
        targets=targets,
        batch_dim=batch_dim,
        worker_device=worker_device,
        worker_stream=worker_stream,
    )
    if result_transform is not None:
        for index, input_tensor, input_feature, target in zip(
            sample_indices, inputs, input_features, targets
        ):
            if index in res:
                res[index] = result_transform(
                    index=index,
                    result=res[index],
                    input_tensor=input_tensor,
                    input_feature=input_feature,
                    target=target,
                )
    return res


class SampleComputationHook(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__sample_selector = None
        self.__sample_result_dict = None
        self.__task_queue = None
        self.__task_size: int | None = None
        self.__result_transform: Callable | None = None

    def set_result_transform(self, f):
        self.__result_transform = f

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    @property
    def sample_result_dict(self):
        if self.__sample_result_dict is None:
            if self.__task_size is None:
                return {}
            self.__sample_result_dict = {}
            for _ in range(self.__task_size):
                self.__sample_result_dict |= self.__task_queue.get_result()
        return self.__sample_result_dict

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_computed_indices(self, indices):
        self.set_sample_selector(lambda sample_index, *args: sample_index in indices)

    def _after_batch(
        self, model_executor, inputs, input_features, targets, batch_info, **kwargs
    ):
        trainer = model_executor
        self.__sample_result_dict = None
        self.__task_size = None

        batch_dim = 0
        if trainer.dataset_collection.dataset_type == DatasetType.Text:
            if (
                inputs.shape[0] != targets.shape[0]
                and inputs.shape[1] == targets.shape[0]
            ):
                inputs = inputs.permute(1, 0)
                if input_features is not None:
                    input_features = input_features.permute(1, 0, 2)
                batch_dim = 1
        sample_indices = [idx.data.item() for idx in batch_info["index"]]
        if input_features is None:
            input_features = [None] * len(sample_indices)

        processed_indices = []
        processed_inputs = []
        processed_features = []
        processed_targets = []
        for (sample_index, sample_input, input_feature, sample_target) in zip(
            sample_indices, inputs, input_features, targets
        ):
            if self.__sample_selector is not None and not self.__sample_selector(
                sample_index, sample_input
            ):
                continue
            sample_input = sample_input.unsqueeze(batch_dim)
            if input_feature is not None:
                input_feature = input_feature.unsqueeze(batch_dim)
            sample_target = sample_target.unsqueeze(0)
            processed_indices.append(sample_index)
            processed_inputs.append(sample_input)
            processed_features.append(input_feature)
            processed_targets.append(sample_target)
        if not processed_indices:
            return
        self.__schedule_computation(
            trainer=trainer,
            sample_indices=processed_indices,
            inputs=processed_inputs,
            input_features=processed_features,
            targets=processed_targets,
            batch_dim=batch_dim,
        )

    def _after_execute(self, **_):
        self.__sample_result_dict = None

    def __del__(self):
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None

    def __process_samples(
        self,
        sample_indices: list,
        inputs: list,
        input_features: list,
        targets: list,
    ) -> list:
        return zip(
            *(
                tuple(
                    split_list_to_chunks(
                        data,
                        (len(data) + self.__task_queue.worker_num - 1)
                        // self.__task_queue.worker_num,
                    )
                )
                for data in (sample_indices, inputs, input_features, targets)
            )
        )

    def __schedule_computation(
        self,
        trainer: Any,
        sample_indices: list,
        inputs: list,
        input_features: list,
        targets: list,
        batch_dim,
    ) -> None:
        model_with_loss = trainer.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.cpu()
        if self.__task_queue is None:
            max_needed_cuda_bytes = None
            stats = torch.cuda.memory_stats(device=trainer.device)
            if stats:
                max_needed_cuda_bytes = stats["allocated_bytes.all.peak"]

            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=functools.partial(
                    common_worker_fun, self.__result_transform, self._get_worker_fun()
                ),
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self.__task_queue.start()
        self.__task_size = 0
        for task in self.__process_samples(
            sample_indices, inputs, input_features, targets
        ):
            task = (*task, batch_dim)
            self.__task_size += 1
            self.__task_queue.add_task((model_with_loss, task))
