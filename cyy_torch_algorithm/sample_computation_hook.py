from typing import Callable

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import DatasetType


class SampleComputationHook(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__sample_selector = None
        self.__sample_result_dict = None
        self._task_queue = None
        self.__task_size: int | None = None
        self.extra_args: dict = {}

    def _get_worker_fun(self) -> Callable:
        raise NotImplementedError()

    def iterate_result(self):
        if self.__task_size is None:
            return
        for _ in range(self.__task_size):
            for sample_index, result in self._task_queue.get_result().items():
                yield (sample_index, result)
        return

    @property
    def sample_result_dict(self):
        if self.__sample_result_dict is None:
            if self.__task_size is None:
                return {}
            self.__sample_result_dict = {}
            for _ in range(self.__task_size):
                self.__sample_result_dict |= self._task_queue.get_result()
        return self.__sample_result_dict

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_computed_indices(self, indices):
        self.set_sample_selector(lambda sample_index, *args: sample_index in indices)

    def _after_batch(self, inputs, input_embeddings, targets, batch_info, **kwargs):
        trainer = kwargs["model_executor"]
        instance_indices = [idx.data.item() for idx in batch_info["index"]]
        self.__sample_result_dict = None
        self.__task_size = None

        dimension_permuted = False
        if trainer.dataset_collection.dataset_type == DatasetType.Text:
            if (
                inputs.shape[0] != targets.shape[0]
                and inputs.shape[1] == targets.shape[0]
            ):
                inputs = inputs.permute(1, 0)
                if input_embeddings is not None:
                    input_embeddings = input_embeddings.permute(1, 0, 2)
                dimension_permuted = True
        if input_embeddings is None:
            input_embeddings = [None] * len(instance_indices)

        sample_indices = []
        processed_inputs = []
        processed_embeddings = []
        processed_targets = []
        for (instance_input, input_embedding, instance_target, instance_index) in zip(
            inputs, input_embeddings, targets, instance_indices
        ):
            if (self.__sample_selector is not None) and not self.__sample_selector(
                instance_index, instance_input
            ):
                continue
            unsqueeze_idx = 0 if not dimension_permuted else 1
            instance_input.unsqueeze_(unsqueeze_idx)
            if input_embedding is not None:
                input_embedding.unsqueeze_(unsqueeze_idx)
            instance_target.unsqueeze_(0)
            sample_indices.append(instance_index)
            processed_inputs.append(instance_input)
            processed_embeddings.append(input_embedding)
            processed_targets.append(instance_target)
        if not sample_indices:
            return
        self.__schedule_computation(
            trainer,
            sample_indices,
            processed_inputs,
            processed_embeddings,
            processed_targets,
        )

    def _after_execute(self, **_):
        self.__sample_result_dict = None
        if self._task_queue is not None:
            self._task_queue.release()
            self._task_queue = None

    def _process_samples(
        self,
        model_with_loss,
        sample_indices: list,
        inputs: list,
        embeddings: list,
        targets: list,
    ) -> list:
        return zip(
            *(
                tuple(
                    split_list_to_chunks(
                        data,
                        (len(data) + self._task_queue.worker_num - 1)
                        // self._task_queue.worker_num,
                    )
                )
                for data in (sample_indices, inputs, embeddings, targets)
            )
        )

    def __schedule_computation(
        self,
        trainer,
        sample_indices: list,
        inputs: list,
        embeddings: list,
        targets: list,
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
                worker_fun=self._get_worker_fun(),
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self._task_queue.start()
        self.__task_size = 0
        for task in self._process_samples(
            model_with_loss, sample_indices, inputs, embeddings, targets
        ):
            self.__task_size += 1
            if self.extra_args:
                self._task_queue.add_task((model_with_loss, task, self.extra_args))
            else:
                self._task_queue.add_task((model_with_loss, task))
