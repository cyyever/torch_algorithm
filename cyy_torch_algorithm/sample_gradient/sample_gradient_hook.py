import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
# from cyy_torch_toolbox.data_structure.torch_thread_task_queue import \
#     TorchThreadTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import DatasetType

from .sample_gradient import (sample_gradient_worker_fun,
                              sample_gradient_worker_fun2)

# sample_gradient_worker_fun,


class SampleGradientHook(Hook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__computed_indices: set | None = None
        self.__sample_gradient_dict = None
        self.__sample_gradient_indices = None
        self.__task_queue = None
        self.__task_size = None
        self.use_new = True

    @property
    def sample_gradient_dict(self):
        if self.__sample_gradient_dict is None:
            if self.__task_size is None:
                return {}
            gradient_dict = {}
            for _ in range(self.__task_size):
                idx, gradient_list = self.__task_queue.get_result()
                gradient_dict[idx] = gradient_list

            gradient_list = []
            for idx in range(self.__task_size):
                gradient_list += gradient_dict[idx]
            assert len(gradient_list) == len(self.__sample_gradient_indices)
            self.__sample_gradient_dict = dict(
                zip(self.__sample_gradient_indices, gradient_list)
            )
        return self.__sample_gradient_dict

    def set_computed_indices(self, computed_indices):
        self.__computed_indices = set(computed_indices)

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        instance_inputs, instance_targets, instance_info = trainer.decode_batch(batch)
        instance_indices = {idx.data.item() for idx in instance_info["index"]}
        self.__sample_gradient_dict = None
        self.__task_size = None

        sample_gradient_inputs = []
        sample_gradient_targets = []
        self.__sample_gradient_indices = []
        dimension_permuted = False
        if trainer.dataset_collection.dataset_type == DatasetType.Text:
            if (
                instance_inputs.shape[0] != instance_targets.shape[0]
                and instance_inputs.shape[1] == instance_targets.shape[0]
            ):
                instance_inputs = instance_inputs.permute(1, 0)
                dimension_permuted = True

        for (instance_input, instance_target, instance_index) in zip(
            instance_inputs, instance_targets, instance_indices
        ):
            if (
                self.__computed_indices is not None
                and instance_index not in self.__computed_indices
            ):
                continue
            if not dimension_permuted:
                instance_input.unsqueeze_(0)
            else:
                instance_input.unsqueeze_(1)
            instance_target.unsqueeze_(0)
            sample_gradient_inputs.append(instance_input)
            sample_gradient_targets.append(instance_target)
            self.__sample_gradient_indices.append(instance_index)
        if not self.__sample_gradient_indices:
            return
        self.__compute_sample_gradient(
            trainer,
            sample_gradient_inputs,
            sample_gradient_targets,
        )

    def _after_execute(self, **_):
        self.__sample_gradient_dict = None
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None

    def __compute_sample_gradient(self, trainer, inputs, targets):
        trainer.model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss = trainer.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.cpu()
        if self.__task_queue is None:
            max_needed_cuda_bytes = None
            stats = torch.cuda.memory_stats(device=trainer.device)
            if stats:
                max_needed_cuda_bytes = stats["allocated_bytes.all.peak"]

            if self.use_new:
                worker_fun = sample_gradient_worker_fun2
            else:
                worker_fun = sample_gradient_worker_fun
            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=worker_fun,
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self.__task_queue.start()
        input_chunks = split_list_to_chunks(
            inputs,
            (len(inputs) + self.__task_queue.worker_num - 1)
            // self.__task_queue.worker_num,
        )

        target_chunks = split_list_to_chunks(
            targets,
            (len(targets) + self.__task_queue.worker_num - 1)
            // self.__task_queue.worker_num,
        )
        self.__task_size = 0
        for idx, (input_chunk, target_chunk) in enumerate(
            zip(input_chunks, target_chunks)
        ):
            self.__task_size += 1
            self.__task_queue.add_task(
                (idx, input_chunk, target_chunk, model_with_loss)
            )
