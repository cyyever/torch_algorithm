import torch
from cyy_torch_toolbox.data_structure.torch_process_task_queue import \
    TorchProcessTaskQueue
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import DatasetType


class SampleComputationHook(Hook):
    def __init__(self, worker_fun, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__computed_indices: set | None = None
        self.__sample_result_dict = None
        self.__task_queue = None
        self.__task_size = None
        self.__worker_fun = worker_fun

    @property
    def task_queue(self):
        return self.__task_queue

    @property
    def sample_result_dict(self):
        if self.__sample_result_dict is None:
            if self.__task_size is None:
                return {}
            self.__sample_result_dict = {}
            for _ in range(self.__task_size):
                self.__sample_result_dict |= self.__task_queue.get_result()
        return self.__sample_result_dict

    def set_computed_indices(self, computed_indices):
        self.__computed_indices = set(computed_indices)

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        _, instance_inputs, instance_targets, instance_info = trainer.decode_batch(
            batch
        )
        instance_indices = {idx.data.item() for idx in instance_info["index"]}
        self.__sample_result_dict = None
        self.__task_size = None

        sample_indices = []
        real_inputs = []
        real_targets = []
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
            real_inputs.append(instance_input)
            real_targets.append(instance_target)
            sample_indices.append(instance_index)
        if not sample_indices:
            return
        self.__compute_sample_info(
            trainer,
            sample_indices,
            real_inputs,
            real_targets,
        )

    def _after_execute(self, **_):
        self.__sample_result_dict = None
        if self.__task_queue is not None:
            self.__task_queue.release()
            self.__task_queue = None

    def _process_samples(self, sample_indices: list, inputs: list, targets: list):
        raise NotImplementedError()

    def __compute_sample_info(
        self, trainer, sample_indices: list, inputs: list, targets: list
    ):
        trainer.model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss = trainer.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.cpu()
        if self.__task_queue is None:
            max_needed_cuda_bytes = None
            stats = torch.cuda.memory_stats(device=trainer.device)
            if stats:
                max_needed_cuda_bytes = stats["allocated_bytes.all.peak"]

            self.__task_queue = TorchProcessTaskQueue(
                worker_fun=self.__worker_fun,
                move_data_in_cpu=True,
                max_needed_cuda_bytes=max_needed_cuda_bytes,
            )
            self.__task_queue.start()
        self.__task_size = 0
        for task in self._process_samples(sample_indices, inputs, targets):
            self.__task_size += 1
            self.__task_queue.add_task((model_with_loss, task))
