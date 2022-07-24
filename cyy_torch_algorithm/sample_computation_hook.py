import functools
from typing import Callable

from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.ml_type import DatasetType

from cyy_torch_algorithm.computation_hook import ComputationHook


class SampleComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__sample_selector = None

    @property
    def sample_result_dict(self):
        return self.result_dict

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_computed_indices(self, indices):
        self.set_sample_selector(lambda sample_index, *args: sample_index in indices)

    def _after_forward(
        self, model_executor, inputs, input_features, targets, batch_info, **kwargs
    ):
        trainer = model_executor
        self._result_dict = None

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

        model_with_loss = trainer.copy_model_with_loss(deepcopy=True)
        model_with_loss.model.zero_grad(set_to_none=True)
        model_with_loss.model.cpu()
        worker_fun = functools.partial(
            SampleComputationHook.common_worker_fun,
            self._result_transform,
            self._get_worker_fun(),
        )
        for task in self._split_data(
            [processed_indices, processed_inputs, processed_features, processed_targets]
        ):
            self._add_task(
                trainer=trainer, worker_fun=worker_fun, task=(model_with_loss, *task)
            )

    @classmethod
    def common_worker_fun(cls, result_transform, worker_fun, task, args):
        worker_device, worker_stream = ComputationHook._setup_cuda_device(
            args["device"]
        )
        model_with_loss, sample_indices, inputs, input_features, targets = task
        res = worker_fun(
            model_with_loss=model_with_loss,
            sample_indices=sample_indices,
            inputs=inputs,
            input_features=input_features,
            targets=targets,
            worker_device=worker_device,
            worker_stream=worker_stream,
        )
        if result_transform is not None:
            for index, input_tensor, input_feature, target in zip(
                sample_indices, inputs, input_features, targets
            ):
                res[index] = result_transform(
                    index=index,
                    result=res[index],
                    input_tensor=input_tensor,
                    input_feature=input_feature,
                    target=target,
                )
        return res
