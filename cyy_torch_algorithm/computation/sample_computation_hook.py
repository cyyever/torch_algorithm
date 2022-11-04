import functools
from typing import Callable

import torch
from cyy_torch_algorithm.computation.computation_hook import ComputationHook
# from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.hooks.add_index_to_dataset import AddIndexToDataset
from cyy_torch_toolbox.tensor import tensor_to


class SampleComputationHook(ComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_index_hook = AddIndexToDataset()
        self.__sample_selector = None
        self.__input_transform: Callable | None = None

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_input_transform(self, transform: Callable) -> None:
        self.__input_transform = transform

    def set_computed_indices(self, indices):
        self.set_sample_selector(lambda sample_index, *args: sample_index in indices)

    def add_task(
        self,
        model_with_loss,
        batch_index,
        sample_indices,
        inputs,
        targets,
        input_features=None,
        batch_dim=0,
    ):
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
            if isinstance(sample_input, torch.Tensor):
                sample_input = sample_input.unsqueeze(batch_dim)
            if input_feature is not None:
                input_feature = input_feature.unsqueeze(batch_dim)
            sample_target = sample_target.unsqueeze(0)
            if self.__input_transform is not None:
                res = self.__input_transform(
                    sample_index=sample_index,
                    sample_input=sample_input,
                    input_feature=input_feature,
                )
                match res:
                    case None:
                        pass
                    case list():
                        for new_input in res:
                            processed_indices.append(
                                new_input.get("sample_index", sample_index)
                            )
                            processed_features.append(
                                new_input.get("input_feature", None)
                            )
                            if processed_features[-1] is None:
                                if "sample_input" in new_input:
                                    processed_inputs.append(new_input["sample_input"])
                                else:
                                    processed_inputs.append(sample_input.clone())
                            else:
                                processed_inputs.append(None)
                            processed_targets.append(sample_target.clone())
                    case _:
                        raise NotImplementedError()
            else:
                processed_indices.append(sample_index)
                processed_inputs.append(sample_input)
                processed_features.append(input_feature)
                processed_targets.append(sample_target)
        if not processed_indices:
            return

        # fectch pending results
        self.result_dict
        for sample_index, sample_input, sample_input_feature, targrt in zip(
            processed_indices, processed_inputs, processed_features, processed_targets
        ):
            self._add_task(
                task=(
                    batch_index,
                    sample_index,
                    sample_input,
                    sample_input_feature,
                    targrt,
                ),
            )
        self._broadcast_one_shot_data(
            batch_index=batch_index, model_with_loss=model_with_loss
        )

    def _after_optimizer_step(self, step_skipped: bool, **kwargs) -> None:
        if step_skipped:
            self._drop_result()

    def _get_sample_computation_fun(self):
        raise NotImplementedError()

    def _get_worker_fun(self):
        return functools.partial(
            SampleComputationHook.common_worker_fun,
            self._result_transform,
            self._get_sample_computation_fun(),
        )

    def _after_forward(
        self,
        model_executor,
        batch_index,
        inputs,
        input_features,
        targets,
        sample_indices,
        **kwargs
    ):
        inputs, batch_dim, input_features = model_executor.split_batch_input(
            inputs=inputs, targets=targets, input_features=input_features
        )
        self.add_task(
            model_with_loss=model_executor.model_with_loss,
            batch_index=batch_index,
            sample_indices=sample_indices.tolist(),
            inputs=inputs,
            input_features=input_features,
            targets=targets,
            batch_dim=batch_dim,
        )

    @classmethod
    def common_worker_fun(
        cls, result_transform, worker_fun, tasks, device, worker_queue, **kwargs
    ):
        # counter = TimeCounter()
        worker_device, worker_stream = ComputationHook._setup_cuda_device(
            device,
        )

        with torch.cuda.stream(worker_stream):
            tasks = tensor_to(
                tasks, device=worker_device, non_blocking=True, check_slowdown=True
            )

            batch_index = tasks[0][0]
            batch_size = len(tasks)
            sample_indices = [task[1] for task in tasks]
            inputs = [task[2] for task in tasks]
            input_features = [task[3] for task in tasks]
            targets = [task[4] for task in tasks]
            model_data = cls.get_cached_one_shot_data(
                batch_index=batch_index,
                worker_device=worker_device,
                worker_queue=worker_queue,
            )

            is_input_feature = input_features[0] is not None
            if is_input_feature:
                inputs = input_features
            worker_fun = ComputationHook.get_cached_function(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(
                sample_indices=sample_indices,
                inputs=inputs,
                input_features=input_features,
                targets=targets,
                worker_device=worker_device,
                **model_data,
            )
            result_transform = ComputationHook.get_cached_function(
                "result_transform", result_transform, worker_device=worker_device
            )
            if result_transform is not None:
                for sample_index, input_tensor, input_feature, target in zip(
                    sample_indices, inputs, input_features, targets
                ):
                    res[sample_index] = result_transform(
                        sample_index=sample_index,
                        result=res[sample_index],
                        input_tensor=input_tensor,
                        input_feature=input_feature,
                        target=target,
                    )
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    res[k] = v.item()
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if v2.numel() == 1:
                        v[k2] = v2.item()
        # get_logger().error("use %s ms", counter.elapsed_milliseconds())
        return batch_size, res


def sample_dot_product(
    sample_index, result, input_tensor, input_feature, target, vector
):
    return result.dot(vector)
