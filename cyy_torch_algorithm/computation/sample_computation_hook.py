import functools
from typing import Callable

import torch
from cyy_torch_toolbox.tensor import (cat_tensor_dict, recursive_tensor_op,
                                      tensor_to)

from .computation_hook import ComputationHook


class SampleComputationHook(ComputationHook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__sample_selector: Callable | None = None
        self.__input_transform: Callable | None = None
        self.__batch_index: int = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_SampleComputationHook__sample_selector"] = None
        return state

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_input_transform(self, transform: Callable) -> None:
        self.__input_transform = transform

    def set_computed_indices(self, indices):
        self.set_sample_selector(lambda sample_index, *args: sample_index in indices)

    def add_task(
        self,
        model_evaluator,
        sample_indices,
        inputs,
        targets,
        input_features=None,
        batch_dim=0,
    ) -> None:
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
        self._broadcast_one_shot_data(
            batch_index=self.__batch_index, model_evaluator=model_evaluator
        )
        for sample_index, sample_input, sample_input_feature, targrt in zip(
            processed_indices, processed_inputs, processed_features, processed_targets
        ):
            self._add_task(
                task=(
                    self.__batch_index,
                    sample_index,
                    sample_input,
                    sample_input_feature,
                    targrt,
                ),
            )
        self.__batch_index += 1

    def _get_sample_computation_fun(self):
        raise NotImplementedError()

    def _get_worker_fun(self):
        return functools.partial(
            SampleComputationHook.common_worker_fun,
            self._result_transform,
            self._get_sample_computation_fun(),
        )

    def _after_forward(
        self, executor, inputs, targets, sample_indices, input_features=None, **kwargs
    ):
        if executor is not None:
            inputs, batch_dim, input_features = executor.split_batch_input(
                inputs=inputs, targets=targets, input_features=input_features
            )
            model_evaluator = executor.model_evaluator
        else:
            model_evaluator = kwargs["model_evaluator"]
            batch_dim = 0
        self.add_task(
            model_evaluator=model_evaluator,
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
        worker_device, worker_stream = ComputationHook._setup_device(
            device,
        )

        with torch.cuda.stream(worker_stream):
            tasks = tensor_to(
                tasks, device=worker_device, non_blocking=True, check_slowdown=False
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
            worker_fun = ComputationHook.get_cached_item(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(
                sample_indices=sample_indices,
                inputs=inputs,
                input_features=input_features,
                targets=targets,
                worker_device=worker_device,
                is_input_feature=is_input_feature,
                **model_data,
            )
            result_transform = ComputationHook.get_cached_item(
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

        def result_transform2(tensor, **kwargs):
            if tensor.numel() == 1:
                return tensor.item()
            tensor.share_memory_()
            return tensor

        res = recursive_tensor_op(res, result_transform2)
        return batch_size, res


def sample_dot_product(result, vector, **kwargs) -> float:
    match vector:
        case dict():
            product: None | torch.Tensor = None
            for k, v in vector.items():
                tmp = v.view(-1).dot(result[k].view(-1))
                if product is None:
                    product = tmp
                else:
                    product += tmp
            assert product is not None
            return product.item()
    match result:
        case dict():
            result = cat_tensor_dict(result)

    return result.dot(vector).item()
