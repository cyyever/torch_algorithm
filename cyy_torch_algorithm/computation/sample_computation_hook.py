import functools
from typing import Callable

import torch
from cyy_torch_toolbox.tensor import recursive_tensor_op, tensor_to

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
        is_input_feature,
        batch_dim=0,
    ) -> None:
        processed_indices = []
        processed_inputs = []
        processed_targets = []
        for sample_index, sample_input, sample_target in zip(
            sample_indices, inputs, targets
        ):
            if self.__sample_selector is not None and not self.__sample_selector(
                sample_index, sample_input
            ):
                continue
            if isinstance(sample_input, torch.Tensor):
                sample_input = sample_input.unsqueeze(batch_dim)
            sample_target = sample_target.unsqueeze(0)
            if self.__input_transform is not None:
                res = self.__input_transform(
                    sample_index=sample_index,
                    sample_input=sample_input,
                )
                match res:
                    case None:
                        pass
                    case list():
                        for new_input in res:
                            processed_indices.append(
                                new_input.get("sample_index", sample_index)
                            )
                            if "sample_input" in new_input:
                                processed_inputs.append(new_input["sample_input"])
                            else:
                                processed_inputs.append(sample_input.clone())
                            processed_targets.append(sample_target.clone())
                    case _:
                        raise NotImplementedError()
            else:
                processed_indices.append(sample_index)
                processed_inputs.append(sample_input)
                processed_targets.append(sample_target)
        if not processed_indices:
            return
        self._broadcast_one_shot_data(
            batch_index=self.__batch_index, model_evaluator=model_evaluator
        )
        for item in zip(processed_indices, processed_inputs, processed_targets):
            self._add_task(
                task=(self.__batch_index, *item, is_input_feature),
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
        is_input_feature = input_features is not None
        if is_input_feature:
            inputs = input_features
        if executor is not None:
            model_evaluator = executor.model_evaluator
        else:
            model_evaluator = kwargs["model_evaluator"]
        inputs, batch_dim = model_evaluator.split_batch_input(
            inputs=inputs, targets=targets
        )

        self.add_task(
            model_evaluator=model_evaluator,
            sample_indices=sample_indices.tolist(),
            inputs=inputs,
            is_input_feature=is_input_feature,
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
            targets = [task[3] for task in tasks]
            is_input_feature = tasks[0][4]
            model_data = cls.get_cached_one_shot_data(
                batch_index=batch_index,
                worker_device=worker_device,
                worker_queue=worker_queue,
            )

            worker_fun = ComputationHook.get_cached_item(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(
                sample_indices=sample_indices,
                inputs=inputs,
                targets=targets,
                worker_device=worker_device,
                is_input_feature=is_input_feature,
                **model_data,
            )
            result_transform = ComputationHook.get_cached_item(
                "result_transform", result_transform, worker_device=worker_device
            )
            if result_transform is not None:
                for sample_index, input_tensor, target in zip(
                    sample_indices, inputs, targets
                ):
                    res[sample_index] = result_transform(
                        sample_index=sample_index,
                        result=res[sample_index],
                        input_tensor=input_tensor,
                        target=target,
                    )

        def result_transform2(tensor, **kwargs):
            if tensor.numel() == 1:
                return tensor.view(-1).item()
            tensor.share_memory_()
            return tensor

        res = recursive_tensor_op(res, result_transform2)
        return batch_size, res


def dot_product(a, b, **kwargs) -> float:
    match b:
        case dict():
            product = 0
            for k, v in b.items():
                if v.device == a[k].device:
                    product += v.view(-1).dot(a[k].view(-1)).item()
                else:
                    product += v.cpu().view(-1).dot(a[k].cpu().view(-1)).item()
            return product
        case _:
            a = a.view(-1)
            b = b.view(-1)
            if a.device == b.device:
                return a.dot(b).item()
            return a.cpu().dot(b.cpu()).item()
