import functools
from collections.abc import Callable

import torch
from cyy_torch_toolbox import Executor, IndicesType, ModelEvaluator
from cyy_torch_toolbox.tensor import dot_product as dot_product_impl
from cyy_torch_toolbox.tensor import recursive_tensor_op, tensor_to

from .computation_hook import ComputationHook


class SampleComputationHook(ComputationHook):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__sample_selector: Callable | None = None
        self.__input_transform: Callable | None = None
        self.__batch_index: int = 0

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        state["_SampleComputationHook__sample_selector"] = None
        return state

    def set_sample_selector(self, selector: Callable) -> None:
        self.__sample_selector = selector

    def set_input_transform(self, transform: Callable) -> None:
        self.__input_transform = transform

    def set_computed_indices(self, indices: IndicesType) -> None:
        index_set = set(indices)
        self.set_sample_selector(lambda sample_index, *args: sample_index in index_set)

    def add_task(
        self,
        model_evaluator: ModelEvaluator,
        sample_indices: torch.Tensor,
        inputs,
        targets: torch.Tensor,
    ) -> None:
        res = model_evaluator.split_batch_input(
            inputs=inputs, batch_size=targets.shape[0]
        )
        inputs = res["inputs"]
        batch_dim = res["batch_dim"]

        processed_indices = []
        processed_inputs = []
        processed_targets = []

        for sample_index, sample_input, sample_target in zip(
            sample_indices.tolist(), inputs, targets, strict=False
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
            batch_index=self.__batch_index,
            model_evaluator=model_evaluator,
        )
        for item in zip(
            processed_indices, processed_inputs, processed_targets, strict=False
        ):
            self._add_task(
                task=(self.__batch_index, *item),
            )
        self.__batch_index += 1

    def _get_sample_computation_fun(self):
        raise NotImplementedError()

    def _get_worker_fun(self) -> Callable:
        return functools.partial(
            self.common_worker_fun,
            self._get_sample_computation_fun(),
        )

    def _before_batch(
        self,
        executor: Executor | None,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_indices: torch.Tensor,
        **kwargs,
    ) -> None:
        if executor is not None:
            model_evaluator = executor.model_evaluator
        else:
            model_evaluator = kwargs["model_evaluator"]
        self.add_task(
            model_evaluator=model_evaluator,
            sample_indices=sample_indices,
            inputs=inputs,
            targets=targets,
        )

    def common_worker_fun(
        self,
        worker_fun: Callable,
        tasks: list,
        device: torch.device,
        model_queue,
        **kwargs,
    ) -> tuple:
        worker_device, worker_stream = self._setup_device(
            device,
        )

        with torch.cuda.stream(worker_stream):
            tasks = tensor_to(
                tasks, device=worker_device, non_blocking=True, check_slowdown=False
            )
            batch_index: int = tasks[0][0]
            batch_size: int = len(tasks)
            sample_indices: list[int] = [task[1] for task in tasks]
            inputs: list = [task[2] for task in tasks]
            targets: list = [task[3] for task in tasks]
            model_data: dict = self.get_cached_one_shot_data(
                batch_index=batch_index,
                worker_device=worker_device,
                model_queue=model_queue,
            )
            forward_fun: str | None = None
            input_features: list = [
                model_data["model_evaluator"].get_input_feature(input_element)
                for input_element in inputs
            ]
            if input_features[0] is not None:
                inputs = input_features
                forward_fun = model_data["model_evaluator"].get_feature_forward_fun()
                model_data["model_evaluator"].set_forward_fun(forward_fun)

            worker_fun = self.get_cached_item(
                "worker_fun", worker_fun, worker_device=worker_device
            )
            res = worker_fun(
                sample_indices=sample_indices,
                inputs=inputs,
                targets=targets,
                worker_device=worker_device,
                **model_data,
            )
            result_transform = self.get_cached_item(
                "result_transform", self._result_transform, worker_device=worker_device
            )
            if result_transform is not None:
                for sample_index, input_tensor, target in zip(
                    sample_indices, inputs, targets, strict=False
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


def dot_product(result, rhs, **kwargs) -> float:
    return dot_product_impl(result, rhs)
