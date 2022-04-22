import functools
from typing import Callable

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_gradient import sample_gradient_worker_fun


class SampleGradientHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__gradient_transform: Callable | None = None
        self.set_gradient_transform(None)

    @property
    def sample_gradient_dict(self):
        return super().sample_result_dict

    def set_gradient_transform(self, f):
        self.__gradient_transform = f
        super()._set_worker_fun(
            functools.partial(sample_gradient_worker_fun, self.__gradient_transform)
        )

    def _process_samples(self, model_with_loss, sample_indices, inputs, targets):
        return zip(
            *(
                tuple(
                    split_list_to_chunks(
                        data,
                        (len(data) + self._task_queue.worker_num - 1)
                        // self._task_queue.worker_num,
                    )
                )
                for data in (sample_indices, inputs, targets)
            )
        )
