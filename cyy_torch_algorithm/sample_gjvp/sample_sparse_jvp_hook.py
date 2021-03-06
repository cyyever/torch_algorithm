from typing import Callable

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

from .sample_sparse_jvp import sample_sparse_jvp_worker_fun


class SampleSparseJVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__sample_vector_fun: None | Callable = None

    def set_sample_vector_fun(self, sample_vector_fun: Callable) -> None:
        self.__sample_vector_fun = sample_vector_fun

    def _get_worker_fun(self):
        return sample_sparse_jvp_worker_fun

    def _process_samples(
        self,
        model_with_loss: ModelWithLoss,
        sample_indices: list,
        inputs: list,
        targets: list,
    ) -> list:
        assert self.__sample_vector_fun is not None
        vectors = []

        if hasattr(model_with_loss.model, "forward_embedding"):
            sample_inputs = inputs
            inputs = [
                model_with_loss.model.get_embedding(sample_input).detach()
                for sample_input in inputs
            ]
            for idx, sample_input, sample_embedding in zip(
                sample_indices, sample_inputs, inputs
            ):
                vectors.append(
                    self.__sample_vector_fun(idx, sample_input, sample_embedding)
                )
        else:
            for idx, sample_input in zip(sample_indices, inputs):
                vectors.append(self.__sample_vector_fun(idx, sample_input, None))
        return list(
            zip(
                *(
                    tuple(
                        split_list_to_chunks(
                            data,
                            (len(data) + self._task_queue.worker_num - 1)
                            // self._task_queue.worker_num,
                        )
                    )
                    for data in (sample_indices, inputs, targets, vectors)
                )
            )
        )
