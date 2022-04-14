from typing import Callable

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_jvp import sample_jvp_worker_fun


class SampleJVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(worker_fun=sample_jvp_worker_fun, **kwargs)
        self.__sample_vector_fun: None | Callable = None

    @property
    def sample_jvp_dict(self) -> dict:
        return super().sample_result_dict

    def set_sample_vector_fun(self, sample_vector_fun: Callable) -> None:
        self.__sample_vector_fun = sample_vector_fun

    def _process_samples(
        self, model_with_loss, sample_indices: list, inputs: list, targets: list
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
                            (len(data) + self.task_queue.worker_num - 1)
                            // self.task_queue.worker_num,
                        )
                    )
                    for data in (sample_indices, inputs, targets, vectors)
                )
            )
        )
