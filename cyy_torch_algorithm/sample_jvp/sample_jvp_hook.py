from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_jvp import sample_jvp_worker_fun


class SampleJVPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(worker_fun=sample_jvp_worker_fun, **kwargs)
        self.__sample_vector: dict = {}

    @property
    def sample_jvp_dict(self) -> dict:
        return super().sample_result_dict

    def set_sample_vectors(self, index: int, vectors: list) -> None:
        self.__sample_vector[index] = vectors

    def _process_samples(self, sample_indices: list, inputs: list, targets: list):
        vectors = []
        for idx in sample_indices:
            vectors.append(self.__sample_vector[idx])
        return zip(
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