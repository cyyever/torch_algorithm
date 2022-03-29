from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_gradient import sample_gradient_worker_fun


class SampleGradientHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(worker_fun=sample_gradient_worker_fun, **kwargs)

    @property
    def sample_gradient_dict(self):
        return super().sample_result_dict

    def _process_samples(self, sample_indices, inputs, targets):
        return zip(
            *(
                tuple(
                    split_list_to_chunks(
                        data,
                        (len(data) + self.task_queue.worker_num - 1)
                        // self.task_queue.worker_num,
                    )
                )
                for data in (sample_indices, inputs, targets)
            )
        )
