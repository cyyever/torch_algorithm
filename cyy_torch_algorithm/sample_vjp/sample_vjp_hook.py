import functools

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

from .sample_vjp import sample_vjp_worker_fun


class SampleVJPHook(SampleComputationHook):
    def set_vector(self, vector):
        self._set_worker_fun(functools.partial(sample_vjp_worker_fun, vector))

    def _process_samples(
        self,
        model_with_loss: ModelWithLoss,
        sample_indices: list,
        inputs: list,
        targets: list,
    ) -> list:
        if hasattr(model_with_loss.model, "forward_embedding"):
            inputs = [
                model_with_loss.model.get_embedding(sample_input).detach()
                for sample_input in inputs
            ]
        return list(
            zip(
                split_list_to_chunks(
                    data,
                    (len(data) + self._task_queue.worker_num - 1)
                    // self._task_queue.worker_num,
                )
                for data in (sample_indices, inputs, targets)
            )
        )
