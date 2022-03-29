from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_gradient import sample_gradient_worker_fun


class SampleGradientHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(worker_fun=sample_gradient_worker_fun, **kwargs)

    @property
    def sample_gradient_dict(self):
        return super().sample_result_dict
