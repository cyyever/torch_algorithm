import functools

from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_vjp import sample_vjp_worker_fun


class SampleVJPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__vector = None

    def set_vector(self, vector):
        self.__vector = vector

    def _get_worker_fun(self):
        return functools.partial(sample_vjp_worker_fun, self.__vector)
