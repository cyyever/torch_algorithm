import functools
from typing import Callable

from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_gradient import sample_gradient_worker_fun


class SampleGradientHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__gradient_transform: Callable | None = None
        self.set_gradient_transform(None)

    def set_gradient_transform(self, f):
        self.__gradient_transform = f

    def _get_worker_fun(self):
        return functools.partial(sample_gradient_worker_fun, self.__gradient_transform)
