import functools
from typing import Callable

from cyy_torch_algorithm.sample_computation_hook import SampleComputationHook

from .sample_vjp import sample_vjp_worker_fun


class SampleVJPHook(SampleComputationHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__product_transform: Callable | None = None
        self.__vector = None

    def set_product_transform(self, f):
        self.__product_transform = f

    def set_vector(self, vector):
        self.__vector = vector

    def _get_worker_fun(self):
        return functools.partial(
            sample_vjp_worker_fun, self.__product_transform, self.__vector
        )
