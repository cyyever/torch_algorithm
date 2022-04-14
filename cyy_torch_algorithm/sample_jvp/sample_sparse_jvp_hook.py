from .sample_jvp_hook import SampleJVPHook
from .sample_sparse_jvp import sample_sparse_jvp_worker_fun


class SampleSparseJVPHook(SampleJVPHook):
    def __init__(self, **kwargs):
        super().__init__(worker_fun=sample_sparse_jvp_worker_fun, **kwargs)
