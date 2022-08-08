import functools

import torch
import torch.cuda
# from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_computation_hook import \
    BatchComputationHook
# from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.computation.evaluation import eval_model
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from functorch import grad, jvp, vmap


def batch_hvp_worker_fun(
    model_with_loss,
    inputs,
    targets,
    data,
    worker_device,
):
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    vectors = data

    def hvp_wrapper(vector):
        f = functools.partial(
            eval_model,
            inputs=inputs,
            targets=targets,
            device=worker_device,
            model_with_loss=model_with_loss,
            phase=MachineLearningPhase.Test,
            non_blocking=True,
        )

        def grad_f(parameter_list):
            return grad(f, argnums=0)(parameter_list).view(-1)

        return jvp(
            grad_f,
            (parameter_list,),
            (vector,),
        )[1]

    if not isinstance(vectors, torch.Tensor):
        vectors = torch.stack(vectors)

    products = vmap(hvp_wrapper, randomness="same")(vectors)
    # get_logger().error("use %s ms", time_counter.elapsed_milliseconds())
    return {0: products}


class BatchHVPHook(BatchComputationHook):
    def set_vectors(self, vectors):
        self.set_data_fun(lambda: vectors)

    def _get_worker_fun(self):
        return batch_hvp_worker_fun
