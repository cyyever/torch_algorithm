import functools

import torch
import torch.cuda
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.batch_computation_hook import BatchComputationHook
# from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.evaluation import eval_model
from cyy_torch_toolbox.device import put_data_to_device
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from functorch import grad, jvp, vmap


def batch_hvp_worker_fun(
    model_with_loss,
    inputs,
    targets,
    data,
    worker_device,
    worker_stream,
):
    # time_counter = TimeCounter(debug_logging=False)
    # time_counter.reset_start_time()
    model_with_loss.model.to(worker_device)
    parameter_list = model_with_loss.model_util.get_parameter_list(detach=True)
    with torch.cuda.stream(worker_stream):
        vectors = put_data_to_device(data, device=worker_device, non_blocking=True)
        inputs = put_data_to_device(inputs, device=worker_device, non_blocking=True)
        targets = put_data_to_device(targets, device=worker_device, non_blocking=True)

        def vjp_wrapper(vector):
            return jvp(
                grad(
                    functools.partial(
                        eval_model,
                        inputs=inputs,
                        targets=targets,
                        device=worker_device,
                        model_with_loss=model_with_loss,
                        phase=MachineLearningPhase.Test,
                        non_blocking=True,
                    )
                ),
                (parameter_list,),
                (vector,),
            )[1]

        products = vmap(vjp_wrapper, randomness="different",)(
            torch.stack(vectors),
        )
        # get_logger().error("use %s ms", time_counter.elapsed_milliseconds())
        return {0: products}


class BatchHVPHook(BatchComputationHook):
    def set_vectors(self, vectors):
        self.set_data_fun(lambda: vectors)

    def _get_worker_fun(self):
        return batch_hvp_worker_fun
