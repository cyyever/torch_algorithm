#!/usr/bin/env python3
import torch
import torch.nn
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException
from cyy_torch_toolbox.tensor import cat_tensor_dict


def test_CV_jvp():
    torch.autograd.set_detect_anomaly(True)
    config = DefaultConfig("MNIST", "lenet5")
    config.hook_config.use_amp = False
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()

    time_counter = TimeCounter(debug_logging=False)
    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD,
        "reset_time",
        lambda **kwargs: time_counter.reset_start_time()
        and get_logger().error("begin count time"),
    )
    hook = BatchHVPHook()
    trainer.append_hook(hook)

    def print_products(**kwargs):
        if hook.result_dict:
            products = hook.result_dict
            assert len(products) == 10
            get_logger().error("use time %s", time_counter.elapsed_milliseconds())
            assert (
                torch.linalg.vector_norm(
                    cat_tensor_dict(products[1]).cpu() * 4
                    - cat_tensor_dict(products[4]).cpu()
                ).item()
                < 0.05
            )
            del products
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "check results", print_products
    )
    parameter_dict = trainer.model_util.get_parameter_dict()
    vectors = []
    for i in range(10):
        vectors.append({})
        for k, v in parameter_dict.items():
            vectors[i][k] = torch.ones_like(v, device="cpu") * i
    hook.set_vectors(vectors)
    trainer.train()
    hook.reset()
