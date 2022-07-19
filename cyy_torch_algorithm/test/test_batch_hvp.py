#!/usr/bin/env python3
import torch
import torch.nn
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.batch_hvp.batch_hvp_hook import BatchHVPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_CV_jvp():
    torch.autograd.set_detect_anomaly(True)
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    parameter_vector = trainer.model_util.get_parameter_list()

    time_counter = TimeCounter(debug_logging=False)
    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH,
        "reset_time",
        lambda **kwargs: time_counter.reset_start_time() and get_logger().error("begin count time"),
    )
    hook = BatchHVPHook()
    trainer.append_hook(hook)

    def print_products(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            get_logger().error("use time %s", time_counter.elapsed_milliseconds())
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check results", print_products
    )
    v = torch.ones_like(parameter_vector).view(-1)
    hook.set_vectors([v * (i + 1) for i in range(100)])

    trainer.train()
    hook.set_vectors([v * (i + 1) for i in range(100)])

    trainer.train()
