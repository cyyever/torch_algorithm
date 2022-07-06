#!/usr/bin/env python3
import torch
import torch.nn as nn
from cyy_torch_algorithm.sample_jvp.sample_jvp_hook import SampleJVPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_CV_jvp():
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleJVPHook()
    hook.set_vector(torch.ones((32,32)).view(-1))
    trainer.append_hook(hook)

    def print_products(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check results", print_products
    )
    trainer.train()
