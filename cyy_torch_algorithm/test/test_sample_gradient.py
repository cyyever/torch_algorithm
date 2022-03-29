#!/usr/bin/env python3

# import os

from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)

# from cyy_torch_toolbox.reproducible_env import global_reproducible_env

# global_reproducible_env.enable()


def test_get_sample_gradient():
    config = DefaultConfig("MNIST", "lenet5")
    config.make_reproducible_env = True
    config.apply_global_config()
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleGradientHook()
    hook.set_computed_indices(set(range(10)))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.sample_gradient_dict:
            print(hook.sample_gradient_dict)
            # raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()

    # global_reproducible_env.disable()
    # global_reproducible_env.load_last_seed()
    # global_reproducible_env.enable()
    # os.environ["reseed_dropout"] = "1"

    # config = DefaultConfig("MNIST", "lenet5")
    # config.hyper_parameter_config.epoch = 1
    # config.hyper_parameter_config.batch_size = 8
    # config.hyper_parameter_config.learning_rate = 0.01
    # config.hyper_parameter_config.find_learning_rate = False
    # trainer = config.create_trainer()
    # hook = SampleGradientHook()
    # hook.use_new = False
    # hook.set_computed_indices([1])
    # trainer.append_hook(hook)

    # trainer.append_named_hook(
    #     ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    # )
    # trainer.train()
