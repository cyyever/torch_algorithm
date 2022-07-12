#!/usr/bin/env python3
import torch
import torch.nn
from cyy_torch_algorithm.sample_gjvp.sample_gjvp_hook import \
    SampleGradientJVPHook
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
    hook = SampleGradientJVPHook()
    hook.set_vector(torch.ones((32, 32)).view(-1))
    trainer.append_hook(hook)

    def print_products(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check results", print_products
    )
    trainer.train()


def test_NLP_vjp():
    config = DefaultConfig("IMDB", "TransformerClassificationModel")
    config.model_config.model_kwargs["max_len"] = 300
    config.model_config.model_kwargs["d_model"] = 100
    config.model_config.model_kwargs["nhead"] = 5
    config.model_config.model_kwargs["num_encoder_layer"] = 1
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.1
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    trainer.model_util.freeze_modules(module_type=torch.nn.Embedding)
    hook = SampleGradientJVPHook()
    hook.set_vector(torch.ones((1, 100*300)).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check results", print_result
    )
    trainer.train()
