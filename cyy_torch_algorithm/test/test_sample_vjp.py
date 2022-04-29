#!/usr/bin/env python3
import torch
import torch.nn as nn
from cyy_torch_algorithm.sample_vjp.sample_vjp_hook import SampleVJPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_CV_vjp():
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
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
    trainer.model_util.freeze_sub_modules(nn.Embedding)
    hook = SampleVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()
