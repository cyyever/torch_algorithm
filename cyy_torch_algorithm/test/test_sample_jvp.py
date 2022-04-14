#!/usr/bin/env python3
import torch
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
    hook.set_computed_indices([1])
    hook.set_sample_vector_fun(lambda *_: [torch.zeros((32, 32)).reshape(-1)])
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.sample_jvp_dict:
            print(hook.sample_jvp_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()


def test_NLP_jvp():
    config = DefaultConfig("IMDB", "TransformerClassificationModel")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    config.model_config.model_kwargs = {
        "max_len": 500,
        "word_vector_name": "glove.6B.200d",
        "num_encoder_layer": 1,
        "d_model": 200,
    }

    trainer = config.create_trainer()
    hook = SampleJVPHook()
    # hook.set_computed_indices([1])
    hook.set_sample_vector_fun(
        lambda idx, sample_input: [
            torch.zeros_like(sample_input).reshape(-1),
            torch.ones_like(sample_input).reshape(-1),
        ]
    )
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.sample_jvp_dict:
            print(hook.sample_jvp_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()
