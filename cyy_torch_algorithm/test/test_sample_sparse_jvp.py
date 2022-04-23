#!/usr/bin/env python3
import torch
from cyy_torch_algorithm.sample_jvp.sample_sparse_jvp_hook import \
    SampleSparseJVPHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_CV_jvp():
    return
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleSparseJVPHook()
    hook.set_computed_indices([1])

    def sample_vector_fun(_, sample_input, sample_embedding):
        vectors = []
        vectors.append((0, ((0, 5), (15, 20)), (torch.ones(5), torch.zeros(5))))
        return vectors

    hook.set_sample_vector_fun(sample_vector_fun)

    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        for _, product in hook.iterate_result():
            print(product)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()


def test_NLP_jvp():
    return
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
    hook = SampleSparseJVPHook()

    def sample_vector_fun(_, sample_input, sample_embedding):
        vectors = []
        vectors.append((0, ((0, 5),), (torch.ones(5),)))
        return vectors

    hook.set_sample_vector_fun(sample_vector_fun)

    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        for _, product in hook.iterate_result():
            print(product)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()
