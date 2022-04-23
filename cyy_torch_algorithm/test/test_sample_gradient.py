#!/usr/bin/env python3


from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_CV_sample_gradient():
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
        if hook.sample_result_dict:
            print(hook.sample_result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()


# def test_NLP_sample_gradient():
#     config = DefaultConfig("IMDB", "simplelstm")
#     config.make_reproducible_env = True
#     config.apply_global_config()
#     config.hyper_parameter_config.epoch = 1
#     config.hyper_parameter_config.batch_size = 8
#     config.hyper_parameter_config.learning_rate = 0.01
#     config.hyper_parameter_config.find_learning_rate = False
#     trainer = config.create_trainer()
#     hook = SampleGradientHook()
#     hook.set_computed_indices(set(range(10)))
#     trainer.append_hook(hook)

#     def print_sample_gradients(**kwargs):
#         if hook.sample_gradient_dict:
#             print(hook.sample_gradient_dict)
#             raise StopExecutingException()

#     trainer.append_named_hook(
#         ModelExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
#     )
#     trainer.train()
