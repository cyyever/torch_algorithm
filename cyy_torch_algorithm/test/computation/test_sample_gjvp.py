#!/usr/bin/env python3
import torch
import torch.nn
from cyy_torch_algorithm.computation.sample_gjvp.sample_gjvp_hook import \
    SampleGradientJVPHook
from cyy_torch_toolbox.default_config import Config
from cyy_torch_toolbox.ml_type import ExecutorHookPoint, StopExecutingException


def test_CV_jvp():
    config = Config("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleGradientJVPHook()
    hook.set_vector(torch.ones((32, 32)).view(-1))
    trainer.append_hook(hook)

    def print_products(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "check results", print_products
    )
    trainer.train()


# def test_NLP_vjp():
#     config = Config("IMDB", "TransformerClassificationModel")
#     config.model_config.model_kwargs["max_len"] = 300
#     config.model_config.model_kwargs["d_model"] = 100
#     config.model_config.model_kwargs["nhead"] = 5
#     config.model_config.model_kwargs["num_encoder_layer"] = 1
#     config.dc_config.dataset_kwargs["max_len"] = 300
#     config.hyper_parameter_config.epoch = 1
#     config.hyper_parameter_config.learning_rate = 0.1
#     config.hyper_parameter_config.find_learning_rate = False
#     trainer = config.create_trainer()
#     trainer.model_util.cache_buffer_names()
#     trainer.model_evaluator.need_input_features = True
#     trainer.model_util.freeze_modules(module_type=torch.nn.Embedding)
#     hook = SampleGradientJVPHook()
#     hook.set_vector(torch.ones((1, 100 * 300)).view(-1))
#     trainer.append_hook(hook)

#     def print_result(**kwargs):
#         if hook.result_dict:
#             print(hook.result_dict)
#             hook.reset()
#             raise StopExecutingException()

#     trainer.append_named_hook(
#         ExecutorHookPoint.AFTER_BATCH, "check results", print_result
#     )
#     trainer.train()
