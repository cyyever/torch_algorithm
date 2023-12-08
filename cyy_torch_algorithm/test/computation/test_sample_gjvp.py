import importlib.util

has_cyy_torch_text: bool = importlib.util.find_spec("cyy_torch_text") is not None
has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None
import torch
import torch.nn
from cyy_torch_algorithm.computation.sample_gjvp.sample_gjvp_hook import \
    SampleGradientJVPHook
from cyy_torch_toolbox import Config, ExecutorHookPoint, StopExecutingException


def test_CV_jvp():
    if not has_cyy_torch_vision:
        return
    import cyy_torch_vision  # noqa: F401

    config = Config("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
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
        ExecutorHookPoint.AFTER_BATCH, "check results", print_products
    )
    trainer.train()


def test_NLP_vjp():
    if not has_cyy_torch_text:
        return
    import cyy_torch_text  # noqa: F401

    config = Config("imdb", "TransformerClassificationModel")
    config.dc_config.dataset_kwargs["max_len"] = 300
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    config.model_config.model_kwargs["max_len"] = 300
    config.model_config.model_kwargs["d_model"] = 100
    config.model_config.model_kwargs["nhead"] = 5
    config.model_config.model_kwargs["num_encoder_layer"] = 1
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.1
    trainer = config.create_trainer()
    hook = SampleGradientJVPHook()
    hook.set_vector(torch.ones((1, 100 * 300)).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check results", print_result
    )
    trainer.train()
