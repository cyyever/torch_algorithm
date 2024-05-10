import importlib.util

import torch
from cyy_torch_algorithm.computation.sample_gvjp.sample_gvjp_hook import \
    SampleGradientVJPHook
from cyy_torch_toolbox import Config, ExecutorHookPoint, StopExecutingException
from torch import nn

has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None
has_cyy_torch_text: bool = importlib.util.find_spec("cyy_torch_text") is not None


def test_CV_vjp() -> None:
    if not has_cyy_torch_vision:
        return
    import cyy_torch_vision  # noqa: F401

    config = Config("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    hook = SampleGradientVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs) -> None:
        if hook.result_dict:
            print(hook.result_dict)
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_result
    )
    trainer.train()
    hook.reset()


def test_NLP_vjp() -> None:
    if not has_cyy_torch_text:
        return
    import cyy_torch_text  # noqa: F401

    config = Config("imdb", "TransformerClassificationModel")
    config.dc_config.dataset_kwargs["input_max_len"] = 300
    config.dc_config.dataset_kwargs["tokenizer"] = {"type": "spacy"}
    config.model_config.model_kwargs["d_model"] = 100
    config.model_config.model_kwargs["nhead"] = 5
    config.model_config.model_kwargs["num_encoder_layer"] = 1
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.1
    trainer = config.create_trainer()
    trainer.model_util.freeze_modules(module_type=nn.Embedding)
    hook = SampleGradientVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_result
    )
    trainer.train()


def test_hugging_face_vjp() -> None:
    if not has_cyy_torch_text:
        return
    import cyy_torch_text  # noqa: F401

    config = Config(
        "imdb", "hugging_face_sequence_classification_distilbert-base-cased"
    )
    config.dc_config.dataset_kwargs["input_max_len"] = 300
    config.model_config.model_kwargs["pretrained"] = True
    config.model_config.model_kwargs["n_layers"] = 1
    config.model_config.model_kwargs["frozen_modules"] = {}
    config.model_config.model_kwargs["frozen_modules"]["names"] = [
        "distilbert.embeddings"
    ]
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.learning_rate = 0.1
    trainer = config.create_trainer()
    trainer.model_util.freeze_modules(module_type=nn.Embedding)
    hook = SampleGradientVJPHook()
    hook.set_vector(torch.ones_like(trainer.model_util.get_parameter_list()).view(-1))
    trainer.append_hook(hook)

    def print_result(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_result
    )
    trainer.train()
