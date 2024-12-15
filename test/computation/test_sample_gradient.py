import importlib.util

from cyy_torch_algorithm import SampleGradientHook
from cyy_torch_toolbox import Config, ExecutorHookPoint, StopExecutingException

has_cyy_huggingface_toolbox: bool = (
    importlib.util.find_spec("cyy_huggingface_toolbox") is not None
)
has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None


def test_CV_sample_gradient() -> None:
    if not has_cyy_torch_vision:
        return
    import cyy_torch_vision  # noqa: F401

    config = Config("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    trainer = config.create_trainer()
    hook = SampleGradientHook()
    hook.set_computed_indices(set(range(10)))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.result_dict:
            print(next(iter(hook.result_dict.values())))
            hook.reset_result()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()
    hook.reset()


def test_huggingface_sample_gradient() -> None:
    if not has_cyy_huggingface_toolbox:
        return
    import cyy_huggingface_toolbox  # noqa: F401

    config = Config(
        "imdb", "hugging_face_sequence_classification_distilbert-base-cased"
    )
    config.trainer_config.hook_config.use_amp = False
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.001
    config.model_config.model_kwargs = {"n_layers": 1, "attn_implementation": "eager"}
    trainer = config.create_trainer()
    hook = SampleGradientHook()
    hook.set_computed_indices(set(range(10)))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.result_dict:
            print(next(iter(hook.result_dict.values())))
            hook.reset_result()
            raise StopExecutingException()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_BATCH, "check gradients", print_sample_gradients
    )
    trainer.train()
    hook.reset()
