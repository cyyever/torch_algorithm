import importlib.util

has_cyy_torch_text: bool = importlib.util.find_spec("cyy_torch_text") is not None
has_cyy_torch_vision: bool = importlib.util.find_spec("cyy_torch_vision") is not None

from cyy_torch_algorithm.computation.sample_gradient import SampleGradientHook
from cyy_torch_toolbox import Config, ExecutorHookPoint, StopExecutingException


def test_CV_sample_gradient():
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


def test_huggingface_sample_gradient():
    if not has_cyy_torch_text:
        return
    import cyy_torch_text  # noqa: F401

    config = Config(
        "imdb", "hugging_face_sequence_classification_distilbert-base-cased"
    )
    config.trainer_config.hook_config.use_amp = False
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.001
    config.model_config.model_kwargs = {"n_layers": 1, "input_max_len": 100}
    config.dc_config.dataset_kwargs = {
        "input_max_len": 100,
    }
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
