from cyy_torch_algorithm.computation.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import ExecutorHookPoint


def test_CV_sample_gradient():
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    hook = SampleGradientHook()
    hook.set_computed_indices(set(range(10)))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset_result()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "check gradients", print_sample_gradients
    )
    trainer.train()
    hook.release_queue()


def test_huggingface_sample_gradient():
    config = DefaultConfig(
        "IMDB", "hugging_face_sequence_classification_distilbert-base-cased"
    )
    config.hyper_parameter_config.epoch = 2
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.001
    config.hyper_parameter_config.find_learning_rate = False
    config.model_config.model_kwargs = {"n_layers": 1, "max_len": 100}
    config.dc_config.dataset_kwargs = {
        "max_len": 100,
        "tokenizer": {"type": "hugging_face", "name": "distilbert-base-cased"},
    }
    trainer = config.create_trainer()
    hook = SampleGradientHook()
    hook.set_computed_indices(set(range(10)))
    trainer.append_hook(hook)

    def print_sample_gradients(**kwargs):
        if hook.result_dict:
            print(hook.result_dict)
            hook.reset_result()

    trainer.append_named_hook(
        ExecutorHookPoint.AFTER_FORWARD, "check gradients", print_sample_gradients
    )
    trainer.train()
    hook.release_queue()
