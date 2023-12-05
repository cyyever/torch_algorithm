from cyy_torch_algorithm.quantization.qat import QuantizationAwareTraining
from cyy_torch_toolbox.default_config import Config


def test_training() -> None:
    trainer = Config("MNIST", "Lenet5").create_trainer()
    trainer.hyper_parameter.epoch = 1
    trainer.hyper_parameter.learning_rate = 0.01
    trainer.hook_config.use_amp = False
    qat = QuantizationAwareTraining()
    trainer.append_hook(qat)
    trainer.train()
