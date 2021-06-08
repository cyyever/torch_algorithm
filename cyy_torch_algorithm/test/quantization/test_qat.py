#!/usr/bin/env python3
from cyy_torch_toolbox.default_config import DefaultConfig

from quantization.qat import QuantizationAwareTraining


def test_training():
    trainer = DefaultConfig("MNIST", "LeNet5").create_trainer()
    trainer.hyper_parameter.set_epoch(1)
    trainer.hyper_parameter.set_learning_rate(0.01)
    qat = QuantizationAwareTraining()
    trainer.append_hook(qat)
    trainer.train()
