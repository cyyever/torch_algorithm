#!/usr/bin/env python3
import torch
from cyy_torch_toolbox.default_config import Config

from cyy_torch_algorithm.quantization.qat import QuantizationAwareTraining


def test_training():
    return
    trainer = Config("CIFAR10", "densenet40").create_trainer()
    trainer.hyper_parameter.set_epoch(1)
    trainer.hyper_parameter.set_learning_rate(0.01)
    qat = QuantizationAwareTraining()
    trainer.append_hook(qat)
    trainer.train()
