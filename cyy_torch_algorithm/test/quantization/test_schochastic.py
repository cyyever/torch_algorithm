#!/usr/bin/env python3
import torch
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization
from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


def test_stochastic_quantization():
    config = DefaultConfig("MNIST", "lenet5")
    config.hyper_parameter_config.epoch = 1
    config.hyper_parameter_config.batch_size = 8
    config.hyper_parameter_config.learning_rate = 0.01
    config.hyper_parameter_config.find_learning_rate = False
    trainer = config.create_trainer()
    trainer.train()
    a = trainer.model_util.get_parameter_list()
    quant, dequant = stochastic_quantization(256)
    pair = quant(a)
    recovered_tensor = dequant(pair)
    print(
        "recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )

    b = torch.rand(2, 100000)
    data = {"key1": a, "key2": b}
    pair = quant(data)
    recovered_data = dequant(pair)
    print(
        "use_l2_norm relative diff",
        torch.linalg.norm(recovered_data["key1"] - a) / torch.linalg.norm(a),
        torch.linalg.norm(recovered_data["key2"] - b) / torch.linalg.norm(b),
    )

    quant, dequant = stochastic_quantization(256, use_l2_norm=True)
    pair = quant(a)
    recovered_tensor = dequant(pair)
    print(
        "use_l2_norm recovered_tensor",
        recovered_tensor,
        "tensor",
        a,
        "relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )
