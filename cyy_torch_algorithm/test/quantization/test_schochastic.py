#!/usr/bin/env python3
import pickle

import torch
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization
from cyy_torch_toolbox.default_config import DefaultConfig


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
    print("compression ration", len(pickle.dumps(pair)) / len(pickle.dumps(a)))
    print(
        "relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )

    quant, dequant = stochastic_quantization(256, use_l2_norm=True)
    pair = quant(a)
    recovered_tensor = dequant(pair)
    print(
        "l2 relative diff",
        torch.linalg.norm(recovered_tensor - a) / torch.linalg.norm(a),
    )
