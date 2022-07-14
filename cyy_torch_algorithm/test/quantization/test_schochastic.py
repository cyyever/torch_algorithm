#!/usr/bin/env python3
import pickle

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.quantization.stochastic import (
    ISQ, NNISQ, NeuralNetworkImprovedStochasticQuant, stochastic_quantization)
from cyy_torch_toolbox.default_config import DefaultConfig
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector
from transformers import BertModel


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


def test_model_stochastic_quantization():
    model = BertModel.from_pretrained("bert-base-cased")
    model_util = ModelUtil(model)
    model_dict = model_util.get_parameter_dict()
    model_vect = cat_tensors_to_vector(get_mapping_values_by_key_order(model_dict))
    print("parameter size is", model_vect.shape)

    quant, dequant = stochastic_quantization(quantization_level=255, use_l2_norm=True)
    quantization_result = quant(model_vect)
    quantized_model_vect = dequant(quantization_result)
    print("old compression ratio", 1 / 4)
    print(
        "old relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )
    quant, dequant = stochastic_quantization(quantization_level=255, use_l2_norm=False)
    quantization_result = quant(model_vect)
    quantized_model_vect = dequant(quantization_result)
    print("old infinity norm compression ratio", 1 / 4)
    print(
        "old infinity norm relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )
    weight = 0.01

    quant, dequant = ISQ(weight=weight, use_l2_norm=True, normalization=False)
    quantization_result = quant(model_vect)
    quantized_model_vect = dequant(quantization_result)
    print("use quantization_level ", quantization_result["quantization_level"])
    print("ISQ L2 norm compression ratio", quantization_result["compression_ratio"])
    print(
        "ISQ L2 norm relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )

    quant, dequant = ISQ(weight=weight, normalization=False)
    quantization_result = quant(model_vect)
    quantized_model_vect = dequant(quantization_result)
    print("use quantization_level ", quantization_result["quantization_level"])
    print(
        "infinity norm diff compression ratio", quantization_result["compression_ratio"]
    )
    print(
        "infinity norm diff relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )

    quant, dequant = ISQ(weight=weight, normalization=True)
    quantization_result = quant(model_vect)
    quantized_model_vect = dequant(quantization_result)
    print("use quantization_level ", quantization_result["quantization_level"])
    print(
        "infinity norm diff and normalization compression ratio",
        quantization_result["compression_ratio"],
    )
    print(
        "infinity norm diff and normalization relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )

    quant, dequant = NNISQ(weight=weight)
    quantization_result = quant(model_dict)
    NeuralNetworkImprovedStochasticQuant.check_compression_ratio(quantization_result)
    quantized_model = dequant(quantization_result)
    quantized_model_vect = cat_tensors_to_vector(
        get_mapping_values_by_key_order(quantized_model)
    )

    print(
        "layer by layer relative diff",
        torch.linalg.norm(quantized_model_vect - model_vect)
        / torch.linalg.norm(model_vect),
    )


def test_zero_tensor():
    tensor = torch.zeros(5)
    quant, dequant = ISQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
    tensor = torch.zeros(5, dtype=torch.int)
    quant, dequant = ISQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
    tensor = torch.zeros(5, dtype=torch.uint8)
    quant, dequant = ISQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
