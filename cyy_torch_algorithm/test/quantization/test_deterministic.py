import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.quantization.deterministic import (
    ADQ, NNADQ, NeuralNetworkAdaptiveDeterministicQuant)
from cyy_torch_toolbox.model.util import ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

try:
    from transformers import BertModel

    def test_model_quantization():
        model = BertModel.from_pretrained("bert-large-cased")
        model_util = ModelUtil(model)
        model_dict = model_util.get_parameter_dict()
        model_vect = cat_tensors_to_vector(get_mapping_values_by_key_order(model_dict))
        print("parameter size is", model_vect.shape)

        weight = 0.01
        quant, dequant = NNADQ(weight=weight)
        quantization_result = quant(model_dict)
        NeuralNetworkAdaptiveDeterministicQuant.check_compression_ratio(
            quantization_result
        )
        quantized_model = dequant(quantization_result)
        quantized_model_vect = cat_tensors_to_vector(
            get_mapping_values_by_key_order(quantized_model)
        )

        print(
            "relative diff",
            torch.linalg.norm(quantized_model_vect - model_vect)
            / torch.linalg.norm(model_vect),
        )

except BaseException:
    pass


def test_zero_tensor():
    tensor = torch.zeros(5)
    quant, dequant = ADQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
    tensor = torch.zeros(5, dtype=torch.int)
    quant, dequant = ADQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
    tensor = torch.zeros(5, dtype=torch.uint8)
    quant, dequant = ADQ(weight=0.1)
    res = quant(tensor)
    tensor2 = dequant(res)
    assert torch.all(tensor == tensor2)
