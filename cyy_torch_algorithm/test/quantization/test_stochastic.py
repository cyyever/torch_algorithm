import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_algorithm.quantization.stochastic import stochastic_quantization
from cyy_torch_toolbox.model.util import ModelUtil
from cyy_torch_toolbox.tensor import cat_tensors_to_vector

try:
    from transformers import BertModel

    def test_model_stochastic_quantization():
        model = BertModel.from_pretrained("bert-large-cased")
        model_util = ModelUtil(model)
        model_dict = model_util.get_parameter_dict()
        model_vect = cat_tensors_to_vector(get_mapping_values_by_key_order(model_dict))

        quant, dequant = stochastic_quantization(quantization_level=255)
        quantization_result = quant(model_dict)
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
