from typing import Callable, Tuple

import numpy
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import (cat_tensors_to_vector,
                                      split_tensor_to_dict)


class StochasticQuant:
    def __init__(self, quantization_level, use_l2_norm):
        self.quantization_level = quantization_level
        self.use_l2_norm = use_l2_norm

    def __call__(self, tensor):
        name_and_shapes = None
        if isinstance(tensor, dict):
            name_and_shapes = []
            for k in sorted(tensor.keys()):
                name_and_shapes.append((k, tensor[k].shape))
            tensor = cat_tensors_to_vector(get_mapping_values_by_key_order(tensor))

        old_tensor_shape = tensor.shape
        tensor = tensor.reshape(-1)
        assert len(tensor.shape) == 1

        norm = None
        if self.use_l2_norm:
            norm = torch.linalg.norm(tensor)
        else:
            norm = torch.linalg.norm(tensor, ord=float("inf"))
        assert norm > 0
        sign_tensor = torch.sign(tensor)
        normalized_abs_tensor = tensor.abs() / norm
        tmp = normalized_abs_tensor * self.quantization_level
        slot_tensor = tmp.trunc()
        prob_tensor = tmp - slot_tensor
        random_vector = torch.distributions.Bernoulli(prob_tensor).sample()
        slot_tensor += random_vector

        sign_tensor = numpy.packbits(
            ((sign_tensor + 1) / 2).to(torch.bool).to(get_cpu_device()).numpy()
        )
        if self.quantization_level <= 256:
            slot_tensor = slot_tensor.to(torch.uint8)
        # slot_tensor = slot_tensor.to(old_device).reshape(old_tensor_shape)
        slot_tensor = slot_tensor.reshape(old_tensor_shape)

        return (
            norm,
            sign_tensor,
            slot_tensor,
            self.quantization_level,
            name_and_shapes,
        )


class StochasticDequant:
    def __call__(self, quantized_pair):
        (
            norm,
            sign_tensor,
            quantized_tensor,
            quantization_level,
            name_and_shapes,
        ) = quantized_pair

        quantized_tensor = quantized_tensor.float()
        quantized_tensor *= norm
        sign_tensor = (
            torch.from_numpy(numpy.unpackbits(sign_tensor)).float() * 2 - 1
        ).reshape(quantized_tensor.shape)
        res = quantized_tensor * sign_tensor / quantization_level

        if name_and_shapes is not None:
            res = split_tensor_to_dict(name_and_shapes, res)
        return res


def stochastic_quantization(
    quantization_level: int, use_l2_norm: bool = False
) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""
    return StochasticQuant(quantization_level, use_l2_norm), StochasticDequant()
