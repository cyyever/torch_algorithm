from typing import Callable, Tuple

import numpy
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import (assemble_tensors, cat_tensors_to_vector,
                                      disassemble_tensor, split_tensor_to_dict)


class StochasticQuant:
    def __init__(self, quantization_level, use_l2_norm):
        self.quantization_level = quantization_level
        self.use_l2_norm = use_l2_norm

    def __call__(self, data):
        if not data:
            return data
        match data:
            case tuple():
                if not isinstance(data[0], torch.Tensor):
                    return tuple(self.__call__(v) for v in data)
            case list():
                if not isinstance(data[0], torch.Tensor):
                    return [self.__call__(v) for v in data]
            case dict():
                if not isinstance(next(iter(data.values())), torch.Tensor):
                    return {k: self.__call__(v) for k, v in data.items()}
        tensor, shapes = assemble_tensors(data)

        old_tensor_shape = tensor.shape
        tensor = tensor.reshape(-1)

        norm = None
        if self.use_l2_norm:
            norm = torch.linalg.norm(tensor)
        else:
            norm = torch.linalg.norm(tensor, ord=float("inf"))
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
        slot_tensor = slot_tensor.reshape(old_tensor_shape)

        return {
            "norm": norm,
            "sign": sign_tensor,
            "slot": slot_tensor,
            "quantization_level": self.quantization_level,
            "name_and_shapes": shapes,
        }


class StochasticDequant:
    def __call__(self, data):
        match data:
            case dict():
                if "quantization_level" not in data:
                    return {k: self.__call__(v) for k, v in data.items()}
                norm = data["norm"]
                sign_tensor = data["sign"]
                quantized_tensor = data["slot"]
                quantization_level = data["quantization_level"]
                name_and_shapes = data["name_and_shapes"]
            case _:
                return data

        quantized_tensor = quantized_tensor.float()
        quantized_tensor *= norm
        sign_tensor = (torch.from_numpy(numpy.unpackbits(sign_tensor)).float() * 2 - 1)[
            : numpy.prod(quantized_tensor.shape)
        ].reshape(quantized_tensor.shape)
        res = (
            quantized_tensor
            * sign_tensor.to(quantized_tensor.device)
            / quantization_level
        )

        return disassemble_tensor(res, name_and_shapes)


def stochastic_quantization(
    quantization_level: int, use_l2_norm: bool = False
) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""
    return StochasticQuant(quantization_level, use_l2_norm), StochasticDequant()
