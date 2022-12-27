from typing import Callable, Tuple

import numpy
import torch
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import assemble_tensors, disassemble_tensor


class StochasticQuant:
    def __init__(self, quantization_level, use_l2_norm):
        self.quantization_level = quantization_level
        self.use_l2_norm = use_l2_norm

    def __call__(self, data):
        tensor, shapes = assemble_tensors(data)
        if tensor is None:
            return data

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
                    return data
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
