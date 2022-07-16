import math
from typing import Any, Callable, Tuple

import numpy
import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.device import get_cpu_device


class AdaptiveDeterministicQuant:
    def __init__(
        self, weight: float, use_l2_norm: bool = False, normalization: bool = True
    ):
        self.weight = weight
        self.use_l2_norm = use_l2_norm
        self.normalization = normalization

    def __call__(self, tensor):
        element_bits = None
        dtype = tensor.dtype
        element_bits = tensor.element_size() * 8
        old_tensor_shape = tensor.shape
        tensor = tensor.to(dtype=torch.float64)
        tensor = tensor.view(-1)
        if self.normalization:
            mean = torch.mean(tensor)
            tensor = tensor - mean
        else:
            mean = None

        norm = None
        if self.use_l2_norm:
            norm = torch.linalg.norm(tensor)
        else:
            norm = torch.linalg.norm(tensor, ord=float("inf"))
        if norm == 0.0:
            return {
                "dtype": dtype,
                "mean": mean,
                "tensor_shape": old_tensor_shape,
                "compression_ratio": 0,
                "quantization_level": 0,
            }

        quantization_level = int(
            max(1, math.sqrt(norm * element_bits * math.log(4) / self.weight))
        )
        compression_ratio = math.ceil(math.log2(quantization_level + 1)) / element_bits
        sign_tensor = torch.sign(tensor)
        normalized_abs_tensor = tensor.abs() / norm
        tmp = normalized_abs_tensor * quantization_level
        slot_tensor = tmp.round()

        sign_tensor = numpy.packbits(
            ((sign_tensor + 1) / 2).to(torch.bool).to(get_cpu_device()).numpy()
        )
        if quantization_level < 2**8:
            new_dtype = numpy.uint8
        elif quantization_level < 2**16:
            new_dtype = numpy.uint16
        elif quantization_level < 2**32:
            new_dtype = numpy.uint32
        else:
            raise RuntimeError(f"invalid quantization level {quantization_level}")
        slot_tensor = (
            slot_tensor.reshape(old_tensor_shape).numpy().astype(dtype=new_dtype)
        )
        return {
            "dtype": dtype,
            "norm": norm,
            "sign_tensor": sign_tensor,
            "quantized_tensor": slot_tensor,
            "quantization_level": quantization_level,
            "mean": mean,
            "compression_ratio": compression_ratio,
        }


class AdaptiveDeterministicDequant:
    def __call__(self, quantized_dict: dict) -> torch.Tensor:
        dtype = quantized_dict["dtype"]
        mean = quantized_dict["mean"]
        if "tensor_shape" in quantized_dict:
            quantized_tensor = torch.zeros(
                quantized_dict["tensor_shape"], dtype=torch.float64
            )
            if mean is not None:
                quantized_tensor += mean
            return quantized_tensor.to(dtype=dtype)

        quantized_tensor = torch.from_numpy(
            quantized_dict["quantized_tensor"].astype(dtype=numpy.int64)
        ).to(dtype=torch.float64)
        sign_tensor = quantized_dict["sign_tensor"]
        quantization_level = quantized_dict["quantization_level"]
        norm = quantized_dict["norm"]
        quantized_tensor *= norm
        sign_tensor = (torch.from_numpy(numpy.unpackbits(sign_tensor)).float() * 2 - 1)[
            : numpy.prod(quantized_tensor.shape)
        ].reshape(quantized_tensor.shape)
        res = (
            quantized_tensor
            * sign_tensor.to(quantized_tensor.device)
            / quantization_level
        )
        if mean is not None:
            res = res + mean
        return res.to(dtype=dtype)


class NeuralNetworkAdaptiveDeterministicQuant(AdaptiveDeterministicQuant):
    def __call__(self, data: Any) -> Any:
        match data:
            case dict():
                return {k: self.__call__(v) for k, v in data.items()}
            case torch.Tensor():
                return super().__call__(data)
            case _:
                return data

    @classmethod
    def check_compression_ratio(cls, quantized_data, prefix=None):
        compressed_parameter_num = 0
        quantization_levels = []
        parameter_numbers = []

        def collection(quantized_data):
            nonlocal quantization_levels, compressed_parameter_num
            if not isinstance(quantized_data, dict):
                return
            if "tensor_shape" in quantized_data:
                parameter_num = numpy.prod(quantized_data["tensor_shape"])
                compressed_parameter_num += (
                    parameter_num * quantized_data["compression_ratio"]
                )
                parameter_numbers.append(parameter_num)
                quantization_levels.append(0)
            elif "quantization_level" in quantized_data:
                parameter_num = numpy.prod(quantized_data["sign_tensor"].shape)
                compressed_parameter_num += (
                    parameter_num * quantized_data["compression_ratio"]
                )
                parameter_numbers.append(parameter_num)
                quantization_levels.append(quantized_data["quantization_level"])
            else:
                for v in quantized_data.values():
                    collection(v)

        collection(quantized_data)
        if not parameter_numbers:
            return 0, 0

        total_parameter_num = sum(parameter_numbers)
        parameter_ratio = [a / total_parameter_num for a in parameter_numbers]
        if prefix is None:
            prefix = ""
        avg_level = sum([a * b for a, b in zip(quantization_levels, parameter_ratio)])
        compression_ratio = compressed_parameter_num / total_parameter_num
        get_logger().info("%s avg quantization level %s", prefix, avg_level)
        get_logger().info("%s NNABQ compression ratio is %s", prefix, compression_ratio)
        return avg_level, compression_ratio


class NeuralNetworkAdaptiveDeterministicDequant(AdaptiveDeterministicDequant):
    def __call__(self, data: Any) -> Any:
        match data:
            case dict():
                if "dtype" in data:
                    return super().__call__(data)
                return {k: self.__call__(v) for k, v in data.items()}
            case _:
                return data


def NNADQ(
    weight: float = None,
    use_l2_norm: bool = False,
    normalization: bool = True,
) -> Tuple[Callable, Callable]:
    return (
        NeuralNetworkAdaptiveDeterministicQuant(
            weight=weight,
            use_l2_norm=use_l2_norm,
            normalization=normalization,
        ),
        NeuralNetworkAdaptiveDeterministicDequant(),
    )


def ADQ(
    weight: float,
    use_l2_norm: bool = False,
    normalization: bool = True,
) -> Tuple[Callable, Callable]:
    return (
        AdaptiveDeterministicQuant(
            weight=weight,
            use_l2_norm=use_l2_norm,
            normalization=normalization,
        ),
        AdaptiveDeterministicDequant(),
    )
