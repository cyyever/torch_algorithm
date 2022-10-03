import math
from typing import Any, Callable, Tuple

import numpy
import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.tensor import tensor_to


class AdaptiveDeterministicQuant:
    def __init__(self, weight: float, use_l2_norm: bool = False):
        self.weight = weight
        self.use_l2_norm = use_l2_norm

    def __get_offset(self, tensor):
        max_value = tensor.max().item()
        min_value = tensor.min().item()
        if min_value >= 0:
            return -(min_value + max_value) / 2
        if max_value <= 0:
            return (min_value + max_value) / 2
        if max_value >= -min_value:
            return -(max_value - math.fabs(min_value)) / 2
        return (math.fabs(min_value) - max_value) / 2

    @torch.no_grad()
    def __call__(self, tensor):
        device = tensor.device
        element_bits = None
        dtype = tensor.dtype
        element_bits = tensor.element_size() * 8
        old_tensor_shape = tensor.shape
        # if torch.cuda.is_available():
        #     tensor = tensor.to(dtype=torch.float64, device="cuda:0", non_blocking=True)
        # else:
        tensor = tensor_to(tensor, dtype=torch.float64, non_blocking=True)
        tensor = tensor.view(-1)
        offset = self.__get_offset(tensor)
        tensor = tensor + offset

        norm = None
        if self.use_l2_norm:
            norm = torch.linalg.norm(tensor).item()
        else:
            norm = torch.linalg.norm(tensor, ord=float("inf")).item()
        if norm == 0.0:
            return {
                "device": device,
                "dtype": dtype,
                "offset": offset,
                "tensor_shape": old_tensor_shape,
                "compression_ratio": 0,
                "quantization_level": 0,
            }
        sign_tensor = tensor.sign()
        sign_tensor = numpy.packbits(
            ((sign_tensor + 1) / 2).to(dtype=torch.bool, device="cpu").numpy()
        )

        normalized_abs_tensor = tensor.abs() / norm

        quantization_level = int(
            max(1, math.sqrt(norm * element_bits * math.log(4) / self.weight))
        )
        quantized_tensor = (normalized_abs_tensor * quantization_level).round()
        compression_ratio = math.ceil(math.log2(quantization_level + 1)) / element_bits
        if quantization_level < 2**8:
            new_dtype = numpy.uint8
        elif quantization_level < 2**16:
            new_dtype = numpy.uint16
        elif quantization_level < 2**32:
            new_dtype = numpy.uint32
        else:
            raise RuntimeError(f"invalid quantization level {quantization_level}")
        quantized_tensor = (
            quantized_tensor.reshape(old_tensor_shape)
            .cpu()
            .numpy()
            .astype(dtype=new_dtype)
        )
        return {
            "device": device,
            "dtype": dtype,
            "norm": norm,
            "sign_tensor": sign_tensor,
            "quantized_tensor": quantized_tensor,
            "quantization_level": quantization_level,
            "offset": offset,
            "compression_ratio": compression_ratio,
        }


class AdaptiveDeterministicDequant:
    def __call__(self, quantized_dict: dict) -> torch.Tensor:
        device = quantized_dict["device"]
        dtype = quantized_dict["dtype"]
        offset = quantized_dict["offset"]
        if "tensor_shape" in quantized_dict:
            quantized_tensor = torch.zeros(
                quantized_dict["tensor_shape"], dtype=torch.float64, device=device
            )
            quantized_tensor -= offset
            return quantized_tensor.to(dtype=dtype)

        sign_tensor = quantized_dict["sign_tensor"]
        quantization_level = quantized_dict["quantization_level"]
        norm = quantized_dict["norm"]
        quantized_tensor = torch.from_numpy(
            quantized_dict["quantized_tensor"].astype(dtype=numpy.int64)
        ).to(dtype=torch.float64, device=device)
        sign_tensor = (torch.from_numpy(numpy.unpackbits(sign_tensor)).float() * 2 - 1)[
            : numpy.prod(quantized_tensor.shape)
        ].reshape(quantized_tensor.shape)
        res = (
            quantized_tensor * norm * sign_tensor.to(device=device) / quantization_level
        )
        res = res - offset
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


def NNADQ(weight: float = None, use_l2_norm: bool = False) -> Tuple[Callable, Callable]:
    return (
        NeuralNetworkAdaptiveDeterministicQuant(
            weight=weight,
            use_l2_norm=use_l2_norm,
        ),
        NeuralNetworkAdaptiveDeterministicDequant(),
    )


def ADQ(weight: float, use_l2_norm: bool = False) -> Tuple[Callable, Callable]:
    return (
        AdaptiveDeterministicQuant(weight=weight, use_l2_norm=use_l2_norm),
        AdaptiveDeterministicDequant(),
    )
