import math
from typing import Any, Callable, Tuple

import numpy
import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger
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
        sign_tensor = (torch.from_numpy(numpy.unpackbits(sign_tensor)).float() * 2 - 1)[
            : numpy.prod(quantized_tensor.shape)
        ].reshape(quantized_tensor.shape)
        res = (
            quantized_tensor
            * sign_tensor.to(quantized_tensor.device)
            / quantization_level
        )

        if name_and_shapes is not None:
            res = split_tensor_to_dict(name_and_shapes, res)
        return res


def stochastic_quantization(
    quantization_level: int, use_l2_norm: bool = False
) -> Tuple[Callable, Callable]:
    """Implement Stochastic Quantization as described in QSGD: Communication-Efficient SGDvia Gradient Quantization and Encoding (https://arxiv.org/pdf/1610.02132.pdf)"""
    return StochasticQuant(quantization_level, use_l2_norm), StochasticDequant()


class ImprovedStochasticQuant:
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
            }

        quantization_level = int(
            max(1, math.sqrt(norm * element_bits * math.log(4) / self.weight))
        )
        compression_ratio = math.ceil(math.log2(quantization_level + 1)) / element_bits
        sign_tensor = torch.sign(tensor)
        normalized_abs_tensor = tensor.abs() / norm
        tmp = normalized_abs_tensor * quantization_level
        slot_tensor = tmp.trunc()
        prob_tensor = tmp - slot_tensor
        random_vector = torch.distributions.Bernoulli(prob_tensor).sample()
        slot_tensor += random_vector

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


class ImprovedStochasticDequant:
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


class NeuralNetworkImprovedStochasticQuant(ImprovedStochasticQuant):
    def __call__(self, data: Any) -> Any:
        match data:
            case dict():
                return {k: self.__call__(v) for k, v in data.items()}
            case torch.Tensor():
                return super().__call__(data)
            case _:
                return data

    @classmethod
    def check_compression_ratio(cls, quantized_data):
        result = []
        compressed_parameter_num = 0
        total_parameter_num = 0
        quantization_levels = []
        parameter_numbers = []

        def collection(quantized_data):
            nonlocal result, total_parameter_num, quantization_levels, compressed_parameter_num
            if not isinstance(quantized_data, dict):
                return
            if "tensor_shape" in quantized_data:
                parameter_num = numpy.prod(quantized_data["tensor_shape"])
                total_parameter_num += parameter_num
                compressed_parameter_num += (
                    parameter_num * quantized_data["compression_ratio"]
                )
                parameter_numbers.append(parameter_num)
                quantization_levels.append(0)
            elif "quantization_level" in quantized_data:
                parameter_num = numpy.prod(quantized_data["sign_tensor"].shape)
                total_parameter_num += parameter_num
                compressed_parameter_num += (
                    parameter_num * quantized_data["compression_ratio"]
                )
                parameter_numbers.append(parameter_num)
                quantization_levels.append(quantized_data["quantization_level"])
            else:
                for v in quantized_data.values():
                    collection(v)

        collection(quantized_data)
        assert parameter_numbers

        parameter_ratio = [a / total_parameter_num for a in parameter_numbers]
        get_logger().info(
            "avg quantization level %s",
            sum([a * b for a, b in zip(quantization_levels, parameter_ratio)]),
        )
        get_logger().info(
            "NN compression ratio is %s", compressed_parameter_num / total_parameter_num
        )


class NeuralNetworkImprovedStochasticDequant(ImprovedStochasticDequant):
    def __call__(self, data: Any) -> Any:
        match data:
            case dict():
                if "dtype" in data:
                    return super().__call__(data)
                return {k: self.__call__(v) for k, v in data.items()}
            case _:
                return data


def NNISQ(
    weight: float = None,
    use_l2_norm: bool = False,
    normalization: bool = True,
) -> Tuple[Callable, Callable]:
    return (
        NeuralNetworkImprovedStochasticQuant(
            weight=weight,
            use_l2_norm=use_l2_norm,
            normalization=normalization,
        ),
        NeuralNetworkImprovedStochasticDequant(),
    )


def ISQ(
    weight: float,
    use_l2_norm: bool = False,
    normalization: bool = True,
) -> Tuple[Callable, Callable]:
    return (
        ImprovedStochasticQuant(
            weight=weight,
            use_l2_norm=use_l2_norm,
            normalization=normalization,
        ),
        ImprovedStochasticDequant(),
    )
