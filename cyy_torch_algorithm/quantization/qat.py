import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.model_with_loss import ModelWithLoss
from cyy_torch_toolbox.trainer import Trainer
from torch.ao.quantization.fuser_method_mappings import \
    DEFAULT_OP_LIST_TO_FUSER_METHOD


class QuantizationAwareTraining(Hook):
    """
    Quantization-aware training
    """

    def __init__(self):
        super().__init__()
        self.__trainer = None

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        self.__prepare_quantization(trainer)

    def __prepare_quantization(self, trainer: Trainer) -> None:
        self.__trainer = trainer
        model_util = self.__trainer.model_util

        if model_util.have_module(module_type=torch.ao.quantization.QuantStub):
            return
        model_util.model.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        quant_model = torch.ao.quantization.QuantWrapper(model_util.model)
        quant_model.eval()

        modules = QuantizationAwareTraining.get_fused_modules(quant_model)
        quant_model = torch.ao.quantization.fuse_modules_qat(
            quant_model,
            modules,
        )
        get_logger().info("fuse modules %s", modules)
        quant_model.train()
        quant_model = torch.ao.quantization.prepare_qat(quant_model)
        get_logger().debug("quant_model is %s", quant_model)
        trainer.set_model_with_loss(
            trainer.model_with_loss.replace_model(model=quant_model)
        )
        trainer.remove_optimizer()

    def get_quantized_model(self, model=None) -> torch.nn.Module:
        if model is None:
            model = self.__trainer.model
        model.cpu()
        model.eval()
        return torch.ao.quantization.convert(model)

    # def load_quantized_parameters(self, parameter_dict: dict) -> dict:
    #     model_util = ModelUtil(self.__original_model)
    #     processed_modules = set()
    #     state_dict = self.quantized_model.state_dict()
    #     quantized_model_util = ModelUtil(self.quantized_model)
    #     for name, module in self.__original_model.named_modules():
    #         if isinstance(module, torch.nn.modules.BatchNorm2d):
    #             get_logger().debug("ignore BatchNorm2d %s", name)
    #             torch.nn.init.ones_(module.weight)
    #             torch.nn.init.zeros_(module.bias)
    #             torch.nn.init.zeros_(module.running_mean)
    #             torch.nn.init.ones_(module.running_var)
    #             # module.eps = 0

    #     for k in state_dict:
    #         if k in ("scale", "zero_point", "quant.scale", "quant.zero_point"):
    #             continue
    #         if "." not in k:
    #             continue
    #         module_name = ".".join(k.split(".")[:-1])
    #         if module_name in processed_modules:
    #             continue
    #         if not quantized_model_util.has_attr(module_name):
    #             continue
    #         sub_module = quantized_model_util.get_attr(module_name)
    #         if module_name.startswith("module."):
    #             module_name = module_name[len("module."):]
    #         if isinstance(
    #             sub_module,
    #             (
    #                 torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
    #                 torch.nn.quantized.modules.linear.Linear,
    #                 torch.nn.quantized.modules.conv.Conv2d,
    #                 torch.nn.quantized.modules.batchnorm.BatchNorm2d,
    #             ),
    #         ):
    #             processed_modules.add(module_name)
    #             weight = parameter_dict[module_name + ".weight"]
    #             if isinstance(weight, tuple):
    #                 (weight, scale, zero_point) = weight
    #                 weight = weight.float()
    #                 for idx, v in enumerate(weight):
    #                     weight[idx] = (v - zero_point[idx]) * scale[idx]
    #             model_util.set_attr(module_name + ".weight", weight)

    #             for suffix in [".bias", ".running_mean", ".running_var"]:
    #                 attr_name = module_name + suffix
    #                 if attr_name in parameter_dict:
    #                     model_util.set_attr(attr_name, parameter_dict[attr_name])
    #             continue
    #         if not isinstance(
    #             sub_module, torch.nn.quantized.modules.linear.LinearPackedParams
    #         ):
    #             get_logger().warning("unsupported sub_module type %s", type(sub_module))
    #     return parameter_dict

    @staticmethod
    def get_fused_modules(quantized_model) -> list:
        quantized_model_util = ModelUtil(quantized_model)
        module_blocks = quantized_model_util.get_module_blocks(
            block_types=set(DEFAULT_OP_LIST_TO_FUSER_METHOD.keys())
        )
        return [[module[0] for module in block] for block in module_blocks]
