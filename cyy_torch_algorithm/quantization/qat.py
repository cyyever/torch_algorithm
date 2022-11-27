import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.trainer import Trainer
from torch.ao.quantization.fuser_method_mappings import \
    _DEFAULT_OP_LIST_TO_FUSER_METHOD


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

    @staticmethod
    def get_fused_modules(quantized_model) -> list:
        quantized_model_util = ModelUtil(quantized_model)
        module_blocks = quantized_model_util.get_module_blocks(
            block_types=set(_DEFAULT_OP_LIST_TO_FUSER_METHOD.keys())
        )
        return [[module[0] for module in block] for block in module_blocks]
