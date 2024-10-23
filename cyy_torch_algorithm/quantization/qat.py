import torch
import torch.ao.quantization
from cyy_naive_lib.log import log_debug
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.model.util import ModelUtil
from cyy_torch_toolbox.trainer import Trainer
from torch.ao.quantization.fuser_method_mappings import _DEFAULT_OP_LIST_TO_FUSER_METHOD


class QuantizationAwareTraining(Hook):
    """
    Quantization-aware training
    """

    def _before_execute(self, **kwargs):
        trainer = kwargs["executor"]
        self.__prepare_quantization(trainer)

    def __prepare_quantization(self, trainer: Trainer) -> None:
        model_util = trainer.model_util

        if model_util.have_module(module_type=torch.ao.quantization.QuantStub):
            return
        model_util.model.eval()
        model_util.model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.backends.quantized.engine = "x86"
        fused_modules = QuantizationAwareTraining.get_fused_modules(model_util.model)
        log_debug("fuse modules %s", fused_modules)

        fused_model = torch.ao.quantization.fuse_modules_qat(
            model_util.model,
            fused_modules,
        )
        fused_model.train()
        quant_model = torch.ao.quantization.prepare_qat(fused_model)
        quant_model = torch.ao.quantization.QuantWrapper(quant_model)
        log_debug("quant_model is %s", quant_model)
        ModelUtil(quant_model).to_device(device=trainer.device)
        trainer.replace_model(lambda model: quant_model)
        trainer.remove_optimizer()

    @classmethod
    def get_quantized_model(cls, model: torch.nn.Module) -> torch.nn.Module:
        model.cpu()
        model.eval()
        return torch.ao.quantization.convert(model)

    @staticmethod
    def get_fused_modules(model: torch.nn.Module) -> list:
        module_blocks = ModelUtil(model).get_module_blocks(
            block_types=set(_DEFAULT_OP_LIST_TO_FUSER_METHOD.keys())
        )
        return [[module[0] for module in block] for block in module_blocks]
