import torch
import torch.ao.quantization
from cyy_naive_lib.log import log_debug, log_info
from cyy_torch_toolbox import Hook, ModelUtil, Trainer
from torch.ao.quantization.fuser_method_mappings import _DEFAULT_OP_LIST_TO_FUSER_METHOD


class QuantizationAwareTraining(Hook):
    """
    Quantization-aware training
    """

    def _before_execute(self, **kwargs):
        trainer = kwargs["executor"]
        if isinstance(trainer, Trainer):
            self.prepare_quantization(trainer)

    @classmethod
    def prepare_quantization(cls, trainer: Trainer) -> None:
        model_util = trainer.model_util

        if model_util.have_module(module_type=torch.ao.quantization.QuantStub):
            return
        # model must be set to eval for fusion to work
        model_util.model.eval()
        model_util.model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
        torch.backends.quantized.engine = "x86"
        fused_modules = cls.get_fused_modules(model_util)
        log_info("fuse modules %s", fused_modules)

        fused_model = torch.ao.quantization.fuse_modules_qat(
            model_util.model,
            fused_modules,
        )
        fused_model.train()
        quant_model = torch.ao.quantization.prepare_qat(fused_model)
        quant_model = torch.ao.quantization.QuantWrapper(quant_model)
        log_debug("quant_model is %s", quant_model)
        trainer.replace_model(lambda _: quant_model)

    @classmethod
    def get_quantized_model_for_inference(
        cls, model: torch.nn.Module
    ) -> torch.nn.Module:
        model.cpu()
        model.eval()
        return torch.ao.quantization.convert(model)

    @classmethod
    def get_fused_modules(cls, model_util: ModelUtil) -> list:
        module_blocks = model_util.get_module_blocks(
            block_types=set(_DEFAULT_OP_LIST_TO_FUSER_METHOD.keys())
        )
        return [[module[0] for module in block] for block in module_blocks]
