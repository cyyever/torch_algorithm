import torch
from cyy_torch_toolbox import EvaluationMode, ModelEvaluator
from cyy_torch_toolbox.typing import TensorDict


def eval_model(
    parameter_dict: TensorDict,
    model_evaluator: ModelEvaluator,
    input_shape: None | torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    model_evaluator.model_util.load_buffer_dict(parameter_dict)
    kwargs |= {
        "evaluation_mode": EvaluationMode.Test,
    }
    if input_shape is not None:
        kwargs["inputs"] = kwargs["inputs"].view(input_shape)
    return model_evaluator(**kwargs)["loss"]
