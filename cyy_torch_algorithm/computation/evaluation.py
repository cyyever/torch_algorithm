import torch
from cyy_torch_toolbox.ml_type import EvaluationMode


def eval_model(
    parameter_dict, model_evaluator, input_shape=None, **kwargs
) -> torch.Tensor:
    model_evaluator.model_util.load_buffer_dict(parameter_dict)
    kwargs |= {
        "non_blocking": True,
        "evaluation_mode": EvaluationMode.Test,
    }
    if input_shape is not None:
        kwargs["inputs"] = kwargs["inputs"].view(input_shape)
    return model_evaluator(**kwargs)["loss"]
