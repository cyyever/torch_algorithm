import torch
from cyy_torch_toolbox.ml_type import EvaluationMode


def eval_model(
    parameter_dict,
    model_evaluator,
    targets,
    device=None,
    inputs=None,
    evaluation_mode=EvaluationMode.Training,
    is_input_feature=False,
    input_shape=None,
    **input_kwargs
) -> torch.Tensor:
    model_evaluator.model_util.load_buffer_dict(parameter_dict)
    kwargs = {
        "targets": targets,
        "device": device,
        "non_blocking": True,
        "is_input_feature": is_input_feature,
        "evaluation_mode": evaluation_mode,
    }
    if input_kwargs:
        kwargs["inputs"] = input_kwargs
    else:
        kwargs["inputs"] = inputs
    if input_shape is not None:
        kwargs["inputs"] = kwargs["inputs"].view(input_shape)
    return model_evaluator(**kwargs)["loss"]
