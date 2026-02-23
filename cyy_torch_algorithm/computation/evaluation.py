from typing import Any

import torch
from cyy_torch_toolbox import EvaluationMode, ModelEvaluator, ModelParameter


def eval_model(
    parameters: ModelParameter,
    model_evaluator: ModelEvaluator,
    **kwargs: Any,
) -> torch.Tensor:
    model_evaluator.model_util.load_buffers(parameters)
    kwargs |= {
        "evaluation_mode": EvaluationMode.SampleInference,
    }
    if "input_tensors" in kwargs:
        input_tensors = kwargs.pop("input_tensors")
        input_keys = kwargs.pop("input_keys", None)
        if input_keys:
            kwargs["inputs"] = dict(zip(input_keys, input_tensors, strict=True))
        else:
            kwargs["inputs"] = input_tensors[0]

    return model_evaluator(**kwargs)["loss"]
