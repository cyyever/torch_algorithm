
import torch
from cyy_torch_toolbox import EvaluationMode, ModelEvaluator
from cyy_torch_toolbox.typing import TensorDict


def eval_model(
    parameter_dict: TensorDict,
    model_evaluator: ModelEvaluator,
    input_shape: None | torch.Tensor = None,
    hugging_face_batch_encoding: dict | None = None,
    **kwargs,
) -> torch.Tensor:
    model_evaluator.model_util.load_buffer_dict(parameter_dict)
    kwargs |= {
        "evaluation_mode": EvaluationMode.Test,
    }
    if input_shape is not None:
        kwargs["inputs"] = kwargs["inputs"].view(input_shape)
    if hugging_face_batch_encoding is not None:
        hugging_face_batch_encoding = hugging_face_batch_encoding | {
            "inputs_embeds": kwargs.pop("inputs")
        }
        kwargs["inputs"] = hugging_face_batch_encoding
    return model_evaluator(**kwargs)["loss"]
