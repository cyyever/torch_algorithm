import torch
from cyy_torch_toolbox import EvaluationMode, ModelEvaluator, ModelParameter


def eval_model(
    parameters: ModelParameter,
    model_evaluator: ModelEvaluator,
    input_shape: None | torch.Size = None,
    hugging_face_batch_encoding: dict | None = None,
    **kwargs,
) -> torch.Tensor:
    model_evaluator.model_util.load_buffers(parameters)
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
