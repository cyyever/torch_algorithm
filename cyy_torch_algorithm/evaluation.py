from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    phase=MachineLearningPhase.Training,
    non_blocking=True,
    is_input_feature=False,
    input_shape=None,
):
    model_util = model_with_loss.model_util
    model_util.load_parameter_list(
        parameter_list.to(device),
        check_parameter=False,
        as_parameter=False,
    )
    kwargs = {
        "targets": targets,
        "device": device,
        "non_blocking": non_blocking,
        "phase": phase,
    }
    if input_shape is not None:
        inputs = inputs.view(input_shape)
    if is_input_feature:
        kwargs["input_features"] = inputs
        kwargs["inputs"] = None
    else:
        kwargs["inputs"] = inputs

    return model_with_loss(**kwargs)["loss"]
