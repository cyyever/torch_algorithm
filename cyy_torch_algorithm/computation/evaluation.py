from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    phase=MachineLearningPhase.Training,
    non_blocking=True,
    input_features=None,
    real_inputs=None,
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
    if real_inputs is not None:
        if input_features is not None:
            real_inputs = real_inputs.view(input_features.shape)
            input_features = real_inputs
        else:
            real_inputs = real_inputs.view(inputs.shape)
            inputs = real_inputs
    return model_with_loss(**kwargs)["loss"]
