from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model(
    parameter_list,
    model_with_loss,
    targets,
    device=None,
    inputs=None,
    phase=MachineLearningPhase.Training,
    non_blocking=True,
    is_input_feature=False,
    input_shape=None,
    parameter_shapes=None,
    **kwargs
):
    model_with_loss.model_util.load_parameter_list(
        parameter_list.to(device, non_blocking=non_blocking),
        check_parameter=False,
        as_parameter=False,
        parameter_shapes=parameter_shapes,
    )
    input_kwargs = kwargs
    kwargs = {
        "targets": targets,
        "device": device,
        "non_blocking": non_blocking,
        "phase": phase,
    }
    if input_kwargs:
        kwargs["inputs"] = input_kwargs
    else:
        if input_shape is not None:
            inputs = inputs.view(input_shape)
        if is_input_feature:
            kwargs["input_features"] = inputs
            kwargs["inputs"] = None
        else:
            kwargs["inputs"] = inputs

    return model_with_loss(**kwargs)["loss"]
