from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model2(
    parameter_dict,
    model_evaluator,
    targets,
    device=None,
    inputs=None,
    phase=MachineLearningPhase.Training,
    is_input_feature=False,
    input_shape=None,
    **kwargs
):
    model_evaluator.model_util.load_parameter_dict(
        parameter_dict,
        check_parameter=False,
        as_parameter=False,
        # parameter_shapes=parameter_shapes,
        update_parameter_shapes=False,
    )
    input_kwargs = kwargs
    kwargs = {
        "targets": targets,
        "device": device,
        "non_blocking": True,
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

    return model_evaluator(**kwargs)["loss"]
