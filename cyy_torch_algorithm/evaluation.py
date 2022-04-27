from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model_by_parameter(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    model_util=None,
    phase=MachineLearningPhase.Training,
    forward_embedding=False,
):
    if model_util is None:
        model_util = model_with_loss.model_util
    model_util.load_parameter_list(
        parameter_list.to(device),
        check_parameter=False,
        as_parameter=False,
    )
    if not forward_embedding:
        model_fun = model_with_loss.model
    else:
        model_fun = model_with_loss.model.forward_embedding

    return model_with_loss(
        inputs,
        targets,
        device=device,
        non_blocking=False,
        phase=phase,
        model_fun=model_fun,
    )["loss"]


def eval_model_foreach(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    input_shape=None,
    model_util=None,
    phase=MachineLearningPhase.Training,
    is_input_feature=False,
    non_blocking=False,
):
    if model_util is None:
        model_util = model_with_loss.model_util
    model_util.load_parameter_list(
        parameter_list.to(device),
        check_parameter=False,
        as_parameter=False,
    )

    if input_shape is not None:
        inputs = [sample_input.view(input_shape) for sample_input in inputs]

    total_loss = None
    kwargs = {
        "device": device,
        "non_blocking": non_blocking,
        "phase": phase,
    }
    for sample_input, sample_target in zip(inputs, targets):
        kwargs["targets"] = sample_target
        if is_input_feature:
            kwargs["input_features"] = sample_input
            kwargs["inputs"] = None
        else:
            kwargs["inputs"] = sample_input
        loss = model_with_loss(**kwargs)["loss"]
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss
    return total_loss
