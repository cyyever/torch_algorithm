from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model_by_parameter(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    phase=MachineLearningPhase.Training,
    forward_embedding=False,
):
    model_with_loss.model_util.load_parameter_list(
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
