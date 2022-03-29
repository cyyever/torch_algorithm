from cyy_torch_toolbox.ml_type import MachineLearningPhase


def eval_model_by_parameter(
    parameter_list,
    inputs,
    targets,
    device,
    model_with_loss,
    phase=MachineLearningPhase.Training,
    *args
):
    match parameter_list:
        case tuple() | list():
            parameter_list = parameter_list[0]
    parameter_list = parameter_list.to(device)
    model_util = model_with_loss.model_util
    model_util.load_parameter_list(
        parameter_list,
        check_parameter=False,
        as_parameter=False,
    )
    return model_with_loss(
        inputs,
        targets,
        device=device,
        non_blocking=False,
        phase=phase,
    )["loss"]
