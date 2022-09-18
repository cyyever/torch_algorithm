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
    # if model_with_loss.model_util.cached_buffer_names is not None:
    #     for name in model_with_loss.model_util.cached_buffer_names:
    #         buf = model_with_loss.model_util.get_attr(name)
    #         print("convert buf", name)
    #         model_with_loss.model_util.set_attr(
    #             name,
    #             buf.to(device=device, non_blocking=non_blocking),
    #             as_parameter=False,
    #         )
    # else:
    for name, buf in list(model_with_loss.model.named_buffers()):
        if buf.device == device:
            break
        model_with_loss.model_util.set_attr(
            name,
            buf.to(device=device, non_blocking=non_blocking),
            as_parameter=False,
        )
    model_with_loss.model_util.load_parameter_list(
        parameter_list.to(device, non_blocking=non_blocking),
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
        kwargs["input_features"] = None

    return model_with_loss(**kwargs)["loss"]
