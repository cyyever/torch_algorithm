import copy

from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint

from .sample_gradient_hook import SampleGradientHook


def get_sample_gradient_dict(inferencer, computed_indices=None) -> dict:
    tmp_inferencer = copy.deepcopy(inferencer)
    tmp_inferencer.disable_logger()
    hook = SampleGradientHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    gradients: dict = {}
    tmp_inferencer.append_hook(hook)

    tmp_inferencer.inference()
    gradients = hook.result_dict
    assert gradients
    return gradients
