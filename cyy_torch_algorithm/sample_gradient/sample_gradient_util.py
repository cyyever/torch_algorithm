import copy

from cyy_torch_toolbox.ml_type import ModelExecutorHookPoint

from .sample_gradient_hook import SampleGradientHook


def get_sample_gradient_dict(inferencer, computed_indices=None) -> dict:
    tmp_inferencer = copy.deepcopy(inferencer)
    hook = SampleGradientHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    gradients: dict = {}
    tmp_inferencer.append_hook(hook)

    def collect_gradients(**kwargs):
        nonlocal gradients
        gradients |= hook.sample_result_dict

    tmp_inferencer.append_named_hook(
        hook_point=ModelExecutorHookPoint.AFTER_BATCH,
        name="collect_gradients",
        fun=collect_gradients,
    )
    tmp_inferencer.inference(use_grad=True)
    assert gradients
    return hook.sample_result_dict
