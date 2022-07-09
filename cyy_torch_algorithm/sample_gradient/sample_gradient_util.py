import copy

from .sample_gradient_hook import SampleGradientHook


def get_sample_gradient_dict(inferencer, computed_indices=None):
    tmp_inferencer = copy.deepcopy(inferencer)
    hook = SampleGradientHook()
    if computed_indices is not None:
        hook.set_computed_indices(computed_indices)
    tmp_inferencer.append_hook(hook)
    tmp_inferencer.inference(use_grad=True)
    return hook.sample_result_dict
