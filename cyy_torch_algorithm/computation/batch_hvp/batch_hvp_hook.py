import collections
import functools

import torch
import torch.cuda
from cyy_naive_lib.algorithm.mapping_op import get_mapping_items_by_key_order
from cyy_torch_algorithm.computation.batch_computation_hook import \
    BatchComputationHook
from cyy_torch_algorithm.computation.evaluation import eval_model2

try:
    from torch.func import grad, jvp, vmap
except BaseException:
    from functorch import grad, jvp, vmap


def batch_hvp_worker_fun(
    model_evaluator, inputs, targets, data, worker_device, parameter_dict, **kwargs
) -> list:
    vector_size = len(data)
    vectors = data
    parameter_dict = collections.OrderedDict(
        list(get_mapping_items_by_key_order(parameter_dict))
    )

    def hvp_wrapper(vector):
        f = functools.partial(
            eval_model2,
            inputs=inputs,
            targets=targets,
            device=worker_device,
            model_evaluator=model_evaluator,
        )

        def grad_f(parameter_dict):
            return grad(f, argnums=0)(parameter_dict)
            # return cat_tensor_dict(grad(f, argnums=0)(parameter_dict))

        return jvp(
            grad_f,
            (parameter_dict,),
            (vector,),
        )[1]

    match vectors[0]:
        case torch.Tensor:
            vectors = torch.stack(vectors)
        case dict():
            vectors = {
                k: torch.stack([vector[k] for vector in vectors]) for k in vectors[0]
            }

    vectors = collections.OrderedDict(list(get_mapping_items_by_key_order(vectors)))
    products = vmap(
        hvp_wrapper,
        in_dims=(collections.OrderedDict((k, 0) for k in vectors),),
        # randomness="same",
    )(vectors)
    return [{k: v[idx] for k, v in products.items()} for idx in range(vector_size)]


class BatchHVPHook(BatchComputationHook):
    def set_vectors(self, vectors):
        self.set_data_fun(lambda: vectors)

    def _get_batch_computation_fun(self):
        return batch_hvp_worker_fun
