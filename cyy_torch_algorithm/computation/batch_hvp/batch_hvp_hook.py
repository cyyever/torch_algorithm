import collections
import functools
from collections.abc import Callable

import torch
import torch.cuda
from cyy_naive_lib.algorithm.mapping_op import get_mapping_items_by_key_order
from cyy_torch_toolbox import ModelEvaluator, ModelParameter, TensorDict
from cyy_torch_toolbox.tensor import cat_tensor_dict, decompose_like_tensor_dict
from torch.func import grad, jvp, vmap

from ..batch_computation_hook import BatchComputationHook
from ..evaluation import eval_model


def batch_hvp_worker_fun(
    model_evaluator: ModelEvaluator,
    inputs,
    targets,
    data,
    worker_device: torch.device,
    parameters: ModelParameter,
) -> list[TensorDict] | list[torch.Tensor]:
    assert data
    vector_size = len(data)
    vectors = data
    parameters = collections.OrderedDict(
        list(get_mapping_items_by_key_order(parameters))
    )

    def hvp_wrapper(vector):
        f = functools.partial(
            eval_model,
            inputs=inputs,
            targets=targets,
            device=worker_device,
            model_evaluator=model_evaluator,
        )

        def grad_f(parameters):
            return grad(f, argnums=0)(parameters)

        return jvp(
            grad_f,
            (parameters,),
            (vector,),
        )[1]

    decompose_vector = False
    if not isinstance(vectors[0], dict):
        decompose_vector = True
        vectors = [decompose_like_tensor_dict(parameters, vector) for vector in vectors]
        assert len(vectors) == vector_size
    new_vectors = {
        k: torch.stack([vector[k] for vector in vectors]) for k in vectors[0]
    }
    new_vectors = collections.OrderedDict(
        list(get_mapping_items_by_key_order(new_vectors))
    )
    res = vmap(
        hvp_wrapper,
        in_dims=(collections.OrderedDict((k, 0) for k in new_vectors),),
        randomness="same",
    )(new_vectors)
    products: list[TensorDict] | list[torch.Tensor] = [
        {k: v[idx] for k, v in res.items()} for idx in range(vector_size)
    ]
    if decompose_vector:
        products = [cat_tensor_dict(product) for product in products]
    assert len(products) == vector_size
    return products


class BatchHVPHook(BatchComputationHook):
    vectors: list[torch.Tensor] | list[TensorDict] = []

    def get_vectors(self) -> list[torch.Tensor] | list[TensorDict]:
        return self.vectors

    def set_vectors(self, vectors: list[torch.Tensor] | list[TensorDict]) -> None:
        self.vectors = vectors
        self.set_data_fun(self.get_vectors)

    def _get_batch_computation_fun(self) -> Callable:
        return batch_hvp_worker_fun
