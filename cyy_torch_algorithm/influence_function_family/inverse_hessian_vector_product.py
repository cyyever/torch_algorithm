#!/usr/bin/env python3

import copy

import torch
from cyy_naive_lib.algorithm.mapping_op import get_mapping_values_by_key_order
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.tensor import cat_tensors_to_vector, tensor_to


def stochastic_inverse_hessian_vector_product(
    inferencer: Inferencer,
    vectors: list,
    repeated_num: int = 1,
    max_iteration: int = 1000,
    dampling_term: float = 0,
    scale: float = 1,
    epsilon: float = 0.0001,
) -> torch.Tensor:
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        dampling_term,
        scale,
        epsilon,
    )

    vectors = torch.stack(vectors)

    def iteration(inferencer, vectors) -> torch.Tensor:
        cur_products = copy.deepcopy(vectors)
        iteration_num = 0
        hook = BatchHVPHook()

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.disable_logger()
        tmp_inferencer.disable_performance_metric_logger()
        cur_products = tensor_to(cur_products, device=tmp_inferencer.device)
        vectors = tensor_to(vectors, device=tmp_inferencer.device)

        results: None | torch.Tensor = None

        def compute_product(epoch, **kwargs) -> None:
            nonlocal cur_products
            nonlocal results
            nonlocal iteration_num
            nonlocal vectors
            next_products = (
                vectors
                + (1 - dampling_term) * cur_products
                - cat_tensors_to_vector(
                    get_mapping_values_by_key_order(
                        tensor_to(hook.result_dict, device=vectors.device)
                    )
                )
                / scale
            )
            hook.reset_result()
            diffs = torch.tensor(
                [torch.dist(a, b) for a, b in zip(cur_products, next_products)]
            )
            get_logger().error(
                "diffs is %s, epsilon is %s, epoch is %s, iteration is %s, max_iteration is %s, scale %s",
                diffs,
                epsilon,
                epoch,
                iteration_num,
                max_iteration,
                scale,
            )
            cur_products = next_products
            iteration_num += 1
            if (
                (diffs <= epsilon).all().bool() and epoch > 1
            ) or iteration_num >= max_iteration:
                results = cur_products / scale
                raise StopExecutingException()

        hook.set_data_fun(lambda: cur_products)
        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_FORWARD,
            name="compute_product",
            fun=compute_product,
        )
        epoch = 1
        while results is None:
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
            normal_stop = tmp_inferencer.inference(use_grad=False, epoch=epoch)
            if not normal_stop:
                break
            epoch += 1
        del cur_products
        hook.release_queue()
        return results.cpu()

    product_list = [iteration(inferencer, vectors) for _ in range(repeated_num)]
    return sum(product_list) / len(product_list)
