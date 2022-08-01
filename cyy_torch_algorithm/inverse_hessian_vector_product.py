#!/usr/bin/env python3

import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)

from cyy_torch_algorithm.batch_hvp.batch_hvp_hook import BatchHVPHook


def stochastic_inverse_hessian_vector_product(
    inferencer: Inferencer,
    vector,
    repeated_num=1,
    max_iteration=1000,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
):
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        dampling_term,
        scale,
        epsilon,
    )

    def iteration() -> torch.Tensor:
        nonlocal vector
        vector = vector.cpu()
        cur_product = copy.deepcopy(vector)
        iteration_num = 0
        hook = BatchHVPHook()

        epoch = 1

        def set_vectors(**kwargs):
            nonlocal cur_product
            hook.set_vectors([cur_product])

        result = None

        def compute_product(**kwargs) -> None:
            nonlocal cur_product
            nonlocal result
            nonlocal iteration_num
            next_product = (
                vector
                + (1 - dampling_term) * cur_product
                - hook.result_dict[0][0].cpu() / scale
            )
            diff = torch.dist(cur_product, next_product)
            get_logger().debug(
                "diff is %s, epsilon is %s, epoch is %s,iteration is %s, max_iteration is %s",
                diff,
                epsilon,
                epoch,
                iteration_num,
                max_iteration,
            )
            cur_product = next_product
            iteration_num += 1
            # and epoch > 1:
            if diff <= epsilon or iteration_num >= max_iteration:
                result = cur_product / scale
                raise StopExecutingException()

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_FORWARD,
            name="set_vectors",
            fun=set_vectors,
        )
        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_FORWARD,
            name="compute_product",
            fun=compute_product,
        )

        while result is None:
            tmp_inferencer.inference(use_grad=False, epoch=epoch)
            epoch += 1
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
        return result

    product_list = [iteration() for _ in range(repeated_num)]
    return sum(product_list) / len(product_list)
