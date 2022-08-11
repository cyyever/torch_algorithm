#!/usr/bin/env python3

import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_algorithm.computation.batch_hvp.batch_hvp_hook import \
    BatchHVPHook
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)


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

    vectors = torch.stack(vectors).to(device="cuda:0")

    def iteration() -> torch.Tensor:
        nonlocal vectors
        cur_products = copy.deepcopy(vectors)
        iteration_num = 0
        hook = BatchHVPHook()

        results = None

        def compute_product(epoch, **kwargs) -> None:
            nonlocal cur_products
            nonlocal results
            nonlocal iteration_num
            get_logger().error("result_dict values are %s", hook.result_dict[0])
            assert hook.result_dict[0].shape[0] == vectors.shape[0]
            next_products = (
                vectors
                + (1 - dampling_term) * cur_products
                - hook.result_dict[0] / scale
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

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.disable_logger()
        tmp_inferencer.disable_performance_metric_logger()
        hook.set_data_fun(lambda: cur_products)
        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_FORWARD,
            name="compute_product",
            fun=compute_product,
        )
        epoch = 1
        while results is None:
            tmp_inferencer.inference(use_grad=False, epoch=epoch)
            epoch += 1
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
        del cur_products
        hook.release_queue()
        return results.cpu()

    product_list = [iteration() for _ in range(repeated_num)]
    # print("product_list is ", product_list)
    return sum(product_list) / len(product_list)
