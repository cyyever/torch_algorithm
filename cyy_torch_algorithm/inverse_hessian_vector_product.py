#!/usr/bin/env python3

import copy

import torch
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.device import get_device
from cyy_torch_toolbox.inferencer import Inferencer
from cyy_torch_toolbox.ml_type import (ModelExecutorHookPoint,
                                       StopExecutingException)
from cyy_torch_toolbox.model_with_loss import ModelWithLoss

from cyy_torch_algorithm.batch_hvp.batch_hvp_hook import BatchHVPHook

from .hessian_vector_product import get_hessian_vector_product_func


def stochastic_inverse_hessian_vector_product(
    dataset,
    model_with_loss: ModelWithLoss,
    v,
    repeated_num=1,
    max_iteration=None,
    batch_size=1,
    dampling_term=0,
    scale=1,
    epsilon=0.0001,
):
    if max_iteration is None:
        max_iteration = 1000
    get_logger().info(
        "repeated_num is %s,max_iteration is %s,batch_size is %s,dampling term is %s,scale is %s,epsilon is %s",
        repeated_num,
        max_iteration,
        batch_size,
        dampling_term,
        scale,
        epsilon,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    product_list = []
    for cur_repeated_id in range(repeated_num):
        cur_product = v
        iteration = 0

        epoch = 1
        looping = True
        while looping:
            for batch in data_loader:
                hvp_function = get_hessian_vector_product_func(model_with_loss, batch)

                next_product = (
                    v
                    + (1 - dampling_term) * cur_product
                    - hvp_function(cur_product).to(get_device()) / scale
                )
                diff = torch.dist(cur_product, next_product)
                get_logger().debug(
                    "diff is %s, cur repeated sequence %s, iteration is %s, max_iteration is %s",
                    diff,
                    cur_repeated_id,
                    iteration,
                    max_iteration,
                )
                cur_product = next_product
                iteration += 1
                if (diff <= epsilon or iteration >= max_iteration) and epoch > 1:
                    product_list.append(cur_product / scale)
                    looping = False
                    break
            epoch += 1
            get_logger().debug(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
    return sum(product_list) / len(product_list)


def stochastic_inverse_hessian_vector_product_new(
    inferencer: Inferencer,
    v,
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
        nonlocal v
        v = v.cpu()
        cur_product = copy.deepcopy(v)
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
                v
                + (1 - dampling_term) * cur_product
                - hook.sample_result_dict[0][0].cpu() / scale
            )
            diff = torch.dist(cur_product, next_product)
            get_logger().error(
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
            if (diff <= epsilon or iteration_num >= max_iteration):
                result = cur_product / scale
                raise StopExecutingException()

        tmp_inferencer = copy.deepcopy(inferencer)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_BATCH,
            name="set_vectors",
            fun=set_vectors,
        )
        tmp_inferencer.append_hook(hook)
        tmp_inferencer.append_named_hook(
            hook_point=ModelExecutorHookPoint.AFTER_BATCH,
            name="compute_product",
            fun=compute_product,
        )

        while result is None:
            tmp_inferencer.inference(use_grad=False, epoch=epoch)
            epoch += 1
            get_logger().error(
                "stochastic_inverse_hessian_vector_product epoch is %s", epoch
            )
        return result

    product_list = [iteration() for _ in range(repeated_num)]
    return sum(product_list) / len(product_list)
