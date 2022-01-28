#!/usr/bin/env python3

import torch
# from cyy_naive_lib.profiling import Profile
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.default_config import DefaultConfig
from sample_gradient.sample_gradient import (get_sample_gradient,
                                             stop_task_queue)


def test_get_sample_gradient():
    trainer = DefaultConfig("CIFAR10", "resnet18").create_trainer()
    training_data_loader = torch.utils.data.DataLoader(
        trainer.dataset,
        batch_size=8,
        shuffle=True,
    )

    for cnt, batch in enumerate(training_data_loader):
        with TimeCounter():
            gradients = get_sample_gradient(
                trainer.copy_model_with_loss(), batch[0], batch[1]
            )
            if cnt == 0:
                print("sample_gradient result", gradients)
            del gradients
        if cnt > 3:
            break

    # with Profile():
    #     for batch in training_data_loader:
    #         get_sample_gradient(trainer.model_with_loss, batch[0], batch[1])
    #         break
    stop_task_queue()
