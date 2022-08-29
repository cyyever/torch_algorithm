import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRAAdamHook(LeanHyDRAHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__gradient_product = None
        self.__first_average_product = None
        self.__second_average_product = None
        self.__step = None

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]

        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if self._contributions is None:
            self._contributions = torch.zeros(self._training_set_size).to(
                trainer.device
            )
            self.__gradient_product = torch.zeros(self._training_set_size).to(
                trainer.device
            )
            self.__first_average_product = torch.zeros(self._training_set_size).to(
                trainer.device
            )
            self.__second_average_product = torch.zeros(self._training_set_size).to(
                trainer.device
            )
            self._test_gradient = self._test_gradient.to(trainer.device)
            self.__step = 0
        self.__step += 1

        cur_learning_rate = trainer.get_data("cur_learning_rates")[0]
        batch_size = kwargs["batch_size"]

        counter = TimeCounter()
        self.__gradient_product = (
            trainer.hyper_parameter.weight_decay * self._contributions
        )
        for idx, dot_product in self.sample_gradient_dict.items():
            self.__gradient_product[idx] += (
                dot_product * self._training_set_size / batch_size
            )

        beta1, beta2 = optimizer.param_groups[0]["betas"]
        self.__first_average_product = (
            beta1 * self.__first_average_product + (1 - beta1) * self.__gradient_product
        ) / (1 - (beta1**self.__step))
        self.__second_average_product = (
            beta2 * self.__second_average_product
            + 2 * (1 - beta2) * self.__gradient_product
        ) / (1 - (beta2**self.__step))
        # self._contributions -= cur_learning_rate * self.__gradient_product
        get_logger().debug(
            "batch use time %s ms",
            counter.elapsed_milliseconds(),
        )

    def _after_optimizer_step(self, **kwargs):
        trainer = kwargs["model_executor"]
        optimizer = trainer.get_optimizer()
        assert self.__step == list(optimizer.state.values())[0]["step"]
        print("exec after optimizer")

    def _after_execute(self, **kwargs):
        self._contributions = (-self._contributions) / self._training_set_size
        super()._after_execute(**kwargs)
