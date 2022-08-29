import torch
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from .lean_hydra_hook import LeanHyDRAHook


class LeanHyDRASGDHook(LeanHyDRAHook):
    __mom_product = None

    def _before_execute(self, **kwargs):
        super()._before_execute(**kwargs)
        trainer = kwargs["model_executor"]
        self.__mom_product = torch.zeros(self._training_set_size).to(
            trainer.device, non_blocking=True
        )

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1

        batch_size = kwargs["batch_size"]
        momentum = optimizer.param_groups[0]["momentum"]
        lr = optimizer.param_groups[0]["lr"]
        weight_decay = optimizer.param_groups[0]["weight_decay"]

        counter = TimeCounter()
        self.__mom_product = (
            self.__mom_product * momentum + weight_decay * self._contributions
        )

        for idx, dot_product in self.sample_gradient_hook.result_dict.items():
            self.__mom_product[idx] += (
                dot_product * self._training_set_size / batch_size
            )
        self.sample_gradient_hook.reset_result()
        self._contributions -= lr * self.__mom_product
        get_logger().debug(
            "batch use time %s ms",
            counter.elapsed_milliseconds(),
        )

    def _after_execute(self, **kwargs):
        self._contributions = (-self._contributions) / self._training_set_size
        super()._after_execute(**kwargs)
