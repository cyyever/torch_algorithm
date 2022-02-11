import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from hydra.hydra_hook import HyDRAHook


class HyDRAAdamHook(HyDRAHook):
    __step = None

    def _before_batch(self, **kwargs):
        super()._before_batch(**kwargs)
        trainer = kwargs["model_executor"]
        assert not self.use_hessian

        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.Adam):
            raise RuntimeError("optimizer is not Adam")

        if self.__step is None:
            self.__step = 0
        self.__step += 1

        cur_learning_rate = trainer.get_data("cur_learning_rates")[0]
        batch_size = kwargs["batch_size"]
        betas = optimizer.param_groups[0]["betas"]
        weight_decay = trainer.hyper_parameter.weight_decay

        for idx in self._computed_indices:
            instance_gradient = self.sample_gradient_dict.get(idx, None)
            if instance_gradient is not None:
                instance_gradient = (
                    instance_gradient.to(self._device)
                    * self._training_set_size
                    / batch_size
                )
            if self.use_approximation:
                if idx not in self._delayed_approximation_computations:
                    self._delayed_approximation_computations[idx] = []
                self._delayed_approximation_computations[idx].append(
                    (betas, weight_decay, cur_learning_rate, instance_gradient)
                )
                if instance_gradient is not None:
                    self._do_delayed_computation(use_approximation=True, index=idx)

    def get_hyper_gradient(self, index, use_approximation):
        return self._get_hyper_gradient_tensors(index, use_approximation)[0]

    def _decode_hyper_gradient_tensors(self, tensor):
        return torch.split(tensor, tensor.shape[0] // 3)

    def _get_hyper_gradient_tensors(self, index, use_approximation):
        data = self._get_hyper_gradient_dict(use_approximation)[index]
        if data is None:
            return None, None, None
        return self._decode_hyper_gradient_tensors(data)

    def _do_delayed_computation(
        self, use_approximation: bool, index, hessian_vector_product=None
    ):

        (
            hyper_gradient,
            first_average_gradient,
            second_average_gradient,
        ) = self._get_hyper_gradient_tensors(index, use_approximation)

        if use_approximation:
            argument_dict = self._delayed_approximation_computations
        else:
            argument_dict = self._hessian_computation_arguments
        for arguments in argument_dict.pop(index):
            (betas, weight_decay, learning_rate, instance_gradient) = arguments
            beta1, beta2 = betas

            gradient_gradient = self._optional_addition(
                self._optional_multiplication(weight_decay, hyper_gradient),
                instance_gradient,
                hessian_vector_product,
            )

            first_average_gradient = self._optional_addition(
                self._optional_multiplication(first_average_gradient, beta1),
                (1 - beta1) * gradient_gradient,
            )
            second_average_gradient = self._optional_addition(
                self._optional_multiplication(second_average_gradient, beta2),
                2 * (1 - beta2) * gradient_gradient,
            )

        if hyper_gradient is not None:
            assert first_average_gradient is not None
            assert second_average_gradient is not None
            self._set_hyper_gradient_tensors(
                index,
                use_approximation,
                hyper_gradient,
                first_average_gradient,
                second_average_gradient,
            )

    def _after_execute(self, **kwargs):
        self.__step = None
        super()._after_execute(**kwargs)
