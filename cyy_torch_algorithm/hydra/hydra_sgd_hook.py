import torch

from hydra.hydra_hook import HyDRAHook


class HyDRASGDHook(HyDRAHook):
    def _before_batch(self, **kwargs):
        super()._before_batch(**kwargs)
        trainer = kwargs["model_executor"]

        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")

        batch_size = kwargs["batch_size"]
        momentum = optimizer.param_groups[0]["momentum"]
        lr = optimizer.param_groups[0]["lr"]
        weight_decay = optimizer.param_groups[0]["weight_decay"]

        for idx in self._computed_indices:
            instance_gradient = self.sample_gradient_dict.get(idx, None)
            if instance_gradient is not None:
                instance_gradient = (
                    instance_gradient.to(self._trainer.device)
                    * self._training_set_size
                    / batch_size
                )
            arguments = (momentum, weight_decay, lr, instance_gradient)
            if self.use_hessian:
                self._hessian_computation_arguments[idx] = [arguments]
            if self.use_approximation:
                if idx not in self._delayed_approximation_computations:
                    self._delayed_approximation_computations[idx] = []
                self._delayed_approximation_computations[idx].append(arguments)
                if instance_gradient is not None:
                    self._do_delayed_computation(use_approximation=True, index=idx)
        if self.use_hessian:
            self._do_computation_with_hessian()

    def _do_delayed_computation(
        self, use_approximation: bool, index, hessian_vector_product=None
    ):

        hyper_gradient, mom_gradient = self._get_hyper_gradient_tensors(
            index, use_approximation, none_num=2
        )

        if use_approximation:
            argument_dict = self._delayed_approximation_computations
        else:
            argument_dict = self._hessian_computation_arguments
        for arguments in argument_dict.pop(index):
            (momentum, weight_decay, learning_rate, instance_gradient) = arguments
            gradient_gradient = self._optional_addition(
                self._optional_multiplication(hyper_gradient, weight_decay),
                instance_gradient,
                hessian_vector_product,
            )
            self._check_overflow_and_underflow(gradient_gradient)

            mom_gradient = self._optional_addition(
                self._optional_multiplication(mom_gradient, momentum), gradient_gradient
            )
            self._check_overflow_and_underflow(mom_gradient)
            hyper_gradient = self._optional_addition(
                hyper_gradient,
                self._optional_multiplication(mom_gradient, -learning_rate),
            )
            self._check_overflow_and_underflow(hyper_gradient)
        if instance_gradient is not None:
            assert mom_gradient is not None
            assert hyper_gradient is not None

        if hyper_gradient is not None:
            assert mom_gradient is not None
            self._set_hyper_gradient_tensors(
                index, use_approximation, hyper_gradient, mom_gradient
            )
