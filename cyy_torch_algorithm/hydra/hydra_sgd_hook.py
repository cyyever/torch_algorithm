from cyy_torch_algorithm.hydra.hydra_hook import HyDRAHook


class HyDRASGDHook(HyDRAHook):
    __momentum = None
    __lr = None
    __weight_decay = None

    def _before_batch(self, model_executor, **kwargs):
        trainer = model_executor
        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1

        self.__momentum = optimizer.param_groups[0]["momentum"]
        self.__lr = optimizer.param_groups[0]["lr"]
        self.__weight_decay = optimizer.param_groups[0]["weight_decay"]
        super()._before_batch(model_executor=model_executor, **kwargs)

    def _after_optimizer_step(self, model_executor, batch_size, step_skipped, **kwargs):
        if step_skipped:
            self._sample_gradient_hook.reset_result()
            return

        for idx in self._computed_indices:
            instance_gradient = self.sample_gradient_dict.get(idx, None)
            if instance_gradient is not None:
                instance_gradient = (
                    instance_gradient.to(self._trainer.device)
                    * self._training_set_size
                    / batch_size
                )
            arguments = (
                self.__momentum,
                self.__weight_decay,
                self.__lr,
                instance_gradient,
            )
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
        self._sample_gradient_hook.reset_result()

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
