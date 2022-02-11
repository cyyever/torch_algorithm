import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter

from hydra.hydra_hook import HyDRAHook


class HyDRASGDHook(HyDRAHook):
    def _before_batch(self, **kwargs):
        super()._before_batch(**kwargs)
        trainer = kwargs["model_executor"]

        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")

        cur_learning_rate = trainer.get_data("cur_learning_rates")[0]
        batch_size = kwargs["batch_size"]
        momentum = optimizer.param_groups[0]["momentum"]
        weight_decay = trainer.hyper_parameter.weight_decay

        for idx in self._computed_indices:
            instance_gradient = self.sample_gradient_dict.get(idx, None)
            if instance_gradient is not None:
                instance_gradient = (
                    instance_gradient.to(self._device)
                    * self._training_set_size
                    / batch_size
                )
            if self.use_hessian:
                self._hessian_computation_arguments[idx] = [
                    (
                        momentum,
                        weight_decay,
                        cur_learning_rate,
                        instance_gradient,
                    )
                ]
            if self.use_approximation:
                if idx not in self._delayed_approximation_computations:
                    self._delayed_approximation_computations[idx] = []
                self._delayed_approximation_computations[idx].append(
                    (momentum, weight_decay, cur_learning_rate, instance_gradient)
                )
                if instance_gradient is not None:
                    self._do_delayed_computation(use_approximation=True, index=idx)
        if self.use_hessian:
            self._do_computation_with_hessian()

    def get_hyper_gradient(self, index, use_approximation):
        return self._get_hyper_gradient_tensors(index, use_approximation)[0]

    def _decode_hyper_gradient_tensors(self, tensor):
        return torch.split(tensor, tensor.shape[0] // 2)

    def _get_hyper_gradient_tensors(self, index, use_approximation):
        data = self._get_hyper_gradient_dict(use_approximation)[index]
        if data is None:
            return None, None
        return self._decode_hyper_gradient_tensors(data)

    def _do_computation_with_hessian(self):
        for chunk in split_list_to_chunks(
            list(self._computed_indices), self._cache_size
        ):
            counter = TimeCounter()
            self._hessian_hyper_gradient_dict.prefetch(chunk)
            hyper_gradients = []
            hyper_gradient_indices = []
            hessian_vector_product_dict = {}
            for index in chunk:
                hyper_gradient = self.get_hyper_gradient(index, use_approximation=False)
                if hyper_gradient is not None:
                    hyper_gradients.append(hyper_gradient)
                    hyper_gradient_indices.append(index)
            if hyper_gradients:
                counter2 = TimeCounter()
                hessian_vector_products = self._hvp_function(hyper_gradients)
                get_logger().debug(
                    "hvp chunk size %s use time %s ms",
                    len(hyper_gradients),
                    counter2.elapsed_milliseconds(),
                )

                assert len(hyper_gradients) == len(hessian_vector_products)
                hessian_vector_product_dict = dict(
                    zip(hyper_gradient_indices, hessian_vector_products)
                )

            for index in chunk:
                self._do_delayed_computation(
                    False, index, hessian_vector_product_dict.get(index, None)
                )
            get_logger().debug(
                "_do_computation_with_hessian chunk size %s use time %s ms",
                len(chunk),
                counter.elapsed_milliseconds(),
            )

    def _do_delayed_computation(
        self, use_approximation: bool, index, hessian_vector_product=None
    ):

        if hessian_vector_product is not None:
            hessian_vector_product.to_(self._device)
        hyper_gradient, mom_gradient = self._get_hyper_gradient_tensors(
            index, use_approximation
        )

        if use_approximation:
            argument_dict = self._delayed_approximation_computations
        else:
            argument_dict = self._hessian_computation_arguments
        for arguments in argument_dict.pop(index):
            (momentum, weight_decay, learning_rate, instance_gradient) = arguments
            gradient_gradient = self._optional_addition(
                self._optional_multiplication(weight_decay, hyper_gradient),
                instance_gradient,
                hessian_vector_product,
            )

            mom_gradient = self._optional_addition(
                self._optional_multiplication(mom_gradient, momentum), gradient_gradient
            )
            hyper_gradient = self._optional_addition(
                hyper_gradient,
                self._optional_multiplication(-learning_rate, mom_gradient),
            )
        if hyper_gradient is not None:
            assert mom_gradient is not None
            self._set_hyper_gradient_tensors(
                index, use_approximation, hyper_gradient, mom_gradient
            )
