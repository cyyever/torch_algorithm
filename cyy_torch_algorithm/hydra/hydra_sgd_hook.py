import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from hessian_vector_product import get_hessian_vector_product_func

from hydra.hydra_hook import HyDRAHook


class HyDRASGDHook(HyDRAHook):
    hessian_computation_arguments = None

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        if self.use_hessian:
            self._hvp_function = get_hessian_vector_product_func(
                trainer.copy_model_with_loss(deepcopy=True), batch
            )
            self.hessian_computation_arguments = {}

        optimizer = trainer.get_optimizer()
        assert len(optimizer.param_groups) == 1
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")

        cur_learning_rate = trainer.get_data("cur_learning_rates")[0]
        batch_size = kwargs["batch_size"]
        momentum = optimizer.param_groups[0]["momentum"]
        weight_decay = trainer.hyper_parameter.weight_decay
        training_set_size = len(trainer.dataset)

        counter = TimeCounter()
        for idx in self._computed_indices:
            instance_gradient = None
            if idx in self.sample_gradient_dict:
                instance_gradient = self.sample_gradient_dict[idx]
                instance_gradient = (
                    instance_gradient.detach() * training_set_size / batch_size
                )
            if self.use_hessian:
                self.hessian_computation_arguments[idx] = (
                    momentum,
                    weight_decay,
                    cur_learning_rate,
                    instance_gradient,
                )
            if self.use_approximation:
                if idx not in self.delayed_approximation_computations:
                    self.delayed_approximation_computations[idx] = []
                self.delayed_approximation_computations[idx].append(
                    (momentum, weight_decay, cur_learning_rate, instance_gradient)
                )
                if instance_gradient is not None:
                    self._do_delayed_computation(idx)
        get_logger().debug(
            "use_approximation use time %s ms",
            counter.elapsed_milliseconds(),
        )
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
            list(self._computed_indices),
            self._cache_size // 2,
        ):
            counter = TimeCounter()
            self._hessian_hyper_gradient_dict.prefetch(chunk)
            hyper_gradients = []
            hyper_gradient_indices = []
            hessian_vector_product_dict = {}
            for index in chunk:
                if index in self._hessian_hyper_gradient_dict:
                    hyper_gradients.append(
                        self.get_hyper_gradient(index, use_approximation=False)
                    )
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
                for idx, hessian_vector_product in zip(
                    hyper_gradient_indices, hessian_vector_products
                ):
                    hessian_vector_product_dict[idx] = hessian_vector_product

            for index in chunk:
                (
                    momentum,
                    weight_decay,
                    learning_rate,
                    instance_gradient,
                ) = self.hessian_computation_arguments[index]

                hyper_gradient = None
                mom_gradient = None
                if index in self._hessian_hyper_gradient_dict:
                    (
                        hyper_gradient,
                        mom_gradient,
                    ) = self._get_hyper_gradient_tensors(index, use_approximation=False)

                if mom_gradient is not None:
                    mom_gradient *= momentum

                if hyper_gradient is not None:
                    res = weight_decay * hyper_gradient
                    res += hessian_vector_product_dict[index]
                    if mom_gradient is not None:
                        mom_gradient += res
                    else:
                        mom_gradient = res

                if instance_gradient is not None:
                    if mom_gradient is not None:
                        mom_gradient += instance_gradient
                    else:
                        mom_gradient = instance_gradient

                if mom_gradient is not None:
                    if hyper_gradient is not None:
                        hyper_gradient -= learning_rate * mom_gradient
                    else:
                        hyper_gradient = -learning_rate * mom_gradient

                assert (hyper_gradient is not None and mom_gradient is not None) or (
                    hyper_gradient is None and mom_gradient is None
                )
                if hyper_gradient is not None:
                    self._set_hyper_gradient_tensors(
                        index, False, hyper_gradient, mom_gradient
                    )
                self.hessian_computation_arguments[index] = None
            get_logger().debug(
                "_do_computation_with_hessian chunk size %s use time %s ms",
                len(chunk),
                counter.elapsed_milliseconds(),
            )

    def _do_delayed_computation(self, index=None):
        if index is None:
            self._get_hyper_gradient_dict(True).prefetch(
                self.delayed_approximation_computations.keys()
            )
            for k in self.delayed_approximation_computations:
                get_logger().debug("do delayed_approximation_computations for %s", k)
                self._do_delayed_computation(k)
            return

        hyper_gradient, mom_gradient = self._get_hyper_gradient_tensors(
            index, use_approximation=True
        )
        for arguments in self.delayed_approximation_computations[index]:
            (momentum, weight_decay, learning_rate, instance_gradient) = arguments
            if mom_gradient is not None:
                mom_gradient *= momentum

            if hyper_gradient is not None:
                res = weight_decay * hyper_gradient
                if mom_gradient is not None:
                    mom_gradient += res
                else:
                    mom_gradient = res

            if instance_gradient is not None:
                if mom_gradient is not None:
                    mom_gradient += instance_gradient
                else:
                    mom_gradient = instance_gradient

            if mom_gradient is not None:
                if hyper_gradient is not None:
                    hyper_gradient -= learning_rate * mom_gradient
                else:
                    hyper_gradient = -learning_rate * mom_gradient

        assert hyper_gradient is not None
        assert mom_gradient is not None
        del self.delayed_approximation_computations[index]
        self._set_hyper_gradient_tensors(index, True, hyper_gradient, mom_gradient)
