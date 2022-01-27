import json
import os
import pickle
import shutil

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_toolbox.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import MachineLearningPhase
from cyy_torch_toolbox.model_util import ModelUtil
from hessian_vector_product import get_hessian_vector_product_func
from sample_gradient.sample_gradient_hook import SampleGradientHook


class HyDRAHook(Hook):
    def __init__(self, cache_size, **kwargs):
        super().__init__(stripable=True)
        self.sample_gradient_hook = SampleGradientHook()
        self.__cache_size = cache_size
        self.__save_dir = None

        self.__computed_indices = None
        self.hessian_computation_arguments = None
        self.delayed_approximation_computations = None

        self.use_hessian = kwargs.get("use_hessian", False)
        self.__hessian_hyper_gradient_and_momentum_dir = None
        self.__hvp_function = None
        self.hessian_hyper_gradient_mom_dict = None
        self.use_approximation = kwargs.get("use_approximation", None)

        if self.use_approximation is None:
            self.use_approximation = not self.use_hessian

        self.__approx_hyper_gradient_and_momentum_dir = None
        self.__approx_hyper_gradient_mom_dict = None
        self.__batch_gradient_indices = None

    @property
    def sample_gradient_dict(self):
        return self.sample_gradient_hook.sample_gradient_dict

    def __get_save_dir(self, trainer):
        if self.__save_dir is None:
            self.__save_dir = os.path.join(trainer.save_dir, "HyDRA")
            os.makedirs(self.__save_dir, exist_ok=True)
        return self.__save_dir

    def get_hessian_hydra_dir(self, trainer=None):
        if self.__hessian_hyper_gradient_and_momentum_dir is None:
            self.__hessian_hyper_gradient_and_momentum_dir = os.path.join(
                self.__get_save_dir(trainer), "hessian_hyper_gradient_and_momentum_dir"
            )
        return self.__hessian_hyper_gradient_and_momentum_dir

    def get_approx_hydra_dir(self, trainer=None):
        if self.__approx_hyper_gradient_and_momentum_dir is None:
            self.__approx_hyper_gradient_and_momentum_dir = os.path.join(
                self.__get_save_dir(trainer), "approx_hyper_gradient_and_momentum_dir"
            )
        return self.__approx_hyper_gradient_and_momentum_dir

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        if not self.__computed_indices:
            self.__computed_indices = set(range(len(trainer.dataset)))
        else:
            get_logger().info("only compute %s indices", len(self.__computed_indices))
        with open(
            os.path.join(self.__get_save_dir(trainer), "tracking_indices.json"),
            mode="wb",
        ) as f:
            pickle.dump(self.__computed_indices, f)
        if self.use_hessian:
            get_logger().info("use hessian to compute hyper-gradients")
            self.hessian_hyper_gradient_mom_dict = HyDRAHook.create_hypergradient_dict(
                self.__cache_size,
                trainer.model,
                storage_dir=self.get_hessian_hydra_dir(trainer),
            )
            get_logger().debug(
                "use hessian_hyper_gradient_mom_dir:%s",
                os.path.abspath(self.hessian_hyper_gradient_mom_dict.get_storage_dir()),
            )
        if self.use_approximation:
            self.__approx_hyper_gradient_mom_dict = HyDRAHook.create_hypergradient_dict(
                self.__cache_size,
                trainer.model,
                storage_dir=self.get_approx_hydra_dir(trainer),
            )
            get_logger().info(
                "use hyper_gradient_mom_dir:%s",
                os.path.abspath(self.get_approx_hydra_dir(trainer)),
            )
            self.delayed_approximation_computations = {
                k: [] for k in self.__computed_indices
            }

    def set_computed_indices(self, computed_indices):
        self.__computed_indices = set(computed_indices)
        self.sample_gradient_hook.set_computed_indices(computed_indices)

    def _after_execute(self, **kwargs):
        get_logger().info("end hyper-gradient tracking")
        trainer = kwargs["model_executor"]
        tester = trainer.get_inferencer(phase=MachineLearningPhase.Test)
        tester.disable_logger()
        tester.disable_performance_metric_logger()
        test_gradient = tester.get_gradient()
        if self.use_approximation:
            self.__save_hyper_gradients(
                trainer,
                test_gradient,
                use_approximation=True,
            )
        if self.use_hessian:
            self.__save_hyper_gradients(
                trainer,
                test_gradient,
                use_approximation=False,
            )

    def __do_computation_with_hessian(self):
        for chunk in split_list_to_chunks(
            list(self.__computed_indices),
            self.__cache_size // 2,
        ):
            counter = TimeCounter()
            self.hessian_hyper_gradient_mom_dict.prefetch(chunk)
            hyper_gradients = []
            hyper_gradient_indices = []
            hessian_vector_product_dict = {}
            for index in chunk:
                if index in self.hessian_hyper_gradient_mom_dict:
                    hyper_gradients.append(
                        self.get_hyper_gradient(index, use_approximation=False)
                    )
                    hyper_gradient_indices.append(index)
            if hyper_gradients:
                counter2 = TimeCounter()
                hessian_vector_products = self.__hvp_function(hyper_gradients)
                get_logger().info(
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
                if index in self.hessian_hyper_gradient_mom_dict:
                    (
                        hyper_gradient,
                        mom_gradient,
                    ) = self.__get_hyper_gradient_and_momentum(
                        index, use_approximation=False
                    )

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
                    self.__set_hyper_gradient_and_momentum(
                        index, hyper_gradient, mom_gradient, use_approximation=False
                    )
                self.hessian_computation_arguments[index] = None
            get_logger().info(
                "__do_computation_with_hessian chunk size %s use time %s ms",
                len(chunk),
                counter.elapsed_milliseconds(),
            )

    def do_delayed_computation(self, index=None):
        if index is None:
            unfinished_keys = []
            for k, v in self.delayed_approximation_computations.items():
                if v:
                    unfinished_keys.append(k)

            if unfinished_keys:
                for (k, _) in self.__approx_hyper_gradient_mom_dict.iterate(
                    unfinished_keys
                ):
                    get_logger().debug(
                        "do delayed_approximation_computations for %s", k
                    )
                    self.do_delayed_computation(k)
            return

        if index not in self.delayed_approximation_computations:
            return
        if not self.delayed_approximation_computations[index]:
            return

        hyper_gradient = None
        mom_gradient = None
        if index in self.__approx_hyper_gradient_mom_dict:
            hyper_gradient, mom_gradient = self.__get_hyper_gradient_and_momentum(
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
        self.delayed_approximation_computations[index] = []
        self.__set_hyper_gradient_and_momentum(
            index, hyper_gradient, mom_gradient, use_approximation=True
        )

    @staticmethod
    def create_hypergradient_dict(
        cache_size,
        model=None,
        storage_dir=None,
        concat_momentum=True,
    ):
        mask = None
        gradient_shape = None
        if model is not None:
            model_util = ModelUtil(model)
            if model_util.is_pruned:
                get_logger().info(
                    "use pruned model, sparsity is %s", model_util.get_sparsity()[0]
                )
                parameters = model_util.get_parameter_list()
                gradient_shape = parameters.shape
                mask = model_util.get_pruning_mask_list()
                assert len(mask) == len(parameters)
        if mask is not None:
            if concat_momentum:
                mask = torch.cat((mask, mask))
                gradient_shape[1] *= 2
        tensor_dict = SyncedTensorDict.create(
            key_type=int,
            cache_size=cache_size,
            storage_dir=storage_dir,
            mask=mask,
            tensor_shape=gradient_shape,
        )
        return tensor_dict

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]

        assert len(batch) == 3
        instance_indices = {idx.data.item() for idx in batch[2]["index"]}

        batch_gradient_indices = instance_indices & self.__computed_indices

        if self.use_approximation:
            self.__approx_hyper_gradient_mom_dict.prefetch(batch_gradient_indices)
        self.__batch_gradient_indices = batch_gradient_indices

        # def _after_optimizer_step(self, **kwargs):

        assert self.__batch_gradient_indices == self.sample_gradient_dict.keys()

        if self.use_hessian:
            self.__hvp_function = get_hessian_vector_product_func(
                trainer.copy_model_with_loss(deepcopy=True), batch
            )
            self.hessian_computation_arguments = {}
        else:
            self.hessian_computation_arguments = None

        optimizer = trainer.get_optimizer()
        if not isinstance(optimizer, torch.optim.SGD):
            raise RuntimeError("optimizer is not SGD")

        cur_learning_rates = trainer.get_data("cur_learning_rates")
        assert len(cur_learning_rates) == 1
        cur_learning_rate = cur_learning_rates[0]
        batch_size = kwargs["batch_size"]

        momentums = [group["momentum"] for group in optimizer.param_groups]
        if len(momentums) != 1:
            raise RuntimeError("unsupported momentums")

        momentum = momentums[0]
        weight_decay = trainer.hyper_parameter.weight_decay
        training_set_size = len(trainer.dataset)

        for idx in self.__computed_indices:
            instance_gradient = None
            if idx in self.sample_gradient_dict:
                instance_gradient = (
                    (self.sample_gradient_dict[idx] * training_set_size / batch_size)
                    .detach()
                    .clone()
                )
            if self.use_hessian:
                self.hessian_computation_arguments[idx] = (
                    momentum,
                    weight_decay,
                    cur_learning_rate,
                    instance_gradient,
                )
            if self.use_approximation:
                self.delayed_approximation_computations[idx].append(
                    (momentum, weight_decay, cur_learning_rate, instance_gradient)
                )
        if self.use_hessian:
            self.__do_computation_with_hessian()
        if self.use_approximation:
            for idx in self.__computed_indices:
                if idx in self.sample_gradient_dict:
                    self.do_delayed_computation(idx)

    def __get_hyper_gradient_and_momentum(self, index, use_approximation):
        return self.__decode_hyper_gradient_and_momentum(
            self.__get_hyper_gradient_mom_dict(use_approximation)[index]
        )

    def __decode_hyper_gradient_and_momentum(self, tensor):
        return torch.split(tensor, tensor.shape[0] // 2)

    def __set_hyper_gradient_and_momentum(
        self, index, hyper_gradient, mom_gradient, use_approximation
    ):
        self.__get_hyper_gradient_mom_dict(use_approximation)[index] = torch.cat(
            (hyper_gradient, mom_gradient)
        )

    def get_hyper_gradient(self, index, use_approximation):
        return self.__get_hyper_gradient_and_momentum(index, use_approximation)[0]

    def __get_hyper_gradient_mom_dict(self, use_approximation):
        return (
            self.__approx_hyper_gradient_mom_dict
            if use_approximation
            else self.hessian_hyper_gradient_mom_dict
        )

    def __save_hyper_gradients(self, trainer, test_gradient, use_approximation):
        contribution = {}
        if use_approximation:
            get_logger().info("begin do do_delayed_computation")
            self.do_delayed_computation()
            get_logger().info("end do do_delayed_computation")
        tensor_dict = self.__get_hyper_gradient_mom_dict(use_approximation)
        training_set_size = len(trainer.dataset)
        for (index, value) in tensor_dict.iterate():
            hyper_gradient, _ = self.__decode_hyper_gradient_and_momentum(value)
            contribution[index] = (
                -(test_gradient @ hyper_gradient) / training_set_size
            ).data.item()
            tensor_dict[index] = hyper_gradient
        tensor_dict.flush_all(True)
        tensor_dict.release()
        if use_approximation:
            with open(
                os.path.join(
                    self.__get_save_dir(trainer), "approx_hydra_contribution.json"
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(contribution, f)
            hyper_gradient_dir = os.path.join(
                self.__get_save_dir(trainer), "approximation_hyper_gradient_dir"
            )
            shutil.move(tensor_dict.get_storage_dir(), hyper_gradient_dir)
        else:
            with open(
                os.path.join(
                    self.__get_save_dir(trainer), "hessian_hydra_contribution.json"
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(contribution, f)
            hyper_gradient_dir = os.path.join(
                self.__get_save_dir(trainer), "hessian_hyper_gradient_dir"
            )
            shutil.move(tensor_dict.get_storage_dir(), hyper_gradient_dir)
        with open(
            os.path.join(self.__get_save_dir(trainer), "training_set_size"), "wb"
        ) as f:
            pickle.dump(training_set_size, f)

    def foreach_hyper_gradient(self, use_approximation: bool, callback):
        if use_approximation:
            self.do_delayed_computation()
        hyper_gradient_mom_dict = self.__get_hyper_gradient_mom_dict(use_approximation)
        for (index, _) in hyper_gradient_mom_dict.iterate():
            hyper_gradient, _ = self.__get_hyper_gradient_and_momentum(
                index, use_approximation
            )
            callback(index, hyper_gradient)

    def foreach_approx_and_hessian_hyper_gradient(self, callback):
        assert self.use_approximation and self.use_hessian
        self.do_delayed_computation()
        hyper_gradient_mom_dict = self.__get_hyper_gradient_mom_dict(True)
        for (index, _) in hyper_gradient_mom_dict.iterate():
            approx_hyper_gradient, _ = self.__get_hyper_gradient_and_momentum(
                index, True
            )
            hessian_hyper_gradient, _ = self.__get_hyper_gradient_and_momentum(
                index, False
            )
            callback(index, approx_hyper_gradient, hessian_hyper_gradient)
