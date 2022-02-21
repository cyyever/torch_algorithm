import json
import os
import pickle
import shutil

import torch
from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.log import get_logger
from cyy_naive_lib.time_counter import TimeCounter
from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
    SyncedTensorDict
from cyy_torch_algorithm.hessian_vector_product import \
    get_hessian_vector_product_func
from cyy_torch_algorithm.sample_gradient.sample_gradient_hook import \
    SampleGradientHook
from cyy_torch_toolbox.hook import Hook
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)


class HyDRAHook(Hook):
    def __init__(self, cache_size, **kwargs):
        super().__init__(stripable=True)
        self.__sample_gradient_hook = SampleGradientHook()
        self._cache_size = cache_size
        self.__save_dir = None
        self._trainer = None

        self._computed_indices = None
        self._delayed_approximation_computations: dict = None
        self._training_set_size = None
        self.__hyper_parameter_size = None

        self.use_hessian = kwargs.get("use_hessian", False)
        self._hvp_function = None
        self._hessian_hyper_gradient_dict = None
        self._hessian_computation_arguments = None
        self.use_approximation = kwargs.get("use_approximation", None)

        if self.use_approximation is None:
            self.use_approximation = not self.use_hessian

        self._approx_hyper_gradient_dict = None

    def _before_batch(self, **kwargs):
        trainer = kwargs["model_executor"]
        batch = kwargs["batch"]
        if self._trainer is None:
            self._trainer = trainer
        if self._training_set_size is None:
            self._training_set_size = len(trainer.dataset)

        if self.use_hessian:
            self._hvp_function = get_hessian_vector_product_func(
                trainer.copy_model_with_loss(deepcopy=True), batch
            )
            assert not self._hessian_computation_arguments
            self._hessian_computation_arguments = {}

    @property
    def sample_gradient_dict(self):
        return self.__sample_gradient_hook.sample_gradient_dict

    def get_save_dir(self, trainer=None):
        if self.__save_dir is None:
            self.__save_dir = os.path.join(trainer.save_dir, "HyDRA")
            os.makedirs(self.__save_dir, exist_ok=True)
        return self.__save_dir

    def _before_execute(self, **kwargs):
        trainer = kwargs["model_executor"]
        if not self._computed_indices:
            self._computed_indices = set(range(len(trainer.dataset)))
        else:
            get_logger().info("only compute %s indices", len(self._computed_indices))
        with open(
            os.path.join(self.get_save_dir(trainer), "tracking_indices.json"),
            mode="wb",
        ) as f:
            pickle.dump(self._computed_indices, f)
        if self.use_hessian:
            get_logger().info("use hessian to compute hyper-gradients")
            self._hessian_hyper_gradient_dict = HyDRAHook.create_hypergradient_dict(
                cache_size=self._cache_size,
                storage_dir=os.path.join(
                    self.get_save_dir(trainer),
                    "hessian_hyper_gradient_computation_dir",
                ),
            )
            get_logger().debug(
                "use hessian_hyper_gradient_mom_dir:%s",
                os.path.abspath(self._hessian_hyper_gradient_dict.get_storage_dir()),
            )
        if self.use_approximation:
            self._approx_hyper_gradient_dict = HyDRAHook.create_hypergradient_dict(
                cache_size=self._cache_size,
                storage_dir=os.path.join(
                    self.get_save_dir(trainer),
                    "approx_hyper_gradient_computation_dir",
                ),
            )
            get_logger().info(
                "use approx dict:%s",
                os.path.abspath(self._approx_hyper_gradient_dict.get_storage_dir()),
            )
            self._delayed_approximation_computations = {}
            trainer.prepend_named_hook(
                hook_point=ModelExecutorHookPoint.BEFORE_BATCH,
                name="prepare_hook",
                fun=self.__prepare_hook,
                stripable=True,
            )

    def set_computed_indices(self, computed_indices):
        self._computed_indices = set(computed_indices)
        self.__sample_gradient_hook.set_computed_indices(computed_indices)

    def _after_execute(self, **kwargs):
        get_logger().info("end hyper-gradient tracking")
        trainer = kwargs["model_executor"]
        trainer.remove_hook(name="prepare_hook")
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

    @classmethod
    def create_hypergradient_dict(
        cls,
        cache_size,
        storage_dir,
    ):
        tensor_dict = SyncedTensorDict.create(
            key_type=int,
            cache_size=cache_size,
            storage_dir=storage_dir,
        )
        return tensor_dict

    def __prepare_hook(self, **kwargs):
        batch = kwargs["batch"]
        if self.use_approximation:
            instance_indices = {idx.data.item() for idx in batch[2]["index"]}
            batch_gradient_indices: set = instance_indices & self._computed_indices
            if batch_gradient_indices:
                self._get_hyper_gradient_dict(self.use_approximation).prefetch(
                    batch_gradient_indices
                )

    def _set_hyper_gradient_tensors(self, index, use_approximation, *tensors):
        if self.__hyper_parameter_size is None:
            self.__hyper_parameter_size = tensors[0].shape[0]
        self._get_hyper_gradient_dict(use_approximation)[index] = torch.cat(tensors)

    def _decode_hyper_gradient_tensors(self, tensor):
        return torch.split(tensor, self.__hyper_parameter_size)

    def _get_hyper_gradient_tensors(self, index, use_approximation, none_num=1):
        data = self._get_hyper_gradient_dict(use_approximation)[index]
        if data is None:
            return (None,) * none_num
        return self._decode_hyper_gradient_tensors(data)

    def _get_hyper_gradient_dict(self, use_approximation):
        return (
            self._approx_hyper_gradient_dict
            if use_approximation
            else self._hessian_hyper_gradient_dict
        )

    def _do_all_delayed_computation(self):
        if self.use_approximation:
            delayed_keys = list(self._delayed_approximation_computations.keys())
            for chunk in split_list_to_chunks(delayed_keys, self._cache_size):

                self._get_hyper_gradient_dict(True).prefetch(chunk)
                for k in chunk:
                    get_logger().debug(
                        "do _delayed_approximation_computations for %s", k
                    )
                    self._do_delayed_computation(True, k)
            return

    def _do_computation_with_hessian(self):
        for chunk in split_list_to_chunks(
            list(self._computed_indices), self._cache_size
        ):
            hessian_vector_product_dict = self._get_hvp(chunk)
            for index in chunk:
                hessian_vector_product = hessian_vector_product_dict.get(index, None)
                if hessian_vector_product is not None:
                    hessian_vector_product = hessian_vector_product.to(
                        self._trainer.device
                    )
                    self._check_overflow_and_underflow(hessian_vector_product)
                self._do_delayed_computation(False, index, hessian_vector_product)

    def _check_overflow_and_underflow(self, tensor):
        if tensor is None:
            return
        if torch.any(torch.isnan(tensor)):
            get_logger().error("find nan tensor %s", tensor.cpu())
            assert False
        if torch.any(torch.isinf(tensor)):
            get_logger().error("find inf tensor %s", tensor.cpu())
            assert False

    def _optional_addition(self, *args):
        res = None
        for a in args:
            if a is None:
                continue
            if res is None:
                res = a
            else:
                res = res + a
        return res

    def _optional_multiplication(self, *args):
        res = None
        for a in args:
            if a is None:
                return None
            if res is None:
                res = a
            else:
                res = res * a
        return res

    def _optional_division(self, a, b):
        if a is None:
            return None
        return a / b

    def __save_hyper_gradients(self, trainer, test_gradient, use_approximation):
        contribution = {}
        get_logger().info("begin do _do_all_delayed_computation")
        self._do_all_delayed_computation()
        get_logger().info("end do _do_all_delayed_computation")
        tensor_dict = self._get_hyper_gradient_dict(use_approximation)
        test_gradient = test_gradient.cpu()
        for (index, value) in tensor_dict.iterate():
            hyper_gradient = self._decode_hyper_gradient_tensors(value)[0]
            contribution[index] = (
                -(test_gradient @ hyper_gradient.cpu()) / self._training_set_size
            ).data.item()
            tensor_dict[index] = hyper_gradient
        tensor_dict.tensor_dict.flush_all(True)
        if use_approximation:
            with open(
                os.path.join(
                    self.get_save_dir(trainer), "approx_hydra_contribution.json"
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(contribution, f)
            hyper_gradient_dir = os.path.join(
                self.get_save_dir(trainer), "approximation_hyper_gradient_dir"
            )
            shutil.move(tensor_dict.get_storage_dir(), hyper_gradient_dir)
        else:
            with open(
                os.path.join(
                    self.get_save_dir(trainer), "hessian_hydra_contribution.json"
                ),
                mode="wt",
                encoding="utf-8",
            ) as f:
                json.dump(contribution, f)
            hyper_gradient_dir = os.path.join(
                self.get_save_dir(trainer), "hessian_hyper_gradient_dir"
            )
            shutil.move(tensor_dict.get_storage_dir(), hyper_gradient_dir)
        tensor_dict.release()
        with open(
            os.path.join(self.get_save_dir(trainer), "training_set_size"), "wb"
        ) as f:
            pickle.dump(self._training_set_size, f)

    def _get_hvp(self, chunk):
        self._hessian_hyper_gradient_dict.prefetch(chunk)
        hyper_gradients = []
        hyper_gradient_indices = []
        hessian_vector_product_dict = {}
        for index in chunk:
            hyper_gradient = self.get_hyper_gradient(index, use_approximation=False)
            if hyper_gradient is not None:
                hyper_gradients.append(hyper_gradient)
                hyper_gradient_indices.append(index)
        if not hyper_gradients:
            return hessian_vector_product_dict
        with TimeCounter(log_prefix=f"hvp chunk size {len(hyper_gradients)}"):
            hessian_vector_products = self._hvp_function(hyper_gradients)
            assert len(hyper_gradients) == len(hessian_vector_products)
            hessian_vector_product_dict = dict(
                zip(hyper_gradient_indices, hessian_vector_products)
            )
            return hessian_vector_product_dict

    def get_hyper_gradient(self, index, use_approximation):
        return self._get_hyper_gradient_tensors(index, use_approximation)[0]

    def foreach_hyper_gradient(self, use_approximation: bool, callback):
        self._do_all_delayed_computation()
        approximation_hyper_gradient_dir = self._get_hyper_gradient_dict(
            use_approximation
        )
        for (index, _) in approximation_hyper_gradient_dir.iterate():
            hyper_gradient = self.get_hyper_gradient(index, use_approximation)
            callback(index, hyper_gradient)

    def foreach_approx_and_hessian_hyper_gradient(self, callback):
        assert self.use_approximation and self.use_hessian
        self._do_all_delayed_computation()
        approximation_hyper_gradient_dir = self._get_hyper_gradient_dict(True)
        for (index, _) in approximation_hyper_gradient_dir.iterate():
            approx_hyper_gradient = self.get_hyper_gradient(index, True)
            hessian_hyper_gradient = self.get_hyper_gradient(index, False)
            callback(index, approx_hyper_gradient, hessian_hyper_gradient)
