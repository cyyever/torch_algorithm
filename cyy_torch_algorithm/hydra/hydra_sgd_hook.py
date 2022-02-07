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
from cyy_torch_toolbox.ml_type import (MachineLearningPhase,
                                       ModelExecutorHookPoint)
from cyy_torch_toolbox.model_util import ModelUtil
from hessian_vector_product import get_hessian_vector_product_func
from sample_gradient.sample_gradient_hook import SampleGradientHook

from hydra.hydra_hook import HyDRAHook


class HyDRASGDHook(HyDRAHook):
    def get_hyper_gradient(self, index, use_approximation):
        return self._get_hyper_gradient_and_momentum(index, use_approximation)[0]

    def _decode_hyper_gradient_tensors(self, tensor):
        return torch.split(tensor, tensor.shape[0] // 2)
