from collections.abc import MutableMapping
from typing import Generator

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.fs.tempdir import get_temp_dir
from cyy_naive_lib.log import get_logger
from cyy_torch_cpp_extension.data_structure import \
    SyncedTensorDict as SyncedTensorDict__


class SyncedTensorDict(MutableMapping):
    def __init__(self, tensor_dict, key_type=int):
        self.__tensor_dict = tensor_dict
        self.__key_type = key_type
        self.__iterated_keys = None

    def __contains__(self, key):
        return self.__tensor_dict.__contains__(str(key))

    def __getitem__(self, key):
        return self.__tensor_dict.__getitem__(str(key))

    def __setitem__(self, key, value):
        self.__tensor_dict.__setitem__(str(key), value.detach().cpu())

    def __delitem__(self, key):
        self.__tensor_dict.__delitem__(str(key))

    def get_storage_dir(self) -> str:
        return self.__tensor_dict.get_storage_dir()

    def flush(self) -> None:
        return self.__tensor_dict.flush(wait=True)

    def __len__(self):
        return len(self.__tensor_dict)

    def __iter__(self):
        in_memory_keys = self.__in_memory_keys()
        keys = self.__keys()
        self.__iterated_keys = list(in_memory_keys) + list(keys - in_memory_keys)
        self.prefetch(self.__iterated_keys[:10])
        return self

    def __next__(self):
        if not self.__iterated_keys:
            raise StopIteration()
        key = self.__iterated_keys[0]
        self.prefetch(self.__iterated_keys[:10])
        self.__iterated_keys = self.__iterated_keys[1:]
        return key

    def __keys(self) -> set:
        return {self.__eval_key(k) for k in self.__tensor_dict.keys()}

    def __in_memory_keys(self) -> set:
        return {self.__eval_key(k) for k in self.__tensor_dict.in_memory_keys()}

    def __eval_key(self, k):
        if self.__key_type is None:
            return eval(k)
        else:
            return self.__key_type(k)

    def prefetch(self, keys: list) -> None:
        self.__tensor_dict.prefetch([str(k) for k in keys])

    def __getattr__(self, name):
        return getattr(self.__tensor_dict, name)

    def iterate(self, keys: set = None) -> Generator:
        if keys is None:
            keys = set(self.__tensor_dict.keys())
        else:
            keys = {str(k) for k in keys}
        in_memory_keys = set(self.__in_memory_keys()) & keys
        for k in in_memory_keys:
            yield (self.__eval_key(k), self.__tensor_dict[k])
        remain_keys = list(keys - in_memory_keys)
        cache_size = self.__tensor_dict.get_in_memory_number()
        for chunk in split_list_to_chunks(remain_keys, cache_size // 2):
            self.__tensor_dict.prefetch(chunk)
            for k in chunk:
                yield (self.__eval_key(k), self.__tensor_dict[k])

    @classmethod
    def create(
        cls,
        storage_dir=None,
        key_type=int,
        cache_size=None,
    ):
        if storage_dir is None:
            storage_dir = get_temp_dir().name
            m = SyncedTensorDict__(storage_dir)
        else:
            m = SyncedTensorDict__(storage_dir)
            m.set_permanent_storage()
        if cache_size is not None:
            m.set_in_memory_number(cache_size)
        get_logger().info("tensor_dict use cache size %s", m.get_in_memory_number())
        return cls(tensor_dict=m, key_type=key_type)
