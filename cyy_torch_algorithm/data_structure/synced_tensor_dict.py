from collections.abc import MutableMapping
from typing import Generator

from cyy_naive_lib.algorithm.sequence_op import split_list_to_chunks
from cyy_naive_lib.fs.tempdir import get_temp_dir
from cyy_naive_lib.log import get_logger

try:
    from cyy_torch_cpp_extension.data_structure import SyncedTensorDictIMPL

    class SyncedTensorDict(MutableMapping):
        def __init__(self, tensor_dict, key_type=int):
            self.__tensor_dict = tensor_dict
            self.__key_type = key_type
            self.__iterated_keys = None
            self.__cache_size = self.__tensor_dict.get_in_memory_number()
            self.__prefetch_size: int = 0

        def __contains__(self, key):
            return self.__tensor_dict.__contains__(str(key))

        def __getitem__(self, key):
            return self.__tensor_dict.__getitem__(str(key))

        def __setitem__(self, key, value):
            self.__tensor_dict.__setitem__(str(key), value.detach().cpu())

        def __delitem__(self, key):
            self.__tensor_dict.__delitem__(str(key))

        def __len__(self):
            return len(self.__tensor_dict)

        def __iter__(self):
            self.__iterated_keys = self.__keys()
            self.__prefetch_size = self.__cache_size
            return self

        def __next__(self):
            if not self.__iterated_keys:
                raise StopIteration()
            key = self.__iterated_keys[0]
            self.__iterated_keys = self.__iterated_keys[1:]
            self.__prefetch_size -= 1
            if self.__prefetch_size <= 0:
                self.__prefetch_size = self.__cache_size
                self.prefetch(self.__iterated_keys[: self.__prefetch_size])
            return key

        def __keys(self) -> list:
            return [self.__eval_key(k) for k in self.__tensor_dict.keys()]

        def __eval_key(self, k):
            if self.__key_type is None:
                return eval(k)
            return self.__key_type(k)

        def prefetch(self, keys: list) -> None:
            self.__tensor_dict.prefetch([str(k) for k in keys])

        def __getattr__(self, name):
            return getattr(self.__tensor_dict, name)

        def iterate(self, keys: set | list | None = None) -> Generator:
            if keys is None:
                keys = list(self.__tensor_dict.keys())
            else:
                keys = list(set(str(k) for k in keys))
            for chunk in split_list_to_chunks(keys, self.__cache_size // 2):
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
                impl = SyncedTensorDictIMPL(storage_dir)
            else:
                impl = SyncedTensorDictIMPL(storage_dir)
                impl.set_permanent_storage()
            if cache_size is not None:
                impl.set_in_memory_number(cache_size)
            get_logger().info(
                "tensor_dict use cache size %s", impl.get_in_memory_number()
            )
            return cls(tensor_dict=impl, key_type=key_type)

except BaseException:
    pass
