import torch

try:
    from cyy_torch_algorithm.data_structure.synced_tensor_dict import \
        SyncedTensorDict

    tensor_dict = SyncedTensorDict.create(cache_size=10)

    for i in range(100):
        tensor_dict[i] = torch.tensor([i])

    for (key, tensor) in tensor_dict.items():
        assert tensor == torch.tensor([key])

    for (key, tensor) in tensor_dict.iterate({"1", "2"}):
        assert 1 <= key <= 2
        assert tensor == torch.tensor([key])
    tensor_dict.flush()
except BaseException:
    pass
