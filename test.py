import torch

# create some example tensors
tensors = [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
mask = [1, 0, 1]  # 1 indicates valid tensor, 0 indicates invalid tensor

# mask the tensors
tensors = [tensor for tensor, m in zip(tensors, mask) if m]
print(tensors)
