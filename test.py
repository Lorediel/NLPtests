import torch

# create some example tensors
tensors = [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]
mask = [1, 0, 1]  # 1 indicates valid tensor, 0 indicates invalid tensor

# convert the list of tensors to a PyTorch tensor
tensors = torch.stack(tensors)
print(tensors.shape)

# create a boolean tensor from the mask
bool_mask = torch.tensor(mask, dtype=torch.bool)

# use boolean indexing to get the valid tensors
valid_tensors = tensors[bool_mask]
print(valid_tensors.shape)
