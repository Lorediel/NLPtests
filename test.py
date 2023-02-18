import torch

# create some example tensors
tensors = [torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)]

# stack the tensors
stacked = torch.stack(tensors)
print(stacked.shape)
#apply max pooling
max_pool = torch.nn.MaxPool1d(3)
x = max_pool(stacked)

print(x.shape)