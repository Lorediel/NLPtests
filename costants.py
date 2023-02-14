import torch

if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu" 
    device = torch.device(dev) 