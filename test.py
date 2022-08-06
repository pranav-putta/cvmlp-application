import torch

x = torch.rand((256, 256, 3)).float()
y = torch.rand((256, 256, 3)).float()

print(y - x)
