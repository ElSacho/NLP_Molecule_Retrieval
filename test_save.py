import torch
test_tensor = torch.rand(10000, 10000)
torch.save(test_tensor, 'test_tensor.pt')