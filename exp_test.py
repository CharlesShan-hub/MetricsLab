import torch

# 创建两个张量
tensor1 = torch.tensor([[[[1,1,1], [4,4,4]]]])
tensor2 = torch.tensor([[[[7, 8, 9], [10, 11, 12]]]])

# 在维度 0 上拼接两个张量
result = torch.cat((tensor1, tensor2), dim=-2)

print(result)
