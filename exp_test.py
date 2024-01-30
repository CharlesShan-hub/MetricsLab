import torch
import torch.nn as nn
from torchviz import make_dot

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建一个模型实例
model = SimpleNet()

# 创建一个随机输入
input_tensor = torch.randn((1, 10))

# 调用 make_dot 可视化计算图
output_tensor = model(input_tensor)
make_dot(output_tensor, params=dict(model.named_parameters()))
