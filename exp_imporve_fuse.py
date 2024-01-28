from utils import *

ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
fuse_tensor = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)
target_tensor = ir_tensor

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from metrics import ssim_loss
from metrics import rmse_loss

# 定义优化器，使用 fuse 作为参数进行优化
optimizer = optim.Adam([fuse_tensor], lr=0.003)

# 训练参数
num_epochs = 400

# 每隔多少轮次打印一次图像
print_interval = 100

# 训练循环
for epoch in range(num_epochs):
    # 清零梯度
    optimizer.zero_grad()

    # 计算 SSIM 损失
    #loss = ssim_loss(fuse_tensor, target_tensor, window_size=11)
    loss = rmse_loss(fuse_tensor, target_tensor)

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印训练过程
    if epoch % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        plt.subplot(1,2,1)
        plt.imshow(grey_tensor_to_image(fuse_tensor),cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(grey_tensor_to_image(target_tensor),cmap='gray')
        plt.title(f'Epoch {epoch+1}')
        plt.show()

# 可视化最终生成的图像
plt.subplot(1,2,1)
plt.imshow(grey_tensor_to_image(fuse_tensor),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(grey_tensor_to_image(target_tensor),cmap='gray')
plt.title(f'Epoch {epoch+1}')
plt.show()
