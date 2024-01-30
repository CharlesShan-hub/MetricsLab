import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchviz
import matplotlib.pyplot as plt
from utils import *

from metrics import ce_loss
from metrics import en_loss
from metrics import mi_loss
from metrics import psnr_loss
from metrics import ssim_loss
from metrics import rmse_loss
from metrics import ag_loss
from metrics import ei_loss
from metrics import sd_loss
from metrics import sf_loss
from metrics import q_abf
from metrics import q_cb

# 图片
ir_tensor = read_grey_tensor(dataset='TNO',category='ir',name='9.bmp',requires_grad=False)
vis_tensor = read_grey_tensor(dataset='TNO',category='vis',name='9.bmp',requires_grad=False)
fuse_tensor1 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)
fuse_tensor2 = read_grey_tensor(dataset='TNO',category='fuse',name='9.bmp',model='U2Fusion',requires_grad=True)

# Params
num_epochs = 5
learning_rate = 0.01
folder_name = 'APPROACH_Q_CB'
torch.manual_seed(42)
print_interval = 1

# Log
img_array_vis = []; img_array_ir = []
loss_array_vis = []; loss_array_ir = []
#metrics_array_vis = [];metrics_array_ir = []
img_array_vis.append(grey_tensor_to_image(fuse_tensor1))
img_array_ir.append(grey_tensor_to_image(fuse_tensor2))
#metrics_array.append(calculate_metrics(fuse))

# 定义优化器，使用 fuse 作为参数进行优化
optimizer_vis = optim.Adam([fuse_tensor1], lr=learning_rate)
optimizer_ir = optim.Adam([fuse_tensor2], lr=learning_rate)
#optimizer_vis = optim.SGD([fuse_tensor1], lr=learning_rate, momentum=0.9)
#optimizer_ir = optim.SGD([fuse_tensor2], lr=learning_rate, momentum=0.9)

# 训练循环
for epoch in range(num_epochs):
    # 清零梯度
    optimizer_vis.zero_grad()
    optimizer_ir.zero_grad()

    # 处理 nan 的情况
    #fuse_tensor1 = torch.nan_to_num(fuse_tensor1, nan=0.0)
    #fuse_tensor2 = torch.nan_to_num(fuse_tensor2, nan=0.0)

    # 计算损失
    #loss_vis = ssim_loss(fuse_tensor1, vis_tensor, window_size=11)
    #loss_ir = ssim_loss(fuse_tensor2, ir_tensor, window_size=11)
    #loss_vis = rmse_loss(fuse_tensor1, vis_tensor)
    #loss_ir = rmse_loss(fuse_tensor2, ir_tensor)
    #loss_vis = ce_loss(fuse_tensor1, vis_tensor)
    #loss_ir = ce_loss(fuse_tensor2, ir_tensor)
    #loss_vis = en_loss(fuse_tensor1)
    #loss_ir = en_loss(fuse_tensor2)
    #loss_vis = mi_loss(fuse_tensor1,vis_tensor)
    #loss_ir = mi_loss(fuse_tensor2,ir_tensor)
    #loss_vis = psnr_loss(fuse_tensor1,vis_tensor,max_val=1)
    #sloss_ir = psnr_loss(fuse_tensor2,ir_tensor,max_val=1)
    #loss_vis = ag_loss(fuse_tensor1)
    #loss_ir = ag_loss(fuse_tensor2)
    #loss_vis = ei_loss(fuse_tensor1)
    #loss_ir = ei_loss(fuse_tensor2)
    #loss_vis = sd_loss(fuse_tensor1)
    #loss_ir = sd_loss(fuse_tensor2)
    #loss_vis = sf_loss(fuse_tensor1)
    #loss_ir = sf_loss(fuse_tensor2)
    #loss_vis = q_abf(vis_tensor,vis_tensor,vis_tensor)-q_abf(vis_tensor,vis_tensor,fuse_tensor1)
    #loss_ir = q_abf(ir_tensor,ir_tensor,ir_tensor)-q_abf(ir_tensor,ir_tensor,fuse_tensor2)
    loss_vis = 1-q_cb(vis_tensor,vis_tensor,fuse_tensor1,mode='frequency') # 相同图片 Qcb 为 1
    loss_ir = 1-q_cb(ir_tensor,ir_tensor,fuse_tensor2,mode='frequency')

    # 反向传播 - Without Computation Graph
    # loss_vis.backward()
    # loss_ir.backward()
    # 反向传播 - With Computation Graph
    loss_vis.backward(retain_graph=True)
    loss_ir.backward()
    if epoch == 0:
        torchviz.make_dot(loss_vis).render(f'computation_graph', format='png')

    # 梯度检查
    # print(fuse_tensor1.grad)
    # plt.imshow(grey_tensor_to_image(fuse_tensor1.grad),cmap='gray')
    # plt.show()
    #print(torch.isnan(fuse_tensor1.grad).any().item())
    #print("Before clipping - Max Grad: {}, Min Grad: {}, Avg Grad: {}".format(
    #    torch.max(fuse_tensor1.grad), torch.min(fuse_tensor1.grad), torch.mean(fuse_tensor1.grad)
    #))
    #torch.autograd.gradcheck(loss_vis_, (vis_tensor, fuse_tensor1))

    # 更新参数
    optimizer_vis.step()
    optimizer_ir.step()

    # 打印训练过程
    if epoch % print_interval == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_vis.item()}, {loss_ir.item()}')
    else:
        print('.',end='')
    #metrics_array_vis.append(calculate_metrics(fuse_tensor1)
    img_array_vis.append(grey_tensor_to_image(fuse_tensor1))
    loss_array_vis.append(loss_vis.item())
    #metrics_array_ir.append(calculate_metrics(fuse_tensor2)
    img_array_ir.append(grey_tensor_to_image(fuse_tensor2))
    loss_array_ir.append(loss_ir.item())


# Save Images
save_grey_images(folder_name,img_array_vis,step=10,name='vis')
save_grey_images(folder_name,img_array_ir,step=10,name='ir')

# Save Video
video_grey_images(folder_name,img_array_vis,'FUSE_TO_VIS')
video_grey_images(folder_name,img_array_ir,'FUSE_TO_IR')

# Save Loss
plot_and_save_loss_with_lr(folder_name,loss_array_vis,learning_rate,'vis_loss.png','vis_loss_values.txt')
plot_and_save_loss_with_lr(folder_name,loss_array_ir,learning_rate,'ir_loss.png','ir_loss_values.txt')

# Save Metrics
#plot_metrics(folder_name,metrics_array)  

# Save Result Image
save_result(folder_name,img_array_vis[-1],img_array_ir[-1],vis_tensor,ir_tensor)

