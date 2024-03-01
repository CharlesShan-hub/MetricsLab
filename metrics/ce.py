import torch
import kornia

###########################################################################################

__all__ = [
    'ce',
    'ce_approach_loss',
    'ce_metric'
]

def ce(target: torch.Tensor, predict: torch.Tensor, bandwidth: float = 0.1, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the cross-entropy between the target and predicted histograms.

    Args:
        target (torch.Tensor): The target image tensor.
        predict (torch.Tensor): The predicted image tensor.
        bandwidth (float, optional): Bandwidth for histogram smoothing. Default is 0.1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The cross-entropy between the histograms of the target and predicted images.
    """
    # 将预测值和目标值缩放到范围[0, 255]
    predict = predict.view(1, -1) * 255
    target = target.view(1, -1) * 255

    # 创建用于直方图计算的区间
    bins = torch.linspace(0, 255, 256).to(predict.device)

    # 计算目标和预测图像的直方图
    h1 = kornia.enhance.histogram(target, bins=bins, bandwidth=torch.tensor(bandwidth))
    h2 = kornia.enhance.histogram(predict, bins=bins, bandwidth=torch.tensor(bandwidth))

    # 创建一个掩码以排除直方图中小于eps的值 - 这里是与 VIFB 统一的重点
    mask = (h1 > eps)&( h2 > eps)

    # 计算交叉熵
    return torch.sum(h1[mask] * torch.log2(h1[mask]/(h2[mask])))

# 如果两幅图片一样 ce 为 0
def ce_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return -ce(A, F)

# 与 VIFB 统一
def ce_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5
    return w0 * ce(A,F) + w1 * ce(B,F)

###########################################################################################

import numpy as np
import matplotlib.pyplot as plt

def visualize_image_entropy(size):
    # 生成完全覆盖所有像素的张量
    full_tensor = (torch.arange(256*(size**2/256)).view(1, 1, size, size) / (size**2/256)).to(torch.uint8)
    # 生成随机张量
    random_tensor = torch.randint(0, 256, size=(1, 1, size, size), dtype=torch.uint8)
    # 生成纯色张量
    white_tensor = torch.full((1, 1, size, size), 255, dtype=torch.uint8)
    grey_tensor = torch.full((1, 1, size, size), 127, dtype=torch.uint8)
    black_tensor = torch.full((1, 1, size, size), 0, dtype=torch.uint8)
    # 绘制图像
    tensor_colors = ['Uniform', 'Random', 'Grey', 'White', 'Black']
    tensor_list = [full_tensor, random_tensor, grey_tensor, white_tensor, black_tensor]
    entropy_list = [full_tensor_entropy, random_tensor_entropy, grey_tensor_entropy, white_tensor_entropy, black_tensor_entropy]

    fig, axs = plt.subplots(1, len(tensor_list), figsize=(20, 4))

    for i, (tensor, entropy) in enumerate(zip(tensor_list, entropy_list)):
        axs[i].imshow(tensor.view(size, size).numpy(), cmap='gray', vmin=0, vmax=255)
        axs[i].set_title(f'{tensor_colors[i]}\nEntropy: {entropy:.2f}')

    plt.show()

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis = to_tensor(Image.open('../imgs/TNO/vis/9.bmp')).unsqueeze(0)
    ir = to_tensor(Image.open('../imgs/TNO/ir/9.bmp')).unsqueeze(0)
    fused = to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)
    rand = torch.randint(0, 255, size=fused.shape, dtype=torch.uint8)/255.0

    size = 64
    # 生成完全覆盖所有像素的张量
    full_tensor = (torch.arange(256*(size**2/256)).view(1, 1, size, size) / (size**2/256)).to(torch.uint8)/255.0
    # 生成随机张量
    random_tensor = torch.randint(0, 256, size=(1, 1, size, size), dtype=torch.uint8)/255.0
    # 生成纯色张量
    white_tensor = torch.full((1, 1, size, size), 255, dtype=torch.uint8)/255.0
    grey_tensor = torch.full((1, 1, size, size), 127, dtype=torch.uint8)/255.0
    black_tensor = torch.full((1, 1, size, size), 0, dtype=torch.uint8)/255.0

    print("'Distance' with x and ir")
    print(f'CE(ir,ir):   {ce(ir,ir)}')
    print(f'CE(ir,vis):  {ce(ir,vis)}')
    print(f'CE(ir,fused):{ce(ir,fused)}')

    # print("\nIf fused is fused | ir | vis  | average | rand")
    # print(f'[Fused = fused]   CE(ir,fused)+CE(vis,fused):    {ce(ir,fused)+ce(vis,fused)}')
    # print(f'[Fused = ir]      CE(ir,ir)+CE(vis,ir):          {ce(ir,ir)+ce(vis,ir)}')
    # print(f'[Fused = vis]     CE(ir,vis)+CE(vis,vis):        {ce(ir,vis)+ce(vis,vis)}')
    # print(f'[Fused = average] CE(ir,arverge)+CE(vis,arverge):{ce(ir,(vis+ir)/2)+ce(vis,(vis+ir)/2)}')
    # print(f'[Fused = rand]    CE(ir,rand)+CE(vis,rand):      {ce(ir,rand)+ce(vis,rand)}')

    # print("\n")
    # print(f"{ce(full_tensor,full_tensor)}")
    # print(f"{ce(full_tensor,white_tensor)}")
    # print(f"{ce(full_tensor,black_tensor)}")
    # print(f"{ce(random_tensor,white_tensor)}")
    # print(f"{ce(random_tensor,black_tensor)}")
    # print(f"{ce(full_tensor,random_tensor)}")
    # print(f"{ce(white_tensor,black_tensor)}")

if __name__ == '__main__':
    main()
