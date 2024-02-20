import torch
import kornia
# import matplotlib.pyplot as plt

###########################################################################################

__all__ = [
    'te',
    'te_approach_loss',
    'te_metric'
]

def te(image1: torch.Tensor, image2: torch.Tensor,
    q: float = 1.85, bandwidth: float = 0.1, eps: float = 1e-10,
    normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Tsallis entropy (TE) between two input images.

    Args:
        image1 (torch.Tensor): The first input image tensor.
        image2 (torch.Tensor): The second input image tensor.
        q (float, optional): The Tsallis entropy parameter. Default is 1.85.
        bandwidth (float, optional): Bandwidth for histogram smoothing. Default is 0.1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.
        normalize (bool, optional): Whether to normalize input images. Default is False.

    Returns:
        torch.Tensor: The Tsallis entropy between the two input images.
    """
    # 将图片拉平成一维向量,将一维张量转换为二维张量
    if normalize == True:
        x1 = ((image1-torch.min(image1))/(torch.max(image1) - torch.min(image1))).view(1,-1) * 255
        x2 = ((image2-torch.min(image2))/(torch.max(image2) - torch.min(image2))).view(1,-1) * 255
    else:
        x1 = image1.view(1,-1) * 255
        x2 = image2.view(1,-1) * 255

    # 定义直方图的 bins
    bins = torch.linspace(0, 255, 256).to(image1.device)

    # 计算二维直方图
    hist = kornia.enhance.histogram2d(x1, x2, bins, bandwidth=torch.tensor(bandwidth))

    # 计算边缘分布
    marginal_x = torch.sum(hist, dim=2)
    marginal_y = torch.sum(hist, dim=1)

    # plt.plot(marginal_x.squeeze().detach().numpy())
    # plt.show()
    # plt.plot(marginal_y.squeeze().detach().numpy())
    # plt.show()

    temp = marginal_x.unsqueeze(1) * marginal_y.unsqueeze(2) # 转置并广播
    mask = (temp > eps)
    temp2 = (temp[mask]) ** (q-1)
    temp1 = hist[mask] ** q
    # print(torch.sum(temp1),torch.sum(temp2))
    result = torch.sum(hist[mask] ** q / (temp[mask]) ** (q-1))

    return (1-result)/(1-q)

# 两张图一样，平均梯度会相等
def te_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(te(A,A)-te(A,F))

# 与 MEFB 统一
def te_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 1 # MEFB里边没有除 2
    q=1.85;     # Cvejic's constant
    return w0 * te(A, F, q, normalize=False) + w1 * te(B, F, q, normalize=False)

###########################################################################################

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis = to_tensor(Image.open('../imgs/TNO/vis/9.bmp')).unsqueeze(0)
    ir = to_tensor(Image.open('../imgs/TNO/ir/9.bmp')).unsqueeze(0)
    fused = to_tensor(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp')).unsqueeze(0)

    print(f'TE(ir,fused):{te(ir,fused)}')
    print(f'TE(vis,fused):{te(vis,fused)}') # 73.67920684814453 正确
    # print(f'TE(vis,fused):{te(vis,fused,normalize=True)}') # 48536.9453125错了
    print(f'TE(fused,fused):{te(fused,fused)}')
    print(f'TE(ir,ir):{te(ir,ir)}')
    print(f'TE(vis,vis):{te(vis,vis)}')
    print(f'TE_metric(ir,vis,fused):{te_metric(ir,vis,fused)}')

if __name__ == '__main__':
    main()
