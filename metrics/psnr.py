import torch

###########################################################################################

__all__ = [
    'psnr',
    'psnr_approach_loss',
    'psnr_metric'
]

def psnr(A, B, F, MAX=1, eps=1e-10): # 改造成 VIFB 提出的用于融合的 PSNR
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) for image fusion.
    see: https://jason-chen-1992.weebly.com/home/-peak-single-to-noise-ratio

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        MAX (float, optional): The maximum possible pixel value. Default is 1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The PSNR value for the fused image.
    """
    # 计算两个输入图像与融合图像的均方误差（MSE）
    MSE1 = torch.mean((A - F)**2)
    MSE2 = torch.mean((B - F)**2)
    MSE = (MSE1+MSE2)/2.0

    # 计算PSNR值，防止MSE为零
    return 10 * torch.log10(MAX ** 2 / (MSE + eps))

# 两张图完全一样，PSNR 是无穷大
def psnr_approach_loss(A, B, F, MAX=1):
    return -psnr(A, B, F, MAX=MAX)

# 与 VIFB 统一
def psnr_metric(A, B, F):
    # w0 = w1 = 0.5
    # return w0 * psnr_kornia(imgF,imgA,max_val=1) + w1 * psnr_kornia(imgF,imgB,max_val=1)
    # return w0 * psnr_kornia(imgF*255,imgA*255,max_val=255) + w1 * psnr_kornia(imgF*255,imgB*255,max_val=255) # 为了与VIFB 统一
    # 发现原来 VIFB 里边的两个 MAE 竟然在一个 log 里边！所以不能分开算两个 PSNR 再求平均。
    return psnr(A, B, F)

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

    print(f'PSNR(ir,ir,ir):{psnr(ir,ir,ir)}')
    print(f'PSNR(ir,vis,fused):{psnr(ir,vis,fused)}')


if __name__ == '__main__':
    main()
