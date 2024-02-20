import torch

###########################################################################################

__all__ = [
    'snr',
    'snr_approach_loss',
    'snr_metric'
]

def snr(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Signal-to-Noise Ratio (SNR) for image fusion.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The SNR value for the fused image.
    """
    # 计算信号部分与噪声部分
    signal = torch.sum(A**2) + torch.sum(B**2)
    noise = torch.sum((A - F)**2) + torch.sum((B - F)**2)

    # 计算SNR值，防止MSE为零
    return 10 * torch.log10( signal / (noise + eps))

# 两张图完全一样，SNR 是无穷大
def snr_approach_loss(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return -snr(A, B, F)

def snr_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    # print(snr(A*255, B*255, F*255),snr(A, B, F)) # 结果一样，所以简化计算可以不乘 255
    return snr(A, B, F)

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

    print(f'[0-1]   SNR(ir,ir,ir):{snr(ir,ir,ir)}')
    print(f'[0-1]   SNR(vis,vis,vis):{snr(vis,vis,vis)}')
    print(f'[0-1]   SNR(ir,vis,fused):{snr(ir,vis,fused)}')
    print(f'[0-255] SNR(ir,ir,ir):{snr(ir*255,ir*255,ir*255)}')
    print(f'[0-255] SNR(vis,vis,vis):{snr(vis*255,vis*255,vis*255)}')
    print(f'[0-255] SNR(ir,vis,fused):{snr(ir*255,vis*255,fused*255)}')


if __name__ == '__main__':
    main()
