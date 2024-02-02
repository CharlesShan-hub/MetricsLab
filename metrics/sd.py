import torch
import kornia

###########################################################################################

__all__ = [
    'sd',
    'sd_approach_loss',
    'sd_metric'
]

def sd(tensor): # 默认输入的是 0-1 的浮点数
    """
    Calculate the standard deviation of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor, assumed to be in the range [0, 1].

    Returns:
        torch.Tensor: The standard deviation of the input tensor.
    """
    return torch.sqrt(torch.mean((tensor - tensor.mean())**2)) * 255.0  # 与 VIFB 统一，需要乘 255

# 如果两幅图相等，SD 会一致
def sd_approach_loss(A, F):
    return torch.abs(sd(A) - sd(F))

# 与 VIFB 统一
def sd_metric(A, B, F):
    return sd(F)

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

    print(f'SD(ir):{sd(ir)}')
    print(f'SD(vis):{sd(vis)}')
    print(f'SD(fused):{sd(fused)}')

if __name__ == '__main__':
    main()
