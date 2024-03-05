import torch
import kornia

###########################################################################################

__all__ = [
    'sf',
    'sf_approach_loss',
    'sf_metric'
]

def sf(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the standard frequency of a tensor.

    Args:
        tensor (torch.Tensor): Input tensor, assumed to be in the range [0, 1].
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The standard frequency of the input tensor.

    Reference:
        [1] A. M. Eskicioglu and P. S. Fisher, "Image quality measures and their performance,"
        IEEE Transactions on communications, vol. 43, no. 12, pp. 2959-2965, 1995.
    """
    # 使用 Sobel 算子计算水平和垂直梯度 - Old
    grad_x = kornia.filters.filter2d(tensor,torch.tensor([[1,  -1]]).unsqueeze(0),padding='valid')
    grad_y = kornia.filters.filter2d(tensor,torch.tensor([[1],[-1]]).unsqueeze(0),padding='valid')

    # 计算梯度的幅度
    return torch.sqrt(torch.mean(grad_x**2) + torch.mean(grad_y**2) + eps)

# 如果两幅图相等，SF 会一致
def sf_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(sf(A) - sf(F))

# 与 VIFB 统一
def sf_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return sf(F) * 255.0  # 与 VIFB 统一，需要乘 255

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

    print(f'SF(ir):{sf(ir)}')
    print(f'SF(vis):{sf(vis)}')
    print(f'SF(fused):{sf(fused)}')

if __name__ == '__main__':
    main()
