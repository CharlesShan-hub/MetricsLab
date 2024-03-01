import torch

###########################################################################################

__all__ = [
    'mb',
    'mb_approach_loss',
    'mb_metric'
]

def mb(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean bias (MB).

    Args:
        A (torch.Tensor): Input tensor A.
        B (torch.Tensor): Input tensor B.
        F (torch.Tensor): Input tensor F.

    Returns:
        torch.Tensor: The edge intensity of the input tensors.
    """

    [mA, mB, mF] = [torch.mean(I) for I in [A, B, F]]

    return torch.abs(1-2*mF/(mA+mB)) # 我加的绝对值

# 如果两幅图相等，MB 会一致
def mb_approach_loss(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return mb(A, B, F)

def mb_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return mb(A, B, F) # 不乘 255 也一样

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

    print(f'MB(vis,ir,fuse):{mb(vis,ir,fused)}')

if __name__ == '__main__':
    main()
