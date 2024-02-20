import torch
import kornia

###########################################################################################

__all__ = [
    'sam',
    'sam_approach_loss',
    'sam_metric'
]

def sam(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Spectral Angle Mapper (SAM) between two spectral vectors.

    Args:
        src (torch.Tensor): The source spectral vector tensor.
        dst (torch.Tensor): The destination spectral vector tensor.

    Returns:
        torch.Tensor: The SAM value between the two input spectral vectors.
    """
    # 计算张量的转置
    src_T = src.transpose(0, 1)
    dst_T = dst.transpose(0, 1)

    # 计算点积
    val = torch.dot(src_T.flatten(), dst_T.flatten()) / (torch.norm(src) * torch.norm(dst))

    # 计算 SAM
    sam = torch.acos(val)

    return sam

def sam_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(sam(A,A)-sam(A,F))

def sam_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return sam(A,F)+sam(B,F)

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

    print(f'SAM:{sam_metric(vis, ir, fused)}')
    print(f'SAM:{sam_metric(vis, vis, vis)}')
    print(f'SAM:{sam_metric(vis, vis, fused)}')
    print(f'SAM:{sam_metric(vis, vis, ir)}')

if __name__ == '__main__':
    main()
