import torch
import kornia

###########################################################################################

__all__ = [
    'eva',
    'eva_approach_loss',
    'eva_metric'
]

def eva(A: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the image by calculating the sum of the absolute mean values of specific kernels.

    Args:
        A (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The evaluation value.
    """
    corner = 1 / 2 ** 0.5
    border = 1.0
    center = -4*(corner+border)

    k1 = torch.tensor([[[corner,0,0],[0,-corner,0],[0,0,0]]])
    k2 = torch.tensor([[[0,0,corner],[0,-corner,0],[0,0,0]]])
    k3 = torch.tensor([[[0,0,0],[0,-corner,0],[corner,0,0]]])
    k4 = torch.tensor([[[0,0,0],[0,-corner,0],[0,0,corner]]])
    k5 = torch.tensor([[[0,border,0],[0,-border,0],[0,0,0]]])
    k6 = torch.tensor([[[0,0,0],[0,-border,0],[0,border,0]]])
    k7 = torch.tensor([[[0,0,0],[border,-border,0],[0,0,0]]])
    k8 = torch.tensor([[[0,0,0],[0,-border,border],[0,0,0]]])

    res = [torch.mean(torch.abs(kornia.filters.filter2d(A,kernel))) for kernel in [k1,k2,k3,k4,k5,k6,k7,k8]]

    return torch.sum(torch.stack(res))

def eva_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(eva(A)-eva(F))

def eva_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return eva(F) * 255.0

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

    print(f'EVA:{eva_metric(vis, ir, fused)}')

if __name__ == '__main__':
    main()
