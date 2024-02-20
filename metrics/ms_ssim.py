import pytorch_msssim
import torch

###########################################################################################

__all__ = [
    'ms_ssim',
    'ms_ssim_approach_loss',
    'ms_ssim_metric'
]

def ms_ssim(X: torch.Tensor, Y: torch.Tensor,
    data_range: int = 1, size_average: bool = False) -> torch.Tensor:
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two tensors.
    https://github.com/VainF/pytorch-msssim

    Args:
        X (torch.Tensor): The first input tensor.
        Y (torch.Tensor): The second input tensor.
        data_range (int, optional): The dynamic range of the input images (usually 1 or 255). Default is 1.
        size_average (bool, optional): If True, take the average of the SSIM index. Default is False.

    Returns:
        torch.Tensor: The MS-SSIM value between the two input tensors.
    """
    return pytorch_msssim.ms_ssim(X, Y, data_range, size_average)

# https://github.com/VainF/pytorch-msssim
def ms_ssim_approach_loss(X: torch.Tensor, Y: torch.Tensor,
    data_range: int = 1, size_average: bool = False) -> torch.Tensor:
    return 1 - ms_ssim(X,Y,data_range,size_average)

def ms_ssim_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5
    return torch.mean(w0 * ms_ssim(A, F) + w1 * ms_ssim(B ,F))

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

    print(f'MSSIM(ir,ir):{torch.mean(ms_ssim(ir,ir))}')
    print(f'MSSIM(ir,fused):{torch.mean(ms_ssim(ir,fused))}')
    print(f'MSSIM(vis,fused):{torch.mean(ms_ssim(vis,fused))}')

if __name__ == '__main__':
    main()
