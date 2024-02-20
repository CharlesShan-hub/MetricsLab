import torch
import kornia

###########################################################################################

__all__ = [
    'q_c',
    'q_c_approach_loss',
    'q_c_metric'
]

def q_c(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor,
        window_size: int = 7, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the Q_C quality index for image fusion.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        window_size (int, optional): The size of the Gaussian kernel for filtering. Default is 7.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The Q_C quality index between the two input images and their fusion.
    """
    def ssim_yang(A,B): # SSIM_Yang
        C1 = 2e-16
        C2 = 2e-16
        kernel = kornia.filters.get_gaussian_kernel1d(window_size, 1.5, device=A.device, dtype=A.dtype)
        muA = kornia.filters.filter2d_separable(A, kernel, kernel, padding="valid")
        muB = kornia.filters.filter2d_separable(B, kernel, kernel, padding="valid")
        sAA = kornia.filters.filter2d_separable(A**2, kernel, kernel, padding="valid") - muA**2
        sBB = kornia.filters.filter2d_separable(B**2, kernel, kernel, padding="valid") - muB**2
        sAB = kornia.filters.filter2d_separable(A*B, kernel, kernel, padding="valid") - muA*muB
        ssim_map = ((2*muA*muB + C1)*(2*sAB + C2)) / ((muA**2 + muB**2 + C1)*(sAA + sBB + C2)+eps)
        return (ssim_map,sAB)

    (ssimAF, SAF) = ssim_yang(A*255, F*255)
    (ssimBF, SBF) = ssim_yang(B*255, F*255)
    ssimABF = SAF / (SAF+SBF+eps)
    Q_C = ssimABF*ssimAF + (1-ssimABF)*ssimBF
    Q_C[ssimABF>1] = 1
    Q_C[ssimABF<0] = 0
    return torch.mean(Q_C)

def q_c_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1-q_c(A, A, F, window_size=7)

def q_c_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_c(A, B, F, window_size=7)

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

    print(f'q_c_metric:{q_c(vis, ir, fused)}')
    print(f'q_c_metric:{q_c(vis, vis, vis)}')
    print(f'q_c_metric:{q_c(vis, vis, fused)}')
    print(f'q_c_metric:{q_c(vis, vis, ir)}')
    print(f'q_c_metric:{q_c(ir, ir, vis)}')
    print(f'q_c_metric:{q_c(ir, ir, fused)}')
    print(f'q_c_metric:{q_c(ir, ir, ir)}')

if __name__ == '__main__':
    main()
