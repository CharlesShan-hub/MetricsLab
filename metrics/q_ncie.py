import torch
import kornia

###########################################################################################

__all__ = [
    'q_ncie',
    'q_ncie_approach_loss',
    'q_ncie_metric'
]

def _mi(image1: torch.Tensor, image2: torch.Tensor,
    bandwidth: float = 0.1, eps: float = 1e-10,
    normalize: bool = False) -> torch.Tensor:
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

    # 计算互信息
    mask = (hist > eps)
    en_xy = -torch.sum(hist[mask] * torch.log2(hist[mask])) # MEFB里边用的 log2，不是 log(和 VIFB 相反! 服了)
    mask = (marginal_x != 0)
    en_x = -torch.sum(marginal_x[mask] * torch.log2(marginal_x[mask]))
    mask = (marginal_y != 0)
    en_y = -torch.sum(marginal_y[mask] * torch.log2(marginal_y[mask]))

    return (en_x + en_y - en_xy) / 8 # log2 256

def q_ncie(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor,
    bandwidth: float = 0.1, eps: float = 1e-10,
    normalize: bool = False) -> torch.Tensor:
    """
    Calculate the Non-Complementary Information Entropy (NCIE) quality index between two input images and their fusion.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        bandwidth (float, optional): Bandwidth for histogram smoothing. Default is 0.1.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.
        normalize (bool, optional): Whether to normalize input images. Default is False.

    Returns:
        torch.Tensor: The NCIE quality index between the two input images and their fusion.
    """
    # Calculate Normalized Cross-Correlation (NCC) between * and *
    NCC_AB = _mi(A,B,bandwidth,eps,normalize)
    NCC_AF = _mi(A,F,bandwidth,eps,normalize)
    NCC_BF = _mi(B,F,bandwidth,eps,normalize)

    # Create the correlation matrix
    R = torch.tensor([[1, NCC_AB, NCC_AF],
                      [NCC_AB, 1, NCC_BF],
                      [NCC_AF, NCC_BF, 1]])

    # Calculate the eigenvalues
    r, _ = torch.linalg.eig(R)

    # Calculate the HR quality index using the eigenvalues
    K = 3 # MEFB
    HR = torch.sum(r * torch.log2(r / K)) / K
    HR = -HR / 8 # torch.log2(b) , b=256

    # Return the Non-Complementary Information Entropy quality index
    return (1 - HR).real

def q_ncie_approach_loss(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return -q_ncie(A, B, F, bandwidth=0.1, eps=1e-10, normalize=False)

def q_ncie_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return q_ncie(A, B, F, bandwidth=0.1, eps=1e-10, normalize=False)

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

    print(f'Q_NCIE:{q_ncie(vis,ir,fused)}')
    print(f'Q_NCIE:{q_ncie(vis,vis,vis)}')
    print(f'Q_NCIE:{q_ncie(vis,vis,fused)}')
    print(f'Q_NCIE:{q_ncie(vis,vis,ir)}')
    print(f'Q_NCIE:{q_ncie(ir,ir,ir)}')
    print(f'Q_NCIE:{q_ncie(ir,ir,fused)}')
    print(f'Q_NCIE:{q_ncie(ir,ir,vis)}')

if __name__ == '__main__':
    main()
