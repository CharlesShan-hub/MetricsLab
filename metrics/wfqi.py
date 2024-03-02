import torch
import kornia

###########################################################################################

__all__ = [
    'wfqi',
    'wfqi_approach_loss',
    'wfqi_metric'
]

def _q(X: torch.Tensor, Y: torch.Tensor, block_size: int = 8):
    if X.size() != Y.size():
        raise ValueError("Input images must have the same dimensions.")

    N = block_size**2
    mean_filter = torch.ones(1, 1, block_size, block_size).squeeze(0) / N

    XX = X**2
    YY = Y**2
    XY = X*Y

    mX = kornia.filters.filter2d(X, mean_filter, padding='valid')
    mY = kornia.filters.filter2d(Y, mean_filter, padding='valid')
    mXX = kornia.filters.filter2d(XX, mean_filter, padding='valid')
    mYY = kornia.filters.filter2d(YY, mean_filter, padding='valid')
    mXY = kornia.filters.filter2d(XY, mean_filter, padding='valid')

    mXmY = mX * mY
    sum_m2X_m2Y = mX**2 + mY**2

    numerator = 4 * (N * mXY - mXmY) * mXmY
    denominator1 = N * (mXX + mYY) - sum_m2X_m2Y
    denominator = denominator1 * sum_m2X_m2Y

    quality_map = torch.ones_like(denominator)
    index = (denominator1 == 0) & (sum_m2X_m2Y != 0)
    quality_map[index] = 2 * mXmY[index] / sum_m2X_m2Y[index]
    index = (denominator != 0)
    quality_map[index] = numerator[index] / denominator[index]

    return quality_map, mXX - mX**2 # 论文中采用方差代表显著值

def wfqi(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Calculates the Weighted Fusion Quality Index (WFQI) between two images A and B
    using the fusion map F. The images A and B are divided into blocks of size block_size
    for comparison.

    Args:
        A (Tensor): The first input image tensor.
        B (Tensor): The second input image tensor.
        F (Tensor): The fusion map tensor.
        block_size (int, optional): The size of the blocks for comparison. Default is 8.

    Returns:
        Tensor: The weighted fusion quality index.
    """
    map_AF, sA = _q(A,F,block_size)
    map_BF, sB = _q(B,F,block_size)
    sum_s = sA + sB
    r = torch.ones_like(sA) * 0.5
    c = sA / torch.sum(torch.max(sA,sB))
    index = (sum_s != 0)
    r[index] = sA[index] / sum_s[index]
    return torch.sum(c*(r*map_AF + (1-r)*map_BF))

def wfqi_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1 - wfqi(A,A,F)

def wfqi_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return wfqi(A,B,F)

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

    print(f'WFQI(vis,ir,fused):{wfqi(vis,ir,fused)}')
    print(f'WFQI(vis,vis,vis):{wfqi(vis,vis,vis)}')
    print(f'WFQI(vis,vis,fused):{wfqi(vis,vis,fused)}')
    print(f'WFQI(vis,vis,ir):{wfqi(vis,vis,ir)}')

if __name__ == '__main__':
    main()
