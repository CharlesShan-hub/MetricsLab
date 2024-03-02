import torch
import kornia

###########################################################################################

__all__ = [
    'q',
    'q_approach_loss',
    'q_metric'
]

def q(X: torch.Tensor, Y: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Calculate the quality index between two images using the SSIM algorithm.
    See： http://live.ece.utexas.edu/research/Quality/zhou_research_anch/quality_index/demo.html

    Args:
        img1 (torch.Tensor): The first input image tensor.
        img2 (torch.Tensor): The second input image tensor.
        block_size (int, optional): The size of the blocks used in the calculation. Default is 8.

    Returns:
        torch.Tensor: The quality index between the two input images.

    Raises:
        ValueError: If the input images have different dimensions.

    Matlab Version:

    if (nargin == 1 | nargin > 3)
       quality = -Inf;
       quality_map = -1*ones(size(img1));
       return;
    end

    if (size(img1) ~= size(img2))
       quality = -Inf;
       quality_map = -1*ones(size(img1));
       return;
    end

    if (nargin == 2)
       block_size = 8;
    end

    N = block_size.^2;
    sum2_filter = ones(block_size);

    img1_sq   = img1.*img1;
    img2_sq   = img2.*img2;
    img12 = img1.*img2;

    img1_sum   = filter2(sum2_filter, img1, 'valid');
    img2_sum   = filter2(sum2_filter, img2, 'valid');
    img1_sq_sum = filter2(sum2_filter, img1_sq, 'valid');
    img2_sq_sum = filter2(sum2_filter, img2_sq, 'valid');
    img12_sum = filter2(sum2_filter, img12, 'valid');

    img12_sum_mul = img1_sum.*img2_sum;
    img12_sq_sum_mul = img1_sum.*img1_sum + img2_sum.*img2_sum;
    numerator = 4*(N*img12_sum - img12_sum_mul).*img12_sum_mul;
    denominator1 = N*(img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul;
    denominator = denominator1.*img12_sq_sum_mul;

    quality_map = ones(size(denominator));
    index = (denominator1 == 0) & (img12_sq_sum_mul ~= 0);
    quality_map(index) = 2*img12_sum_mul(index)./img12_sq_sum_mul(index);
    index = (denominator ~= 0);
    quality_map(index) = numerator(index)./denominator(index);

    quality = mean2(quality_map);
    """
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

    return torch.mean(quality_map)

def q_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1 - q(A,F)

def q_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5
    return w0 * q(A, F) + w1 * q(B, F)

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

    print(f'Q(vis,vis):{q(vis,vis)}')
    print(f'Q(vis,fused):{q(vis,fused)}')
    print(f'Q(vis,ir):{q(vis,ir)}')

    print(f'Q(vis,vis):{q(vis,vis,256)}')
    print(f'Q(vis,fused):{q(vis,fused,256)}')
    print(f'Q(vis,ir):{q(vis,ir,256)}')

    print(f'Q(vis,vis):{q(vis,vis,4)}')
    print(f'Q(vis,fused):{q(vis,fused,4)}')
    print(f'Q(vis,ir):{q(vis,ir,4)}')


    from torchmetrics.image import UniversalImageQualityIndex
    uqi = UniversalImageQualityIndex()
    print(uqi(vis*255, vis*255))
    print(uqi(vis*255, fused*255))
    print(uqi(vis*255, ir*255))

if __name__ == '__main__':
    main()
