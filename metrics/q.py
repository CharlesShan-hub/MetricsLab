import torch
import kornia

###########################################################################################

__all__ = [
    'q',
    'q_approach_loss',
    'q_metric'
]

def q(img1: torch.Tensor, img2: torch.Tensor, block_size: int = 8) -> torch.Tensor:
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
    """
    if img1.size() != img2.size():
        raise ValueError("Input images must have the same dimensions.")

    N = block_size**2
    sum2_filter = torch.ones(1, 1, block_size, block_size).squeeze(0) / N

    img1_sq = img1**2
    img2_sq = img2**2
    img12 = img1 * img2

    img1_sum = kornia.filters.filter2d(img1, sum2_filter, padding='valid')
    img2_sum = kornia.filters.filter2d(img2, sum2_filter, padding='valid')
    img1_sq_sum = kornia.filters.filter2d(img1_sq, sum2_filter, padding='valid')
    img2_sq_sum = kornia.filters.filter2d(img2_sq, sum2_filter, padding='valid')
    img12_sum = kornia.filters.filter2d(img12, sum2_filter, padding='valid')

    img12_sum_mul = img1_sum * img2_sum
    img12_sq_sum_mul = img1_sum**2 + img2_sum**2

    numerator = 4 * (N * img12_sum - img12_sum_mul) * img12_sum_mul
    denominator1 = N * (img1_sq_sum + img2_sq_sum) - img12_sq_sum_mul
    denominator = denominator1 * img12_sq_sum_mul

    quality_map = torch.ones_like(denominator)
    index = (denominator1 == 0) & (img12_sq_sum_mul != 0)
    quality_map[index] = 2 * img12_sum_mul[index] / img12_sq_sum_mul[index]
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

if __name__ == '__main__':
    main()
