import torch
import kornia
from skimage.feature import graycomatrix
from typing import List

###########################################################################################

__all__ = [
    'asm',
    'asm_approach_loss',
    'asm_metric'
]

def asm(tensor: torch.Tensor, distances: List[int] = [1, 2], angles: List[int] = [0, 90]) -> torch.Tensor:
    """
    Calculate the Angular Second Moment (ASM) of a gray-scale image tensor.

    Args:
        tensor (torch.Tensor): The input gray-scale image tensor.
        distances (List[int], optional): List of pixel pair distances. Default is [1, 2].
        angles (List[int], optional): List of angles in degrees. Default is [0, 90].

    Returns:
        torch.Tensor: The ASM value.
    """
    # 转换为灰度图像
    if tensor.shape[1] == 3:
        tensor = kornia.color.rgb_to_grayscale(tensor)

    # 转换为 uint8 类型
    tensor = (tensor * 255).to(torch.uint8)

    asm_sum = 0
    # 计算每个角度和距离的 ASM，并将其加总
    for d in distances:
        for a in angles:
            m = graycomatrix(tensor.numpy().squeeze(), distances=distances, angles=angles, symmetric=True, normed=True)
            m = torch.tensor(m)
            asm_sum += torch.sum(m**2)

    # 计算 ASM 的平均值
    asm_mean = asm_sum / (len(distances)*len(angles))

    return torch.tensor(asm_mean)

def asm_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return torch.abs(asm(A)-asm(F))

def asm_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return asm(F)

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

    print(f'ASM:{asm_metric(vis, ir, fused)}')

if __name__ == '__main__':
    main()
