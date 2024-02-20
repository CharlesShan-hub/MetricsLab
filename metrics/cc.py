import torch
import numpy as np

###########################################################################################

__all__ = [
    'cc','cc_tang',
    'cc_approach_loss',
    'cc_metric'
]

def cc(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Calculate the correlation coefficient (CC) between two input images and a fused image.

    Args:
        A (torch.Tensor): The first input image tensor.
        B (torch.Tensor): The second input image tensor.
        F (torch.Tensor): The fused image tensor.
        eps (float, optional): A small value to avoid numerical instability. Default is 1e-10.

    Returns:
        torch.Tensor: The correlation coefficient value.
    """
    A_mean = torch.mean(A)
    B_mean = torch.mean(B)
    F_mean = torch.mean(F)

    rAF = torch.sum((A - A_mean) * (F - F_mean)) / torch.sqrt(eps + torch.sum((A - A_mean) ** 2) * torch.sum((F - F_mean) ** 2))
    rBF = torch.sum((B - B_mean) * (F - F_mean)) / torch.sqrt(eps + torch.sum((B - B_mean) ** 2) * torch.sum((F - F_mean) ** 2))

    return torch.mean(torch.stack([rAF, rBF]))

def cc_tang(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """
    Compute the correlation coefficient between two variables A and B with respect to a third variable F.

    Parameters:
    - A (numpy.ndarray): Array representing variable A.
    - B (numpy.ndarray): Array representing variable B.
    - F (numpy.ndarray): Array representing variable F.

    Returns:
    - float: Correlation coefficient between A and F (rAF) and between B and F (rBF), averaged to get the final correlation coefficient (CC).

    Author: Linfeng Tang
    Reference: https://zhuanlan.zhihu.com/p/611295921

    This function calculates the correlation coefficients rAF and rBF between variables A and F, and B and F, respectively.
    It then computes the average of these coefficients to obtain the overall correlation coefficient (CC).

    The formula used for correlation coefficient calculation is based on the Pearson correlation coefficient formula.
    """
    # Calculate mean values for A, B, and F
    mean_A = np.mean(A)
    mean_B = np.mean(B)
    mean_F = np.mean(F)

    # Calculate correlation coefficients rAF and rBF
    rAF = np.sum((A - mean_A) * (F - mean_F)) / np.sqrt(np.sum((A - mean_A) ** 2) * np.sum((F - mean_F) ** 2))
    rBF = np.sum((B - mean_B) * (F - mean_F)) / np.sqrt(np.sum((B - mean_B) ** 2) * np.sum((F - mean_F) ** 2))

    # Calculate the average correlation coefficient CC
    return float(np.mean([rAF, rBF]))

def cc_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return cc(A,A,A) - cc(A,A,F)

# 与 Tang 统一
cc_metric = cc

###########################################################################################

def main():
    from torchvision import transforms
    from torchvision.transforms.functional import to_tensor
    from PIL import Image

    torch.manual_seed(42)

    transform = transforms.Compose([transforms.ToTensor()])

    vis_image = Image.open('../imgs/TNO/vis/9.bmp')
    vis_array = np.array(vis_image).astype(np.int32)
    vis_tensor = to_tensor(vis_image).unsqueeze(0)
    ir_image = np.array(Image.open('../imgs/TNO/ir/9.bmp'))
    ir_array = np.array(ir_image).astype(np.int32)
    ir_tensor = to_tensor(ir_image).unsqueeze(0)
    fused_image = np.array(Image.open('../imgs/TNO/fuse/U2Fusion/9.bmp'))
    fused_array = np.array(fused_image).astype(np.int32)
    fused_tensor = to_tensor(fused_image).unsqueeze(0)

    print(f'CC(ir,vis,fused) by self:{cc(ir_tensor,vis_tensor,fused_tensor)}')
    print(f'CC(ir,vis,fused) by Tang:{cc_tang(ir_array,vis_array,fused_array)}')

if __name__ == '__main__':
    main()
