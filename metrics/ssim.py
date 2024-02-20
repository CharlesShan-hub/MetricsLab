import kornia
import torch

###########################################################################################

__all__ = [
    'ssim',
    'ssim_approach_loss',
    'ssim_metric'
]

# https://kornia.readthedocs.io/en/latest/metrics.html#kornia.metrics.ssim
ssim = kornia.metrics.ssim

# https://kornia.readthedocs.io/en/latest/losses.html#kornia.losses.ssim_loss
def ssim_approach_loss(A: torch.Tensor, F: torch.Tensor,
    window_size: int = 11, max_val: float = 1.0,
    eps: float = 1e-12, reduction: str = 'mean', padding: str = 'same') -> torch.Tensor:
    return kornia.losses.ssim_loss(A, F, window_size, max_val, eps, reduction, padding)

# 与 VIFB 统一
def ssim_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    w0 = w1 = 0.5 # VIFB 忘了除二
    return torch.mean(w0 * ssim(A, F,window_size=11) + w1 * ssim(B ,F,window_size=11)) # 论文的窗大小就是 11

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

    print(f'SSIM(ir,ir):{torch.mean(ssim(ir,ir,window_size=11))}')
    print(f'SSIM(ir,fused):{torch.mean(ssim(ir,fused,window_size=11))}')
    print(f'SSIM(vis,fused):{torch.mean(ssim(vis,fused,window_size=11))}')

if __name__ == '__main__':
    main()
