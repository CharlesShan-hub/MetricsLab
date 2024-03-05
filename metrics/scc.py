import torch
import kornia
from torchmetrics.functional.image import spatial_correlation_coefficient as scc
# https://lightning.ai/docs/torchmetrics/stable/image/spatial_correlation_coefficient.html

###########################################################################################

__all__ = [
    'scc',
    'scc_approach_loss',
    'scc_metric'
]

def scc_approach_loss(A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 1-scc(A,F)

def scc_metric(A: torch.Tensor, B: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    return 0.5 * scc(A, F) + 0.5 * scc(B, F)

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

    print(f'SCC(vis,ir):{scc(vis,ir)}')
    print(f'SCC(vis,fused):{scc(vis,fused)}')
    print(f'SCC(vis,vis):{scc(vis,vis)}')
    print(f'SCC_metric(vis,ir,fused):{scc_metric(vis,ir,fused)}')

if __name__ == '__main__':
    main()
